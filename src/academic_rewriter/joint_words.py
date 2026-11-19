"""Single-word supervision and decoding for the joint six-layer model."""
from collections import defaultdict
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .lexical import LexicalEdit, _TOKEN, _WORD, _example, _replace_case, apply_word_edits


def word_replacements(rows):
    """Build candidate words from positive labels in the training split."""
    replacements = defaultdict(set)
    for row in rows:
        tokens, index, target = _example(row)
        source = tokens[index].group().lower()
        if source != target:
            replacements[source].add(target)
    if not replacements:
        raise ValueError('Joint training requires at least one single-word replacement')
    return {word: sorted(choices) for word, choices in sorted(replacements.items())}


@dataclass(frozen=True)
class WordContext:
    tokens: list[int]
    focus_start: int
    focus_end: int
    choices: tuple[int, ...]
    label: int | None = None


class WordTask:
    def __init__(self, model, tokenizer, device, max_length):
        if model.word_head is None:
            raise ValueError('Shallow word prediction requires a joint checkpoint')
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = min(max_length, model.config.max_positions)
        self.labels = model.config.word_labels
        self.label_ids = {word: index for index, word in enumerate(self.labels)}
        self.choices = {
            word: (0, *(self.label_ids[target] for target in replacements))
            for word, replacements in model.config.word_replacements.items()
        }

    def _context(self, sentence, span, label=None):
        start, end = span.span()
        left, word, right = self.tokenizer([sentence[:start], span.group(), sentence[end:]],
                                           add_special_tokens=False)['input_ids']
        remaining = self.max_length - len(word) - 2  # Reserve BOS and EOS.
        if not word or remaining < 0:
            raise ValueError(f'Word {span.group()!r} cannot fit in the input token budget')
        left_count = min(len(left), (remaining + 1) // 2)
        right_count = min(len(right), remaining - left_count)
        left_count = min(len(left), remaining - right_count)
        focus_start = 1 + left_count
        tokens = [self.model.config.bos_token_id, *left[len(left) - left_count:], *word,
                  *right[:right_count], self.model.config.eos_token_id]
        return WordContext(tokens, focus_start, focus_start + len(word),
                           self.choices.get(span.group().lower(), (0,)), label)

    def training_examples(self, rows):
        examples = []
        for row in rows:
            tokens, index, target = _example(row)
            span = tokens[index]
            label = 0 if span.group().lower() == target else self.label_ids[target]
            context = self._context(row['source'], span, label)
            if label not in context.choices:
                raise ValueError(f'Unexpected word target {target!r} for {span.group()!r}')
            examples.append(context)
        return examples

    def _logits(self, examples, *, refine=True):
        shape = (len(examples), max(len(example.tokens) for example in examples))
        tokens = torch.full(shape, self.model.config.pad_token_id, dtype=torch.long)
        mask = torch.zeros(shape, dtype=torch.long)
        segments = torch.zeros_like(mask)
        allowed = torch.zeros(len(examples), len(self.labels), dtype=torch.bool)
        for index, example in enumerate(examples):
            length = len(example.tokens)
            tokens[index, :length] = torch.tensor(example.tokens)
            mask[index, :length] = 1
            segments[index, example.focus_start:example.focus_end] = 1
            allowed[index, list(example.choices)] = True
        focus = torch.tensor([example.focus_end - 1 for example in examples], device=self.device)
        return self.model.word_logits(tokens.to(self.device), mask.to(self.device),
                                      segments.to(self.device), focus, allowed.to(self.device), refine=refine)

    def loss(self, examples):
        labels = torch.tensor([example.label for example in examples], device=self.device)
        return torch.stack([F.cross_entropy(logits, labels) for logits in self._logits(examples)]).mean()

    def _candidates(self, texts):
        for index, text in enumerate(texts):
            for span in _TOKEN.finditer(text):
                source = span.group()
                if (_WORD.fullmatch(source) and source.lower() in self.choices
                        and (source.islower() or source.istitle())):
                    yield index, span

    def guidance(self, texts, offsets):
        """Align source-only edit scores to subwords and highlight their local boundaries."""
        candidates = list(self._candidates(texts))
        if not candidates:
            return None
        rows = torch.tensor([index for index, _ in candidates], dtype=torch.long)
        starts = torch.tensor([span.start() for _, span in candidates]).unsqueeze(-1)
        ends = torch.tensor([span.end() for _, span in candidates]).unsqueeze(-1)
        token_starts, token_ends = offsets[rows].unbind(-1)
        aligned = (token_starts < ends) & (token_ends > starts) & (token_ends > token_starts)
        examples = [self._context(texts[index], span) for index, span in candidates]
        stages = self._logits(examples)
        edit_scores = torch.stack([1 - logits.softmax(-1)[:, 0] for logits in stages], dim=-1)
        weighted = aligned.to(self.device).unsqueeze(-1) * edit_scores.unsqueeze(1)
        guide = edit_scores.new_zeros((*offsets.shape[:2], len(stages)))
        guide = guide.index_add(0, rows.to(self.device), weighted).clamp(max=1)
        channels = guide.transpose(1, 2)
        dilated = F.max_pool1d(channels, kernel_size=3, stride=1, padding=1)
        eroded = -F.max_pool1d(-channels, kernel_size=3, stride=1, padding=1)
        guide = torch.maximum(guide, (dilated - eroded).transpose(1, 2))
        real_tokens = (offsets[..., 1] > offsets[..., 0]).to(self.device).unsqueeze(-1)
        return guide * real_tokens

    @torch.no_grad()
    def rewrite_many(self, texts):
        edits = [[] for _ in texts]
        candidates = list(self._candidates(texts))
        for start in range(0, len(candidates), 32):
            group = candidates[start:start + 32]
            examples = [self._context(texts[index], span) for index, span in group]
            (shallow,) = self._logits(examples, refine=False)
            for (index, span), prediction in zip(group, shallow.argmax(-1).tolist(), strict=True):
                if prediction:
                    target = _replace_case(span.group(), self.labels[prediction])
                    edits[index].append(LexicalEdit(span.start(), span.end(), span.group(), target))
        return [(apply_word_edits(text, changes), changes)
                for text, changes in zip(texts, edits, strict=True)]

    def rewrite(self, text):
        return self.rewrite_many([text])[0]
