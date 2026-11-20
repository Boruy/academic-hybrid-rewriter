"""Single-word supervision and decoding for the joint six-layer model."""
from collections import defaultdict
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .lexical import LexicalEdit, _TOKEN, _WORD, _example, _replace_case, apply_word_edits
from .six_layer import word_boundary_guide


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


def align_spans(offsets, starts, ends):
    token_start, token_end = offsets.unbind(-1)
    aligned = (token_start < ends) & (token_end > starts) & (token_end > token_start)
    complete = (aligned & (token_start <= starts)).any(-1) & (aligned & (token_end >= ends)).any(-1)
    return aligned, complete


class WordTask:
    def __init__(self, model, tokenizer, device, max_length):
        if model.word_head is None:
            raise ValueError('Shallow word prediction requires a joint checkpoint')
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = min(max_length, model.config.max_positions)
        self.four_stage = model.config.prediction_stages == 4
        self.labels = model.config.word_labels
        self.label_ids = {word: index for index, word in enumerate(self.labels)}
        self.choices = {
            word: (0, *(self.label_ids[target] for target in replacements))
            for word, replacements in model.config.word_replacements.items()
        }

    def _context(self, sentence, span, label=None):
        start, end = span.span()
        choices = self.choices.get(span.group().lower(), (0,))
        if self.four_stage:
            encoded = self.tokenizer(sentence, truncation=True, max_length=self.max_length,
                                     return_offsets_mapping=True)
            aligned, complete = align_spans(torch.tensor(encoded['offset_mapping']), start, end)
            if not complete.item():
                raise ValueError(f'Word {span.group()!r} is outside the input token budget')
            positions = aligned.nonzero().flatten().tolist()
            return WordContext(encoded['input_ids'], positions[0], positions[-1] + 1, choices, label)

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
        return WordContext(tokens, focus_start, focus_start + len(word), choices, label)

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

    def _allowed(self, choices):
        allowed = torch.zeros((len(choices), len(self.labels)), dtype=torch.bool)
        for index, options in enumerate(choices):
            allowed[index, list(options)] = True
        return allowed.to(self.device)

    def _logits(self, examples, *, refine=True):
        shape = (len(examples), max(len(example.tokens) for example in examples))
        tokens = torch.full(shape, self.model.config.pad_token_id, dtype=torch.long)
        mask = torch.zeros(shape, dtype=torch.long)
        segments = torch.zeros_like(mask)
        for index, example in enumerate(examples):
            length = len(example.tokens)
            tokens[index, :length] = torch.tensor(example.tokens)
            mask[index, :length] = 1
            segments[index, example.focus_start:example.focus_end] = 1
        allowed = self._allowed([example.choices for example in examples])
        tokens, mask, segments = (value.to(self.device) for value in (tokens, mask, segments))
        if self.four_stage:
            rows = torch.arange(len(examples), device=self.device)
            positions = torch.arange(tokens.size(1), device=self.device)
            text_mask = (positions > 0) & (positions < mask.sum(1, keepdim=True) - 1)
            logits, _ = self.model.source_word_stages(tokens, mask, rows, segments.bool(), allowed, text_mask)
            return logits
        focus = torch.tensor([example.focus_end - 1 for example in examples], device=self.device)
        return self.model.word_logits(tokens, mask, segments, focus, allowed, refine=refine)

    def losses(self, examples):
        labels = torch.tensor([example.label for example in examples], device=self.device)
        return [F.cross_entropy(logits, labels) for logits in self._logits(examples)]

    def _candidates(self, texts):
        for index, text in enumerate(texts):
            for span in _TOKEN.finditer(text):
                source = span.group()
                if (_WORD.fullmatch(source) and source.lower() in self.choices
                        and (source.islower() or source.istitle())):
                    yield index, span

    def _source_predictions(self, texts, encoding, *, shallow=False):
        offsets = encoding['offset_mapping']
        candidates = list(self._candidates(texts))
        if not candidates:
            return [], [], None
        rows = torch.tensor([index for index, _ in candidates], dtype=torch.long)
        starts = torch.tensor([span.start() for _, span in candidates]).unsqueeze(-1)
        ends = torch.tensor([span.end() for _, span in candidates]).unsqueeze(-1)
        aligned, complete = align_spans(offsets[rows], starts, ends)
        text_mask = (offsets[..., 1] > offsets[..., 0]).to(self.device)
        if self.four_stage:
            candidates = [candidates[index] for index in complete.nonzero().flatten().tolist()]
            if not candidates:
                return [], [], None
            allowed = self._allowed([self.choices[span.group().lower()] for _, span in candidates])
            logits, guides = self.model.source_word_stages(
                encoding['input_ids'].to(self.device), encoding['attention_mask'].to(self.device),
                rows[complete].to(self.device), aligned[complete].to(self.device), allowed, text_mask,
                shallow=shallow)
            return candidates, logits, guides

        examples = [self._context(texts[index], span) for index, span in candidates]
        logits = self._logits(examples)
        rows, aligned = rows.to(self.device), aligned.to(self.device)
        guides = torch.stack([word_boundary_guide(1 - output.softmax(-1)[:, 0], rows, aligned, text_mask)
                              for output in logits], dim=-1)
        return candidates, logits, guides

    def guidance(self, texts, encoding):
        """Build source-only boundary guides and aligned candidate vectors for the saved layout."""
        return self._source_predictions(texts, encoding)[2]

    @torch.no_grad()
    def rewrite_many(self, texts):
        if not texts:
            return []
        edits = [[] for _ in texts]
        if self.four_stage:
            encoding = self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_length,
                                      return_tensors='pt', return_offsets_mapping=True)
            candidates, outputs, _ = self._source_predictions(texts, encoding, shallow=True)
            predictions = outputs[-1].argmax(-1).tolist() if outputs else []
        else:
            candidates = list(self._candidates(texts))
            predictions = []
            for start in range(0, len(candidates), 32):
                group = candidates[start:start + 32]
                examples = [self._context(texts[index], span) for index, span in group]
                (shallow,) = self._logits(examples, refine=False)
                predictions.extend(shallow.argmax(-1).tolist())
        for (index, span), prediction in zip(candidates, predictions, strict=True):
            if prediction:
                target = _replace_case(span.group(), self.labels[prediction])
                edits[index].append(LexicalEdit(span.start(), span.end(), span.group(), target))
        return [(apply_word_edits(text, changes), changes)
                for text, changes in zip(texts, edits, strict=True)]

    def rewrite(self, text):
        return self.rewrite_many([text])[0]
