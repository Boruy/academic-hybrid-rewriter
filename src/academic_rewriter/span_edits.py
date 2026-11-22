"""Encode and apply edits at complete source word boundaries."""
from difflib import SequenceMatcher
from itertools import accumulate
import re

import torch

from .lexical import LexicalEdit


_WORD_UNITS = re.compile(r'\s*\S+|\s+\Z')


def word_units(text):
    """Split before whole words, retaining whitespace and attached punctuation."""
    return _WORD_UNITS.findall(text)


def encode_edit_program(source, reference, tokenizer, config, max_length):
    """Return BOS, ordered (start, stop, replacement) records, and EOS."""
    original, revised = word_units(source), word_units(reference)
    if len(original) >= config.edit_position_count:
        raise ValueError('Source exceeds the edit boundary capacity')
    special = set(tokenizer.all_special_ids)

    def record(start, stop, replacement):
        payload = tokenizer.encode(replacement, add_special_tokens=False)
        if special.intersection(payload) or tokenizer.decode(payload, clean_up_tokenization_spaces=False) != replacement:
            raise ValueError('Edit replacement requires lossless byte-level tokenization')
        return [config.text_vocab_size + start, config.text_vocab_size + stop, *payload]

    body = []
    alignment = SequenceMatcher(None, original, revised, autojunk=False)
    for kind, start, stop, first, last in alignment.get_opcodes():
        if kind != 'equal':
            body.extend(record(start, stop, ''.join(revised[first:last])))
    if len(body) > max_length:
        # A full-reference edit requires only two boundary IDs, even for a fragmented alignment.
        body = record(0, len(original), reference)
    if len(body) > max_length:
        raise ValueError('Reference exceeds the edit target budget')
    return [config.bos_token_id, *body, config.eos_token_id]


def edit_labels(rows, tokenizer, config, max_length):
    programs = [encode_edit_program(row['source'], row['target'], tokenizer, config, max_length) for row in rows]
    labels = torch.full((len(programs), max(map(len, programs))), -100, dtype=torch.long)
    for index, program in enumerate(programs):
        labels[index, :len(program)] = torch.tensor(program)
    return labels


def apply_edit_program(source, tokens, tokenizer, config):
    """Validate the complete program before copying source slices and inserting replacements."""
    end = len(tokens)
    while end and tokens[end - 1] == config.pad_token_id:
        end -= 1
    if end < 2 or tokens[0] != config.bos_token_id or tokens[end - 1] != config.eos_token_id:
        raise ValueError('Incomplete edit program')
    offsets = [0, *accumulate(map(len, word_units(source)))]
    base = config.text_vocab_size
    special = set(tokenizer.all_special_ids)
    edits = []
    next_start = 0
    cursor = 1
    while cursor < end - 1:
        if cursor + 1 >= end - 1:
            raise ValueError('Incomplete edit range')
        start, stop = tokens[cursor] - base, tokens[cursor + 1] - base
        if not next_start <= start <= stop < min(len(offsets), config.edit_position_count):
            raise ValueError('Edit ranges overlap or exceed the source')
        next_start = max(stop, start + 1)
        cursor += 2
        first = cursor
        while cursor < end - 1 and tokens[cursor] < base:
            cursor += 1
        payload = tokens[first:cursor]
        if any(token < 0 or token in special for token in payload):
            raise ValueError('Unexpected control token inside an edit')
        replacement = tokenizer.decode(payload, clean_up_tokenization_spaces=False)
        left, right = offsets[start], offsets[stop]
        if replacement != source[left:right]:
            edits.append(LexicalEdit(left, right, source[left:right], replacement))

    parts = []
    offset = 0
    for edit in edits:
        parts.extend((source[offset:edit.start], edit.target))
        offset = edit.end
    parts.append(source[offset:])
    pieces = [part for part in parts if part]
    if any(not (left[-1].isspace() or right[0].isspace()) for left, right in zip(pieces, pieces[1:])):
        raise ValueError('Edit would join text across a word boundary')
    prediction = ''.join(pieces)
    if source.strip() and not prediction.strip():
        raise ValueError('Edit program produced an empty sentence')
    return prediction, edits


class EditProgramDecoder:
    """Mask the output vocabulary to enforce ordered, nonoverlapping edit records."""

    def __init__(self, config, counts, device):
        self.config = config
        self.counts = torch.as_tensor(counts, dtype=torch.long, device=device)
        self.boundaries = torch.arange(config.edit_position_count, device=device)
        self.text_ids = torch.arange(config.text_vocab_size, device=device)
        self.start = torch.zeros_like(self.counts)
        self.next_start = torch.zeros_like(self.counts)
        self.await_stop = torch.zeros_like(self.counts, dtype=torch.bool)
        self.in_payload = torch.zeros_like(self.await_stop)

    def choose(self, logits, *, first, remaining):
        if first:
            return torch.full_like(self.counts, self.config.bos_token_id)
        base = self.config.text_vocab_size
        allow_text = self.in_payload[:, None] & (self.text_ids != self.config.pad_token_id)
        allow_text = allow_text & (self.text_ids != self.config.bos_token_id)
        allow_text[:, self.config.eos_token_id] = ~self.await_stop
        minimum = torch.where(self.await_stop, self.start, self.next_start)
        allow_range = (self.boundaries >= minimum[:, None]) & (self.boundaries <= self.counts[:, None])
        allow_range = allow_range & (self.await_stop | (remaining >= 3))[:, None]
        allowed = torch.cat((allow_text, allow_range), dim=-1)
        token = logits.masked_fill(~allowed, -torch.inf).argmax(-1)

        boundary = token - base
        is_boundary = boundary >= 0
        finishes_range = is_boundary & self.await_stop
        starts_range = is_boundary & ~self.await_stop
        self.next_start = torch.where(finishes_range, torch.maximum(boundary, self.start + 1), self.next_start)
        self.start = torch.where(starts_range, boundary, self.start)
        self.await_stop = torch.where(is_boundary, starts_range, self.await_stop)
        self.in_payload = torch.where(is_boundary, finishes_range, self.in_payload)
        return token
