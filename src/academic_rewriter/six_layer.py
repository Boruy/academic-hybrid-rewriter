"""A causal rewriting model with 2 LSTM, 2 Transformer and 2 Mamba blocks.

The source is a prefix, followed by the target. Only target tokens contribute to
the training loss. Embeddings and the tied vocabulary projection are not sequence
blocks. This model has no additional BART encoder or decoder.
"""
from dataclasses import asdict, dataclass
from itertools import islice
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers.modeling_outputs import CausalLMOutput

from .models import ReferenceMamba

NEURAL_DEPTHS = {'medium': 4, 'deep': 6}


@dataclass
class SixLayerConfig:
    vocab_size: int
    d_model: int = 192
    attention_heads: int = 4
    ffn_dim: int = 768
    dropout: float = 0.1
    max_positions: int = 258
    d_state: int = 8
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    depth: str = 'deep'

    def __post_init__(self):
        if self.depth not in NEURAL_DEPTHS:
            raise ValueError(f'Unknown neural rewrite depth: {self.depth}')
        if min(self.vocab_size, self.d_model, self.attention_heads, self.ffn_dim, self.d_state) < 1:
            raise ValueError('Model dimensions must be positive')
        if self.d_model % self.attention_heads or self.max_positions < 2:
            raise ValueError('Invalid attention heads or position capacity')
        if not 0 <= self.dropout < 1:
            raise ValueError('Dropout must be in [0, 1)')
        if any(not 0 <= token < self.vocab_size for token in
               [self.pad_token_id, self.bos_token_id, self.eos_token_id]):
            raise ValueError('Special token outside vocabulary')

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2) + '\n', encoding='utf8')

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding='utf8')))


class LSTMBlock(nn.Module):
    kind = 'lstm'

    def __init__(self, config):
        super().__init__()
        self.norm = nn.LayerNorm(config.d_model)
        self.rnn = nn.LSTM(config.d_model, config.d_model, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden, mask):
        packed = pack_padded_sequence(self.norm(hidden), mask.sum(1).cpu(), batch_first=True,
                                      enforce_sorted=False)
        output, _ = self.rnn(packed)
        output, _ = pad_packed_sequence(output, batch_first=True, total_length=hidden.size(1))
        return (hidden + self.dropout(output)) * mask.unsqueeze(-1)

    def step(self, hidden, valid, state):
        output, candidate = self.rnn(self.norm(hidden).unsqueeze(1), state)
        if state is None:
            state = tuple(torch.zeros_like(value) for value in candidate)
        state = tuple(torch.where(valid[None, :, None], new, old)
                      for new, old in zip(candidate, state))
        return (hidden + self.dropout(output[:, 0])) * valid[:, None], state


class TransformerBlock(nn.Module):
    kind = 'transformer'

    def __init__(self, config):
        super().__init__()
        self.norm = nn.LayerNorm(config.d_model)
        self.attention = nn.MultiheadAttention(config.d_model, config.attention_heads,
                                                dropout=config.dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(config.d_model)
        self.ffn = nn.Sequential(nn.Linear(config.d_model, config.ffn_dim), nn.GELU(),
                                 nn.Dropout(config.dropout), nn.Linear(config.ffn_dim, config.d_model))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden, mask):
        normalized = self.norm(hidden)
        causal = torch.ones(hidden.size(1), hidden.size(1), device=hidden.device,
                            dtype=torch.bool).triu(1)
        output, _ = self.attention(normalized, normalized, normalized, attn_mask=causal,
                                   key_padding_mask=~mask.bool(), need_weights=False)
        hidden = hidden + self.dropout(output)
        return (hidden + self.dropout(self.ffn(self.ffn_norm(hidden)))) * mask.unsqueeze(-1)

    def step(self, hidden, valid, state):
        query = self.norm(hidden).unsqueeze(1)
        keys = query if state is None else torch.cat([state[0], query], 1)
        mask = valid[:, None] if state is None else torch.cat([state[1], valid[:, None]], 1)
        output, _ = self.attention(query, keys, keys, key_padding_mask=~mask, need_weights=False)
        hidden = hidden + self.dropout(output[:, 0])
        hidden = (hidden + self.dropout(self.ffn(self.ffn_norm(hidden)))) * valid[:, None]
        return hidden, (keys, mask)


class MambaBlock(nn.Module):
    kind = 'mamba'

    def __init__(self, config):
        super().__init__()
        self.norm = nn.LayerNorm(config.d_model)
        self.mixer = ReferenceMamba(config.d_model, d_state=config.d_state)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden, mask):
        normalized = self.norm(hidden) * mask.unsqueeze(-1)
        return (hidden + self.dropout(self.mixer(normalized, mask))) * mask.unsqueeze(-1)

    def step(self, hidden, valid, state):
        """Same Mamba-1 recurrence as forward, with convolution and SSM caches."""
        mixer = self.mixer
        x, gate = mixer.in_proj(self.norm(hidden)).chunk(2, -1)
        if state is None:
            state = (x.new_zeros(x.size(0), mixer.inner, mixer.conv.kernel_size[0] - 1),
                     torch.zeros(x.size(0), mixer.inner, mixer.d_state,
                                 device=x.device, dtype=torch.float32))
        history, ssm = state
        window = torch.cat([history, x.unsqueeze(-1)], -1)
        convolved = (window * mixer.conv.weight[:, 0]).sum(-1) + mixer.conv.bias
        history = torch.where(valid[:, None, None], window[:, :, 1:], history)
        x = F.silu(convolved)
        delta, b, c = torch.split(mixer.x_proj(x), [mixer.rank, mixer.d_state, mixer.d_state], -1)
        dt = F.softplus(mixer.dt_proj(delta)).float()
        decay = torch.exp(dt[:, :, None] * -mixer.A_log.float().exp())
        candidate = decay * ssm + dt[:, :, None] * b[:, None, :].float() * x[:, :, None].float()
        ssm = torch.where(valid[:, None, None], candidate, ssm)
        output = (ssm * c[:, None, :].float()).sum(-1) + mixer.D.float() * x.float()
        output = mixer.out_proj(output.to(hidden.dtype) * F.silu(gate))
        return (hidden + self.dropout(output)) * valid[:, None], (history, ssm)


class SixLayerRewriter(nn.Module):
    layer_layout = ('lstm', 'lstm', 'transformer', 'transformer', 'mamba', 'mamba')

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model,
                                             padding_idx=config.pad_token_id)
        self.position_embedding = nn.Embedding(config.max_positions, config.d_model)
        self.segment_embedding = nn.Embedding(2, config.d_model)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([LSTMBlock(config), LSTMBlock(config),
                                     TransformerBlock(config), TransformerBlock(config),
                                     MambaBlock(config), MambaBlock(config)])
        self.active_layer_count = NEURAL_DEPTHS[config.depth]
        for layer in islice(self.layers, self.active_layer_count, None):
            layer.requires_grad_(False)
        self.output_norm = nn.LayerNorm(config.d_model)
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        with torch.no_grad():
            self.token_embedding.weight[config.pad_token_id].zero_()

    def sequence_logits(self, tokens, mask, segments=None):
        if tokens.size(1) > self.config.max_positions:
            raise ValueError('Combined source and target exceed position capacity')
        if not mask.bool().any(1).all() or torch.any(mask[:, 1:] > mask[:, :-1]):
            raise ValueError('Nonempty right-padded sequences required')
        segments = torch.zeros_like(tokens) if segments is None else segments
        positions = torch.arange(tokens.size(1), device=tokens.device)
        hidden = self.embedding_dropout(self.token_embedding(tokens) +
                                        self.position_embedding(positions)[None] +
                                        self.segment_embedding(segments)) * mask.unsqueeze(-1)
        for layer in islice(self.layers, self.active_layer_count):
            hidden = layer(hidden, mask)
        return F.linear(self.output_norm(hidden), self.token_embedding.weight)

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        if labels is None:
            return CausalLMOutput(logits=self.sequence_logits(input_ids, attention_mask))
        sources = [row[mask.bool()] for row, mask in zip(input_ids, attention_mask)]
        targets = [row[row != -100] for row in labels]
        if any(len(row) == 0 for row in sources + targets):
            raise ValueError('Source and target must be nonempty')
        lengths = [len(source) + len(target) for source, target in zip(sources, targets)]
        tokens = input_ids.new_full((len(sources), max(lengths)), self.config.pad_token_id)
        mask = torch.zeros_like(tokens)
        segments = torch.zeros_like(tokens)
        expected = tokens.new_full(tokens.shape, -100)
        for i, (source, target) in enumerate(zip(sources, targets)):
            boundary, end = len(source), lengths[i]
            tokens[i, :end] = torch.cat([source, target])
            mask[i, :end] = 1
            segments[i, boundary:end] = 1
            expected[i, boundary:end] = target
        logits = self.sequence_logits(tokens, mask, segments)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, self.config.vocab_size),
                               expected[:, 1:].reshape(-1), ignore_index=-100)
        return CausalLMOutput(loss=loss, logits=logits)

    def step(self, tokens, positions, segments, valid, states):
        hidden = self.embedding_dropout(self.token_embedding(tokens) +
                                        self.position_embedding(positions) +
                                        self.segment_embedding(segments)) * valid[:, None]
        updated = []
        for layer, state in zip(islice(self.layers, self.active_layer_count), states, strict=True):
            hidden, state = layer.step(hidden, valid, state)
            updated.append(state)
        return F.linear(self.output_norm(hidden), self.token_embedding.weight), updated

    @torch.no_grad()
    def generate(self, input_ids, attention_mask, max_new_tokens=128, num_beams=1,
                 do_sample=False, early_stopping=False, **kwargs):
        if num_beams != 1 or do_sample or max_new_tokens < 1:
            raise ValueError('Six-layer generation currently supports deterministic greedy decoding only')
        mask = attention_mask.bool()
        if not mask.any(1).all() or torch.any(mask[:, 1:] > mask[:, :-1]):
            raise ValueError('Nonempty right-padded sources required')
        lengths = mask.sum(1)
        if int(lengths.max()) + max_new_tokens > self.config.max_positions:
            raise ValueError('Requested generation exceeds position capacity')
        states = [None] * self.active_layer_count
        logits = self.token_embedding.weight.new_zeros(input_ids.size(0), self.config.vocab_size)
        positions = mask.long().cumsum(1) - 1
        for index in range(input_ids.size(1)):
            next_logits, states = self.step(input_ids[:, index], positions[:, index].clamp_min(0),
                                            torch.zeros_like(lengths), mask[:, index], states)
            logits = torch.where(mask[:, index, None], next_logits, logits)
        finished = torch.zeros_like(lengths, dtype=torch.bool)
        # Match the evaluator's start-token convention; this first token is excluded from counts.
        output = [torch.full_like(lengths, self.config.bos_token_id)]
        for index in range(max_new_tokens):
            logits[:, self.config.pad_token_id] = -torch.inf
            token = logits.argmax(-1).masked_fill(finished, self.config.pad_token_id)
            output.append(token)
            finished = finished | (token == self.config.eos_token_id)
            if finished.all():
                break
            logits, states = self.step(token, lengths + index, torch.ones_like(lengths),
                                        ~finished, states)
        return torch.stack(output, 1)
