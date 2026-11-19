"""A causal rewriting model with 2 LSTM, 2 Transformer and 2 Mamba blocks.

The source is a prefix, followed by the target. Only target tokens contribute to
the sentence loss. Embeddings and the tied vocabulary projection are not sequence
blocks. This model has no additional BART encoder or decoder.
"""
from dataclasses import asdict, dataclass, field
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
CHECKPOINT_DEPTHS = ('joint', *NEURAL_DEPTHS)
WORD_LAYERS = 2
FOUR_STAGE_LAYERS = (1, WORD_LAYERS, *NEURAL_DEPTHS.values())


def word_boundary_guide(logits, rows, spans, text_mask):
    """Scatter word edit probabilities to tokens and emphasize their local edges."""
    edit = 1 - logits.softmax(-1)[:, 0]
    weighted = spans * edit[:, None]
    guide = edit.new_zeros(text_mask.shape).index_add(0, rows, weighted).clamp(max=1)
    channels = guide.unsqueeze(1)
    dilated = F.max_pool1d(channels, kernel_size=3, stride=1, padding=1)
    eroded = -F.max_pool1d(-channels, kernel_size=3, stride=1, padding=1)
    return torch.maximum(guide, (dilated - eroded).squeeze(1)) * text_mask


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
    word_replacements: dict[str, list[str]] = field(default_factory=dict)
    prediction_stages: int = 3  # Layout of joint checkpoints saved before explicit stage versioning.

    def __post_init__(self):
        if self.depth not in CHECKPOINT_DEPTHS:
            raise ValueError(f'Unknown checkpoint depth: {self.depth}')
        if (self.depth == 'joint') != bool(self.word_replacements):
            raise ValueError('Joint checkpoints require a word replacement vocabulary')
        if self.prediction_stages not in (3, 4) or (self.prediction_stages == 4 and self.depth != 'joint'):
            raise ValueError('Four prediction stages require a joint checkpoint')
        if min(self.vocab_size, self.d_model, self.attention_heads, self.ffn_dim, self.d_state) < 1:
            raise ValueError('Model dimensions must be positive')
        if self.d_model % self.attention_heads or self.max_positions < 2:
            raise ValueError('Invalid attention heads or position capacity')
        if not 0 <= self.dropout < 1:
            raise ValueError('Dropout must be in [0, 1)')
        if any(not 0 <= token < self.vocab_size for token in
               [self.pad_token_id, self.bos_token_id, self.eos_token_id]):
            raise ValueError('Special token outside vocabulary')

    @property
    def word_labels(self):
        outcomes = {word for replacements in self.word_replacements.values() for word in replacements}
        return ('', *sorted(outcomes))  # Class zero retains the source word.

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
        joint = config.depth == 'joint'
        self.sentence_depths = tuple(NEURAL_DEPTHS) if joint else (config.depth,)
        self.active_layer_count = NEURAL_DEPTHS[self.sentence_depths[-1]]
        self.prediction_stage_layers = (FOUR_STAGE_LAYERS[-config.prediction_stages:]
                                        if joint else (self.active_layer_count,))
        self.word_guide_channels = {layer: index for index, layer in enumerate(self.prediction_stage_layers[:-1])}
        for layer in islice(self.layers, self.active_layer_count, None):
            layer.requires_grad_(False)
        self.output_norm = nn.LayerNorm(config.d_model)
        self.medium_output_norm = nn.LayerNorm(config.d_model) if joint else None
        self.word_head = (nn.Sequential(nn.LayerNorm(2 * config.d_model),
                                       nn.Linear(2 * config.d_model, len(config.word_labels)))
                          if joint else None)
        self.word_fusion = (nn.Sequential(nn.LayerNorm(config.d_model),
                                         nn.Linear(config.d_model, config.d_model, bias=False))
                            if joint else None)
        self.mid_predictions = (nn.ModuleDict({
            str(layer): nn.Sequential(nn.Linear(config.d_model, config.d_model), nn.GELU(),
                                      nn.Linear(config.d_model, config.d_model))
            for layer in self.prediction_stage_layers
        }) if config.prediction_stages == 4 else None)
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        with torch.no_grad():
            self.token_embedding.weight[config.pad_token_id].zero_()

    def _resolve_depth(self, depth):
        depth = self.sentence_depths[-1] if depth is None else depth
        if depth not in self.sentence_depths:
            raise ValueError(f'{depth} generation is unavailable for a {self.config.depth} checkpoint')
        return depth

    def _project(self, hidden, depth):
        norm = self.medium_output_norm if depth == 'medium' and self.medium_output_norm is not None else self.output_norm
        projected = self._prediction_features(hidden, NEURAL_DEPTHS[depth])
        return F.linear(norm(projected), self.token_embedding.weight)

    def _prediction_features(self, hidden, layer):
        if self.mid_predictions is None:
            return hidden
        return hidden + self.mid_predictions[str(layer)](hidden)

    def _embed_sequence(self, tokens, mask, segments=None):
        if tokens.size(1) > self.config.max_positions:
            raise ValueError('Combined source and target exceed position capacity')
        if not mask.bool().any(1).all() or torch.any(mask[:, 1:] > mask[:, :-1]):
            raise ValueError('Nonempty right-padded sequences required')
        segments = torch.zeros_like(tokens) if segments is None else segments
        positions = torch.arange(tokens.size(1), device=tokens.device)
        return self.embedding_dropout(self.token_embedding(tokens) +
                                      self.position_embedding(positions)[None] +
                                      self.segment_embedding(segments)) * mask.unsqueeze(-1)

    def _fuse_word_features(self, hidden, strength):
        return hidden + strength.unsqueeze(-1) * self.word_fusion(hidden)

    def _guide_stage(self, hidden, layer_index, word_guidance):
        if word_guidance is not None and layer_index in self.word_guide_channels:
            return self._fuse_word_features(hidden, word_guidance[..., self.word_guide_channels[layer_index]])
        return hidden

    def _encode(self, hidden, mask, start, stop, word_guidance=None):
        for index, layer in enumerate(islice(self.layers, start, stop), start):
            hidden = self._guide_stage(hidden, index, word_guidance)
            hidden = layer(hidden, mask)
        return hidden

    def _validate_guidance(self, tokens, word_guidance):
        expected = (*tokens.shape, len(self.word_guide_channels))
        if word_guidance is not None and (
                self.word_fusion is None or word_guidance.shape != expected):
            raise ValueError('Word guidance needs one score per input token and stage transition')

    def sequence_logits(self, tokens, mask, segments=None, *, depth=None, word_guidance=None):
        depth = self._resolve_depth(depth)
        self._validate_guidance(tokens, word_guidance)
        hidden = self._encode(self._embed_sequence(tokens, mask, segments), mask, 0,
                              NEURAL_DEPTHS[depth], word_guidance)
        return self._project(hidden, depth)

    def word_logits(self, tokens, mask, segments, focus, allowed, *, refine=True):
        """Read the separate word windows used by earlier joint checkpoints."""
        if self.word_head is None:
            raise ValueError('Word prediction requires a joint checkpoint')
        hidden = self._encode(self._embed_sequence(tokens, mask, segments), mask, 0, WORD_LAYERS)
        rows = torch.arange(tokens.size(0), device=tokens.device)
        context_end = mask.long().sum(1) - 1

        def scores(state):
            features = torch.cat((state[rows, focus], state[rows, context_end]), dim=-1)
            return self.word_head(features).masked_fill(~allowed, -torch.inf)

        shallow = scores(hidden)
        if not refine:
            return (shallow,)
        edit = 1 - shallow.softmax(-1)[:, 0]
        hidden = self._fuse_word_features(hidden, edit[:, None] * segments)
        medium = self._encode(hidden, mask, WORD_LAYERS, NEURAL_DEPTHS['medium'])
        return shallow, scores(medium)

    def source_word_stages(self, tokens, mask, rows, spans, allowed, text_mask, *, shallow=False):
        """Cascade word readouts and feedback over the same source backbone."""
        if self.mid_predictions is None:
            raise ValueError('Source word stages require a four-stage checkpoint')
        hidden = self._embed_sequence(tokens, mask)
        positions = torch.arange(tokens.size(1), device=tokens.device)
        focus = (spans.long() * positions).amax(dim=1)
        context_end = mask.long().sum(1)[rows] - 1
        readout_positions = torch.stack((focus, context_end), dim=1)
        layers = self.prediction_stage_layers[:-1]
        if shallow:
            layers = layers[:2]
        outputs = []
        guides = []
        start = 0
        for stop in layers:
            hidden = self._encode(hidden, mask, start, stop)
            local = hidden[rows[:, None], readout_positions]
            features = self._prediction_features(local, stop).flatten(1)
            logits = self.word_head(features).masked_fill(~allowed, -torch.inf)
            guide = word_boundary_guide(logits, rows, spans, text_mask)
            outputs.append(logits)
            guides.append(guide)
            if stop != layers[-1]:
                hidden = self._fuse_word_features(hidden, guide)
            start = stop
        return outputs, torch.stack(guides, dim=-1)

    def _labeled_outputs(self, input_ids, attention_mask, labels, depths, word_guidance):
        self._validate_guidance(input_ids, word_guidance)
        sources = [row[mask.bool()] for row, mask in zip(input_ids, attention_mask)]
        targets = [row[row != -100] for row in labels]
        if any(len(row) == 0 for row in sources + targets):
            raise ValueError('Source and target must be nonempty')
        lengths = [len(source) + len(target) for source, target in zip(sources, targets)]
        tokens = input_ids.new_full((len(sources), max(lengths)), self.config.pad_token_id)
        mask = torch.zeros_like(tokens)
        segments = torch.zeros_like(tokens)
        expected = tokens.new_full(tokens.shape, -100)
        guidance = (word_guidance.new_zeros((*tokens.shape, word_guidance.size(-1)))
                    if word_guidance is not None else None)
        for i, (source, target) in enumerate(zip(sources, targets)):
            boundary, end = len(source), lengths[i]
            tokens[i, :end] = torch.cat([source, target])
            mask[i, :end] = 1
            segments[i, boundary:end] = 1
            expected[i, boundary:end] = target
            if guidance is not None:
                guidance[i, :boundary] = word_guidance[i, attention_mask[i].bool()]
        hidden = self._embed_sequence(tokens, mask, segments)
        expected = expected[:, 1:].reshape(-1)
        outputs = {}
        start = 0
        for depth in depths:
            stop = NEURAL_DEPTHS[depth]
            hidden = self._encode(hidden, mask, start, stop, guidance)
            logits = self._project(hidden, depth)
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, self.config.vocab_size), expected,
                                   ignore_index=-100)
            outputs[depth] = CausalLMOutput(loss=loss, logits=logits)
            start = stop
        return outputs

    def sequence_losses(self, input_ids, attention_mask, labels, word_guidance=None):
        """Compute the sentence objectives with a single pass through the shared blocks."""
        outputs = self._labeled_outputs(input_ids, attention_mask, labels, self.sentence_depths, word_guidance)
        return {depth: output.loss for depth, output in outputs.items()}

    def forward(self, input_ids, attention_mask, labels=None, *, depth=None, word_guidance=None, **kwargs):
        depth = self._resolve_depth(depth)
        if labels is None:
            return CausalLMOutput(logits=self.sequence_logits(input_ids, attention_mask, depth=depth,
                                                            word_guidance=word_guidance))
        return self._labeled_outputs(input_ids, attention_mask, labels, (depth,), word_guidance)[depth]

    def step(self, tokens, positions, segments, valid, states, *, depth=None, word_guidance=None):
        depth = self._resolve_depth(depth)
        hidden = self.embedding_dropout(self.token_embedding(tokens) +
                                        self.position_embedding(positions) +
                                        self.segment_embedding(segments)) * valid[:, None]
        updated = []
        active = islice(self.layers, NEURAL_DEPTHS[depth])
        for index, (layer, state) in enumerate(zip(active, states, strict=True)):
            hidden = self._guide_stage(hidden, index, word_guidance)
            hidden, state = layer.step(hidden, valid, state)
            updated.append(state)
        return self._project(hidden, depth), updated

    @torch.no_grad()
    def generate(self, input_ids, attention_mask, max_new_tokens=128, num_beams=1,
                 do_sample=False, early_stopping=False, *, depth=None, word_guidance=None, **kwargs):
        depth = self._resolve_depth(depth)
        self._validate_guidance(input_ids, word_guidance)
        if num_beams != 1 or do_sample or max_new_tokens < 1:
            raise ValueError('Six-layer generation currently supports deterministic greedy decoding only')
        mask = attention_mask.bool()
        if not mask.any(1).all() or torch.any(mask[:, 1:] > mask[:, :-1]):
            raise ValueError('Nonempty right-padded sources required')
        lengths = mask.sum(1)
        if int(lengths.max()) + max_new_tokens > self.config.max_positions:
            raise ValueError('Requested generation exceeds position capacity')
        states = [None] * NEURAL_DEPTHS[depth]
        logits = self.token_embedding.weight.new_zeros(input_ids.size(0), self.config.vocab_size)
        positions = mask.long().cumsum(1) - 1
        for index in range(input_ids.size(1)):
            guide = word_guidance[:, index] if word_guidance is not None else None
            next_logits, states = self.step(input_ids[:, index], positions[:, index].clamp_min(0),
                                            torch.zeros_like(lengths), mask[:, index], states,
                                            depth=depth, word_guidance=guide)
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
                                        ~finished, states, depth=depth)
        return torch.stack(output, 1)
