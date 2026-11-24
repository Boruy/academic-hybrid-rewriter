"""A causal rewriting model with six blocks and front-end ablations.

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
from .span_edits import EditProgramDecoder

NEURAL_DEPTHS = {'medium': 4, 'deep': 6}
CHECKPOINT_DEPTHS = ('joint', *NEURAL_DEPTHS)
WORD_LAYERS = 2
FOUR_STAGE_LAYERS = (1, WORD_LAYERS, *NEURAL_DEPTHS.values())
RECURRENT_BLOCKS = ('lstm', 'rnn')
MIDDLE_BLOCKS = ('transformer', 'ffn')


def word_boundary_guide(edit, rows, spans, text_mask, *, include_positions=False):
    """Scatter word edit probabilities to tokens and emphasize their local edges."""
    weighted = spans * edit[:, None]
    guide = edit.new_zeros(text_mask.shape).index_add(0, rows, weighted).clamp(max=1)
    channels = guide.unsqueeze(1)
    dilated = F.max_pool1d(channels, kernel_size=3, stride=1, padding=1)
    eroded = -F.max_pool1d(-channels, kernel_size=3, stride=1, padding=1)
    strength = torch.maximum(guide, (dilated - eroded).squeeze(1)) * text_mask
    return (strength, guide * text_mask) if include_positions else strength


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
    fuse_stage_outputs: bool = False
    candidate_guidance: bool = False
    source_copy: bool = False
    contextual_edits: bool = False
    span_edits: bool = False
    recurrent_block: str = 'lstm'
    middle_block: str = 'transformer'

    def __post_init__(self):
        if self.recurrent_block not in RECURRENT_BLOCKS:
            raise ValueError(f'Unknown recurrent block: {self.recurrent_block}')
        if self.middle_block not in MIDDLE_BLOCKS:
            raise ValueError(f'Unknown middle block: {self.middle_block}')
        if self.depth not in CHECKPOINT_DEPTHS:
            raise ValueError(f'Unknown checkpoint depth: {self.depth}')
        if (self.depth == 'joint') != bool(self.word_replacements):
            raise ValueError('Joint checkpoints require a word replacement vocabulary')
        if self.prediction_stages not in (3, 4) or (self.prediction_stages == 4 and self.depth != 'joint'):
            raise ValueError('Four prediction stages require a joint checkpoint')
        if self.fuse_stage_outputs and self.prediction_stages != 4:
            raise ValueError('Final feature fusion requires four prediction stages')
        if self.source_copy and not self.fuse_stage_outputs:
            raise ValueError('Source copying requires the fused final prediction')
        if self.span_edits and not self.fuse_stage_outputs:
            raise ValueError('Span edits require the fused final prediction')
        if self.contextual_edits and not self.source_copy:
            raise ValueError('Contextual edit guidance requires source copying')
        if self.candidate_guidance and (self.prediction_stages != 4 or not self.replacement_labels):
            raise ValueError('Candidate guidance requires four prediction stages and replacement labels')
        if min(self.vocab_size, self.d_model, self.attention_heads, self.ffn_dim, self.d_state) < 1:
            raise ValueError('Model dimensions must be positive')
        if self.d_model % self.attention_heads or self.max_positions < 2:
            raise ValueError('Invalid attention heads or position capacity')
        if not 0 <= self.dropout < 1:
            raise ValueError('Dropout must be in [0, 1)')
        if any(not 0 <= token < self.text_vocab_size for token in
               [self.pad_token_id, self.bos_token_id, self.eos_token_id]):
            raise ValueError('Special token outside vocabulary')

    @property
    def edit_position_count(self):
        return self.max_positions // 2 if self.span_edits else 0

    @property
    def text_vocab_size(self):
        return self.vocab_size - self.edit_position_count

    @property
    def replacement_labels(self):
        outcomes = {word for replacements in self.word_replacements.values() for word in replacements}
        return tuple(sorted(outcomes))

    @property
    def word_labels(self):
        labels = ('', *self.replacement_labels)  # Class zero retains the source word.
        return (*labels, '[EDIT]') if self.contextual_edits else labels

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2) + '\n', encoding='utf8')

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding='utf8')))


class RecurrentBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.kind = config.recurrent_block
        self.norm = nn.LayerNorm(config.d_model)
        if self.kind == 'rnn':
            self.rnn = nn.RNN(config.d_model, config.d_model, nonlinearity='tanh', batch_first=True)
        else:
            self.rnn = nn.LSTM(config.d_model, config.d_model, batch_first=True)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden, mask):
        packed = pack_padded_sequence(self.norm(hidden), mask.sum(1).cpu(), batch_first=True,
                                      enforce_sorted=False)
        output, _ = self.rnn(packed)
        output, _ = pad_packed_sequence(output, batch_first=True, total_length=hidden.size(1))
        return (hidden + self.dropout(output)) * mask.unsqueeze(-1)

    def step(self, hidden, valid, state):
        output, candidate = self.rnn(self.norm(hidden).unsqueeze(1), state)
        update = valid[None, :, None]
        if isinstance(candidate, tuple):
            if state is None:
                state = tuple(torch.zeros_like(value) for value in candidate)
            state = tuple(torch.where(update, new, old) for new, old in zip(candidate, state, strict=True))
        else:
            previous = torch.zeros_like(candidate) if state is None else state
            state = torch.where(update, candidate, previous)
        return (hidden + self.dropout(output[:, 0])) * valid[:, None], state


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.kind = config.middle_block
        self.attention = None
        if self.kind == 'transformer':
            self.norm = nn.LayerNorm(config.d_model)
            self.attention = nn.MultiheadAttention(config.d_model, config.attention_heads,
                                                   dropout=config.dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(config.d_model)
        self.ffn = nn.Sequential(nn.Linear(config.d_model, config.ffn_dim), nn.GELU(),
                                 nn.Dropout(config.dropout), nn.Linear(config.ffn_dim, config.d_model))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden, mask):
        if self.attention is not None:
            normalized = self.norm(hidden)
            causal = torch.ones(hidden.size(1), hidden.size(1), device=hidden.device,
                                dtype=torch.bool).triu(1)
            output, _ = self.attention(normalized, normalized, normalized, attn_mask=causal,
                                       key_padding_mask=~mask.bool(), need_weights=False)
            hidden = hidden + self.dropout(output)
        return (hidden + self.dropout(self.ffn(self.ffn_norm(hidden)))) * mask.unsqueeze(-1)

    def step(self, hidden, valid, state):
        if self.attention is not None:
            query = self.norm(hidden).unsqueeze(1)
            keys = query if state is None else torch.cat([state[0], query], 1)
            mask = valid[:, None] if state is None else torch.cat([state[1], valid[:, None]], 1)
            output, _ = self.attention(query, keys, keys, key_padding_mask=~mask, need_weights=False)
            hidden = hidden + self.dropout(output[:, 0])
            state = (keys, mask)
        hidden = (hidden + self.dropout(self.ffn(self.ffn_norm(hidden)))) * valid[:, None]
        return hidden, state


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


class SourceCopyHead(nn.Module):
    """Mix vocabulary generation with a distribution over source token positions."""

    def __init__(self, config):
        super().__init__()
        self.query = nn.Linear(config.d_model, config.d_model, bias=False)
        self.key = nn.Linear(config.d_model, config.d_model, bias=False)
        self.gate = nn.Linear(config.d_model, 1)
        self.scale = config.d_model ** -0.5
        self.special_ids = (config.pad_token_id, config.bos_token_id, config.eos_token_id)

    def copyable(self, tokens, mask, segments):
        allowed = mask.bool()
        if segments is not None:
            allowed = allowed & (segments == 0)
        for token_id in self.special_ids:
            allowed = allowed & (tokens != token_id)
        return allowed

    def forward(self, features, logits, keys, tokens, allowed, source_bias=None, *, copy_scale=1.):
        has_source = allowed.any(dim=-1, keepdim=True)
        scores = (self.query(features) @ keys.transpose(-1, -2)) * self.scale
        if source_bias is not None:
            scores = scores + source_bias[:, None, :]
        scores = scores.masked_fill(~allowed, -torch.inf).masked_fill(~has_source, 0.)
        attention = scores.softmax(dim=-1, dtype=torch.float32).masked_fill(~allowed, 0.)
        gate = self.gate(features).float()
        copy_weight = gate.sigmoid()
        vocabulary_weight = (-gate).sigmoid()
        if copy_scale != 1.:
            # Return the removed pointer mass to vocabulary generation.
            vocabulary_weight = vocabulary_weight + (1. - copy_scale) * copy_weight
            copy_weight = copy_weight * copy_scale
        vocabulary = logits.softmax(dim=-1, dtype=torch.float32) * vocabulary_weight.masked_fill(~has_source, 1.)
        # Repeated source tokens share one vocabulary entry and add their mass.
        token_ids = tokens[:, None, :].expand_as(attention)
        combined = vocabulary.scatter_add(-1, token_ids, copy_weight * attention)
        return combined.clamp_min(torch.finfo(combined.dtype).tiny).log()


class SixLayerRewriter(nn.Module):
    input_embedding_init_std = 0.02

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model,
                                             padding_idx=config.pad_token_id)
        self.position_embedding = nn.Embedding(config.max_positions, config.d_model)
        self.segment_embedding = nn.Embedding(2, config.d_model)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([RecurrentBlock(config), RecurrentBlock(config),
                                     TransformerBlock(config), TransformerBlock(config),
                                     MambaBlock(config), MambaBlock(config)])
        self.layer_layout = tuple(layer.kind for layer in self.layers)
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
        # Start with the position guide alone. Only replacement labels have embeddings.
        self.word_candidate_embeddings = (
            nn.Parameter(torch.zeros(len(config.replacement_labels), config.d_model))
            if config.candidate_guidance else None)
        self.mid_predictions = (nn.ModuleDict({
            str(layer): nn.Sequential(nn.Linear(config.d_model, config.d_model), nn.GELU(),
                                      nn.Linear(config.d_model, config.d_model))
            for layer in self.prediction_stage_layers
        }) if config.prediction_stages == 4 else None)
        self.final_fusion = (nn.Linear(len(self.prediction_stage_layers) * config.d_model, config.d_model)
                             if config.fuse_stage_outputs else None)
        nn.init.normal_(self.token_embedding.weight, std=self.input_embedding_init_std)
        with torch.no_grad():
            # nn.Embedding already drew unit-normal values. Rescale the existing draws
            # so all three additive inputs start at the same scale for a given seed.
            self.position_embedding.weight.mul_(self.input_embedding_init_std)
            self.segment_embedding.weight.mul_(self.input_embedding_init_std)
            self.token_embedding.weight[config.pad_token_id].zero_()
        self.copy_head = SourceCopyHead(config) if config.source_copy else None

    def _resolve_depth(self, depth):
        depth = self.sentence_depths[-1] if depth is None else depth
        if depth not in self.sentence_depths:
            raise ValueError(f'{depth} generation is unavailable for a {self.config.depth} checkpoint')
        return depth

    def _project(self, hidden, depth, stage_features=None):
        norm = self.medium_output_norm if depth == 'medium' and self.medium_output_norm is not None else self.output_norm
        layer = NEURAL_DEPTHS[depth]
        if self.final_fusion is not None and depth == 'deep':
            combined = torch.cat([stage_features[key] for key in self.prediction_stage_layers], dim=-1)
            projected = self.final_fusion(combined)
        elif stage_features is not None:
            projected = stage_features[layer]
        else:
            projected = self._prediction_features(hidden, layer)
        features = norm(projected)
        vocabulary = self.token_embedding.weight
        if self.config.span_edits and depth != 'deep':
            vocabulary = vocabulary[:self.config.text_vocab_size]
        return features, F.linear(features, vocabulary)

    def _copy_bias(self, word_guidance):
        if not self.config.contextual_edits or word_guidance is None:
            return None
        # Use the final unexpanded edit weights so a nearby edit does not penalize copying a neighbor.
        return -word_guidance[..., -1, -1]

    def _sequence_projection(self, hidden, depth, stage_features, tokens, mask, segments, word_guidance):
        features, logits = self._project(hidden, depth, stage_features)
        if self.copy_head is None or depth != 'deep':
            return logits
        source = self.copy_head.copyable(tokens, mask, segments)
        positions = torch.arange(tokens.size(1), device=tokens.device)
        causal = positions[:, None] >= positions
        return self.copy_head(features, logits, self.copy_head.key(features), tokens,
                              source[:, None, :] & causal, self._copy_bias(word_guidance))

    def _prediction_features(self, hidden, layer):
        if self.mid_predictions is None:
            return hidden
        return hidden + self.mid_predictions[str(layer)](hidden)

    def _record_stage_features(self, stage_features, hidden, layer):
        if stage_features is not None and layer in self.prediction_stage_layers:
            stage_features[layer] = self._prediction_features(hidden, layer)

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

    def _make_word_guide(self, logits, rows, spans, text_mask):
        probabilities = logits.softmax(-1)
        location = word_boundary_guide(1 - probabilities[:, 0], rows, spans, text_mask,
                                      include_positions=self.config.contextual_edits)
        strength, positions = location if self.config.contextual_edits else (location, None)
        if self.word_candidate_embeddings is None:
            return strength if positions is None else torch.stack((strength, positions), dim=-1)

        # Broadcast the probability-weighted replacement vector to the aligned subwords.
        candidate_count = self.word_candidate_embeddings.size(0)
        words = probabilities[:, 1:1 + candidate_count] @ self.word_candidate_embeddings
        word_indices, token_indices = (spans & text_mask[rows]).nonzero(as_tuple=True)
        flat_indices = rows[word_indices] * text_mask.size(1) + token_indices
        candidates = words.new_zeros(text_mask.numel(), self.config.d_model).index_add(
            0, flat_indices, words[word_indices])
        candidates = candidates.reshape(*text_mask.shape, self.config.d_model)
        features = [strength.unsqueeze(-1), candidates]
        if positions is not None:
            features.append(positions.unsqueeze(-1))
        return torch.cat(features, dim=-1)

    def _fuse_word_features(self, hidden, guide):
        if guide.ndim < hidden.ndim:
            return hidden + guide.unsqueeze(-1) * self.word_fusion(hidden)
        hidden = hidden + guide[..., :1] * self.word_fusion(hidden)
        if self.word_candidate_embeddings is not None:
            hidden = hidden + guide[..., 1:1 + self.config.d_model]
        return hidden

    def _guide_stage(self, hidden, layer_index, word_guidance):
        if word_guidance is not None and layer_index in self.word_guide_channels:
            # The transition axis follows the batch and optional sequence axes.
            guide = word_guidance.select(hidden.ndim - 1, self.word_guide_channels[layer_index])
            return self._fuse_word_features(hidden, guide)
        return hidden

    def _encode(self, hidden, mask, start, stop, word_guidance=None, *, stage_features=None):
        for index, layer in enumerate(islice(self.layers, start, stop), start):
            hidden = self._guide_stage(hidden, index, word_guidance)
            hidden = layer(hidden, mask)
            self._record_stage_features(stage_features, hidden, index + 1)
        return hidden

    def _validate_guidance(self, tokens, word_guidance):
        expected = (*tokens.shape, len(self.word_guide_channels))
        if self.config.candidate_guidance or self.config.contextual_edits:
            candidate_size = self.config.d_model if self.config.candidate_guidance else 0
            expected += (1 + candidate_size + int(self.config.contextual_edits),)
        if word_guidance is not None and (
                self.word_fusion is None or word_guidance.shape != expected):
            raise ValueError(f'Expected word guidance with shape {expected}')

    def sequence_logits(self, tokens, mask, segments=None, *, depth=None, word_guidance=None):
        depth = self._resolve_depth(depth)
        self._validate_guidance(tokens, word_guidance)
        stage_features = {} if self.final_fusion is not None and depth == 'deep' else None
        hidden = self._encode(self._embed_sequence(tokens, mask, segments), mask, 0,
                              NEURAL_DEPTHS[depth], word_guidance, stage_features=stage_features)
        return self._sequence_projection(hidden, depth, stage_features, tokens, mask, segments, word_guidance)

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
            guide = self._make_word_guide(logits, rows, spans, text_mask)
            outputs.append(logits)
            guides.append(guide)
            if stop != layers[-1]:
                hidden = self._fuse_word_features(hidden, guide)
            start = stop
        return outputs, torch.stack(guides, dim=2)

    def _labeled_outputs(self, input_ids, attention_mask, labels, depths, word_guidance, label_weights=None):
        self._validate_guidance(input_ids, word_guidance)
        if label_weights is not None and label_weights.shape != labels.shape:
            raise ValueError('Label weights must match target IDs')
        sources = [row[mask.bool()] for row, mask in zip(input_ids, attention_mask)]
        targets = [row[row != -100] for row in labels]
        if any(len(row) == 0 for row in sources + targets):
            raise ValueError('Source and target must be nonempty')
        lengths = [len(source) + len(target) for source, target in zip(sources, targets)]
        tokens = input_ids.new_full((len(sources), max(lengths)), self.config.pad_token_id)
        mask = torch.zeros_like(tokens)
        segments = torch.zeros_like(tokens)
        expected = tokens.new_full(tokens.shape, -100)
        weights = label_weights.new_zeros(tokens.shape) if label_weights is not None else None
        guidance = (word_guidance.new_zeros((*tokens.shape, *word_guidance.shape[2:]))
                    if word_guidance is not None else None)
        for i, (source, target) in enumerate(zip(sources, targets)):
            boundary, end = len(source), lengths[i]
            tokens[i, :end] = torch.cat([source, target])
            mask[i, :end] = 1
            segments[i, boundary:end] = 1
            expected[i, boundary:end] = target
            if weights is not None:
                weights[i, boundary:end] = label_weights[i, labels[i] != -100]
            if guidance is not None:
                guidance[i, :boundary] = word_guidance[i, attention_mask[i].bool()]
        hidden = self._embed_sequence(tokens, mask, segments)
        expected = expected[:, 1:].reshape(-1)
        if weights is not None:
            weights = weights[:, 1:].reshape(-1)
        outputs = {}
        start = 0
        stage_features = {} if self.final_fusion is not None and 'deep' in depths else None
        for depth in depths:
            stop = NEURAL_DEPTHS[depth]
            hidden = self._encode(hidden, mask, start, stop, guidance, stage_features=stage_features)
            logits = self._sequence_projection(hidden, depth, stage_features, tokens, mask, segments, guidance)
            loss_fn = F.nll_loss if self.copy_head is not None and depth == 'deep' else F.cross_entropy
            next_logits = logits[:, :-1].reshape(-1, logits.size(-1))
            if weights is not None and depth == 'deep':
                token_loss = loss_fn(next_logits, expected, ignore_index=-100, reduction='none')
                loss = (token_loss * weights).sum() / weights.sum()
            else:
                loss = loss_fn(next_logits, expected, ignore_index=-100)
            outputs[depth] = CausalLMOutput(loss=loss, logits=logits)
            start = stop
        return outputs

    def sequence_losses(self, input_ids, attention_mask, labels, word_guidance=None, label_weights=None,
                        edit_labels=None, edit_label_weights=None):
        """Compute the auxiliary sentence and final rewrite objectives."""
        if self.config.span_edits:
            return {depth: self(input_ids, attention_mask, labels, depth=depth, word_guidance=word_guidance,
                                label_weights=label_weights, edit_labels=edit_labels,
                                edit_label_weights=edit_label_weights).loss
                    for depth in self.sentence_depths}
        outputs = self._labeled_outputs(input_ids, attention_mask, labels, self.sentence_depths,
                                        word_guidance, label_weights)
        return {depth: output.loss for depth, output in outputs.items()}

    def forward(self, input_ids, attention_mask, labels=None, *, depth=None, word_guidance=None,
                label_weights=None, edit_labels=None, edit_label_weights=None, **kwargs):
        depth = self._resolve_depth(depth)
        if self.config.span_edits and depth == 'deep' and (labels is not None or edit_labels is not None):
            if edit_labels is None or label_weights is not None:
                raise ValueError('The final span-edit loss requires edit labels without sentence token weights')
            labels = edit_labels
            label_weights = edit_label_weights
        if labels is None:
            return CausalLMOutput(logits=self.sequence_logits(input_ids, attention_mask, depth=depth,
                                                            word_guidance=word_guidance))
        return self._labeled_outputs(input_ids, attention_mask, labels, (depth,), word_guidance, label_weights)[depth]

    def step(self, tokens, positions, segments, valid, states, *, depth=None, word_guidance=None, copy_scale=1.):
        depth = self._resolve_depth(depth)
        hidden = self.embedding_dropout(self.token_embedding(tokens) +
                                        self.position_embedding(positions) +
                                        self.segment_embedding(segments)) * valid[:, None]
        updated = []
        stage_features = {} if self.final_fusion is not None and depth == 'deep' else None
        layer_count = NEURAL_DEPTHS[depth]
        use_copy = self.copy_head is not None and depth == 'deep'
        layer_states = states[:layer_count] if use_copy else states
        active = islice(self.layers, layer_count)
        for index, (layer, state) in enumerate(zip(active, layer_states, strict=True)):
            hidden = self._guide_stage(hidden, index, word_guidance)
            hidden, state = layer.step(hidden, valid, state)
            self._record_stage_features(stage_features, hidden, index + 1)
            updated.append(state)
        features, logits = self._project(hidden, depth, stage_features)
        if use_copy:
            # The copy head uses its own prefix memory after the six layer states.
            bias = self._copy_bias(word_guidance)
            if bias is None:
                bias = features.new_zeros(tokens.shape)
            memory = (self.copy_head.key(features)[:, None], tokens[:, None],
                      self.copy_head.copyable(tokens, valid, segments)[:, None], bias[:, None])
            if len(states) > layer_count:
                memory = tuple(torch.cat((past, current), dim=1)
                               for past, current in zip(states[layer_count], memory, strict=True))
            keys, source_tokens, allowed, source_bias = memory
            logits = self.copy_head(features[:, None], logits[:, None], keys, source_tokens, allowed[:, None],
                                    source_bias if self.config.contextual_edits else None, copy_scale=copy_scale)[:, 0]
            updated.append(memory)
        return logits, updated

    def validate_span_copy_scale(self, scale):
        if not 0. <= scale <= 1.:
            raise ValueError('span-copy-scale must be between 0 and 1')
        if scale != 1. and (not self.config.span_edits or self.copy_head is None):
            raise ValueError('span-copy-scale requires a span-edit checkpoint with source copying')

    @torch.no_grad()
    def generate(self, input_ids, attention_mask, max_new_tokens=128, num_beams=1,
                 do_sample=False, early_stopping=False, *, depth=None, word_guidance=None,
                 edit_unit_counts=None, span_copy_scale=1., **kwargs):
        depth = self._resolve_depth(depth)
        self.validate_span_copy_scale(span_copy_scale)
        if self.final_fusion is not None and depth != 'deep':
            raise ValueError('This checkpoint generates only the fused final rewrite')
        self._validate_guidance(input_ids, word_guidance)
        if num_beams != 1 or do_sample or max_new_tokens < 1:
            raise ValueError('Six-layer generation currently supports deterministic greedy decoding only')
        mask = attention_mask.bool()
        if not mask.any(1).all() or torch.any(mask[:, 1:] > mask[:, :-1]):
            raise ValueError('Nonempty right-padded sources required')
        lengths = mask.sum(1)
        if int(lengths.max()) + max_new_tokens > self.config.max_positions:
            raise ValueError('Requested generation exceeds position capacity')
        decoder = None
        if self.config.span_edits:
            if edit_unit_counts is None or max_new_tokens < 2:
                raise ValueError('Span edits require source word counts and at least two generation tokens')
            decoder = EditProgramDecoder(self.config, edit_unit_counts, input_ids.device)
            if decoder.counts.shape != lengths.shape or not (
                    (decoder.counts >= 0) & (decoder.counts < self.config.edit_position_count)).all():
                raise ValueError('Expected one supported edit word count per source')
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
            if decoder is None:
                logits[:, self.config.pad_token_id] = -torch.inf
                token = logits.argmax(-1)
            else:
                token = decoder.choose(logits, first=index == 0, remaining=max_new_tokens - index)
            token = token.masked_fill(finished, self.config.pad_token_id)
            output.append(token)
            finished = finished | (token == self.config.eos_token_id)
            if finished.all():
                break
            logits, states = self.step(token, lengths + index, torch.ones_like(lengths),
                                        ~finished, states, depth=depth, copy_scale=span_copy_scale)
        return torch.stack(output, 1)
