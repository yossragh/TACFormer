from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutput


class GPTConfig(PretrainedConfig):
    model_type = "protac_gpt"

    def __init__(
        self,
        vocab_size: int = 61,
        max_seq_len: int = 512,
        n_layers: int = 8,
        d_model: int = 512,
        n_heads: int = 8,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        bias: bool = True,
        pad_token_id: int = 0,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        tie_word_embeddings: bool = True,
        # NEW: optional POI conditioning configuration.
        poi_conditioning: bool = False,
        poi_embedding_dim: int = 1280,
        **kwargs,
    ) -> None:
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.bias = bias

        # NEW: defaults preserve the original unconditional model.
        self.poi_conditioning = poi_conditioning
        self.poi_embedding_dim = poi_embedding_dim

        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()

        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.head_dim = config.d_model // config.n_heads

        self.attn_dropout_p = config.dropout

        self.qkv_proj = nn.Linear(
            config.d_model,
            3 * config.d_model,
            bias=config.bias,
        )

        self.out_proj = nn.Linear(
            config.d_model,
            config.d_model,
            bias=config.bias,
        )

        self.out_proj._is_residual_proj = True

        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        qkv = self.qkv_proj(x)  # [B, T, 3C]
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.reshape(
            bsz,
            seq_len,
            self.n_heads,
            self.head_dim,
        ).transpose(1, 2)

        k = k.reshape(
            bsz,
            seq_len,
            self.n_heads,
            self.head_dim,
        ).transpose(1, 2)

        v = v.reshape(
            bsz,
            seq_len,
            self.n_heads,
            self.head_dim,
        ).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=(
                self.attn_dropout_p
                if self.training
                else 0.0
            ),
            is_causal=True,
        )

        y = y.transpose(
            1,
            2,
        ).contiguous().reshape(
            bsz,
            seq_len,
            self.d_model,
        )

        y = self.out_proj(y)
        y = self.resid_dropout(y)

        return y


class MLP(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()

        hidden_dim = (
            config.mlp_ratio
            * config.d_model
        )

        self.fc1 = nn.Linear(
            config.d_model,
            hidden_dim,
            bias=config.bias,
        )

        self.fc2 = nn.Linear(
            hidden_dim,
            config.d_model,
            bias=config.bias,
        )

        self.fc2._is_residual_proj = True

        self.act = nn.GELU()
        self.dropout = nn.Dropout(
            config.dropout
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.dropout(x)

        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        config: GPTConfig,
    ) -> None:
        super().__init__()

        self.ln_1 = nn.LayerNorm(
            config.d_model
        )

        self.attn = CausalSelfAttention(
            config
        )

        self.ln_2 = nn.LayerNorm(
            config.d_model
        )

        self.mlp = MLP(
            config
        )

        if config.poi_conditioning:
            self.condition_gate = nn.Parameter(
                torch.zeros(())
            )
        else:
            self.register_parameter(
                "condition_gate",
                None,
            )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        poi_condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(
            self.ln_1(x),
            attention_mask=attention_mask,
        )

        if poi_condition is not None:
            x = (
                x
                + torch.tanh(
                    self.condition_gate
                )
                * poi_condition.unsqueeze(1)
            )

        x = x + self.mlp(
            self.ln_2(x)
        )

        return x


class GPTForCausalLM(PreTrainedModel):
    config_class = GPTConfig
    base_model_prefix = "protac_gpt"
    main_input_name = "input_ids"

    _tied_weights_keys = {
        "lm_head.weight": "token_emb.weight"
    }

    def __init__(
        self,
        config: GPTConfig,
    ) -> None:
        super().__init__(config)

        self.token_emb = nn.Embedding(
            config.vocab_size,
            config.d_model,
        )

        self.pos_emb = nn.Embedding(
            config.max_seq_len,
            config.d_model,
        )

        self.drop = nn.Dropout(
            config.dropout
        )

        if config.poi_conditioning:
            self.poi_norm = nn.LayerNorm(
                config.poi_embedding_dim
            )

            self.poi_proj = nn.Linear(
                config.poi_embedding_dim,
                config.d_model,
                bias=config.bias,
            )
        else:
            self.poi_norm = None
            self.poi_proj = None

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(config)
                for _ in range(
                    config.n_layers
                )
            ]
        )

        self.ln_f = nn.LayerNorm(
            config.d_model
        )

        self.lm_head = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=False,
        )

        # weight tying
        self.post_init()

        if self.config.tie_word_embeddings:
            self.tie_weights()

    def get_input_embeddings(
        self,
    ) -> nn.Embedding:
        return self.token_emb

    def set_input_embeddings(
        self,
        value: nn.Embedding,
    ) -> None:
        self.token_emb = value

    def get_output_embeddings(
        self,
    ) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(
        self,
        value: nn.Linear,
    ) -> None:
        self.lm_head = value

    def _init_weights(
        self,
        module: nn.Module,
    ) -> None:
        if isinstance(
            module,
            nn.Linear,
        ):
            std = 0.02

            if getattr(
                module,
                "_is_residual_proj",
                False,
            ):
                std = (
                    0.02
                    / math.sqrt(
                        2
                        * self.config.n_layers
                    )
                )

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=std,
            )

            if module.bias is not None:
                nn.init.zeros_(
                    module.bias
                )

        elif isinstance(
            module,
            nn.Embedding,
        ):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        poi_embeddings: torch.Tensor | None = None,
    ) -> CausalLMOutput:
        batch_size, seq_len = (
            input_ids.shape
        )

        if (
            seq_len
            > self.config.max_seq_len
        ):
            raise ValueError(
                f"Input sequence length {seq_len} "
                f"exceeds model max_seq_len="
                f"{self.config.max_seq_len}"
            )

        positions = torch.arange(
            0,
            seq_len,
            device=input_ids.device,
        ).unsqueeze(0)

        x = (
            self.token_emb(
                input_ids
            )
            + self.pos_emb(
                positions
            )
        )

        x = self.drop(x)

        poi_condition = None

        if self.config.poi_conditioning:
            if poi_embeddings is None:
                raise ValueError(
                    "poi_embeddings is required "
                    "when poi_conditioning=True."
                )

            if poi_embeddings.ndim != 2:
                raise ValueError(
                    "poi_embeddings must have "
                    "shape [B, poi_embedding_dim]."
                )
            if (
                poi_embeddings.shape[0]
                != batch_size
            ):
                raise ValueError(
                    "POI embedding batch size "
                    "must match input_ids batch size."
                )

            poi_embeddings = poi_embeddings.to(
                device=x.device,
                dtype=self.poi_proj.weight.dtype,
            )

            poi_condition = self.poi_proj(
                self.poi_norm(
                    poi_embeddings
                )
            )

        elif poi_embeddings is not None:
            raise ValueError(
                "Received poi_embeddings when "
                "poi_conditioning=False."
            )

        for block in self.blocks:
            x = block(
                x,
                attention_mask=attention_mask,
                poi_condition=poi_condition,
            )

        x = self.ln_f(x)

        logits = self.lm_head(x)

        loss = None

        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(
                    -1,
                    logits.size(-1),
                ),
                labels.reshape(-1),
                ignore_index=-100,
            )

        return CausalLMOutput(
            loss=loss,
            logits=logits,
        )

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        eos_token_id: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = 0.9,
        bad_token_ids: list[int] | None = None,
        poi_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if temperature <= 0:
            raise ValueError(
                "temperature must be > 0"
            )

        if (
            top_k is not None
            and top_k <= 0
        ):
            raise ValueError(
                "top_k must be > 0"
            )

        if (
            top_p is not None
            and not (
                0.0
                < top_p
                <= 1.0
            )
        ):
            raise ValueError(
                "top_p must be in (0, 1]"
            )

        was_training = self.training
        self.eval()

        try:
            batch_size = input_ids.size(0)

            finished = torch.zeros(
                batch_size,
                dtype=torch.bool,
                device=input_ids.device,
            )

            finished = (
                finished
                | (
                    input_ids
                    == eos_token_id
                ).any(
                    dim=1
                )
            )

            for _ in range(
                max_new_tokens
            ):
                if (
                    input_ids.size(1)
                    > self.config.max_seq_len
                ):
                    model_input_ids = (
                        input_ids[
                            :,
                            -self.config.max_seq_len:,
                        ]
                    )
                else:
                    model_input_ids = (
                        input_ids
                    )
                out = self(
                    input_ids=model_input_ids,
                    poi_embeddings=poi_embeddings,
                )

                logits = out.logits[
                    :,
                    -1,
                    :,
                ]

                logits = (
                    logits
                    / temperature
                )

                if bad_token_ids is not None:
                    for token_id in bad_token_ids:
                        if (
                            0
                            <= token_id
                            < logits.size(-1)
                        ):
                            logits[
                                :,
                                token_id,
                            ] = float(
                                "-inf"
                            )

                if top_k is not None:
                    k = min(
                        top_k,
                        logits.size(-1),
                    )

                    values, _ = torch.topk(
                        logits,
                        k,
                    )

                    logits = logits.masked_fill(
                        logits
                        < values[
                            :,
                            -1,
                        ].unsqueeze(-1),
                        float("-inf"),
                    )

                if (
                    top_p is not None
                    and top_p < 1.0
                ):
                    (
                        sorted_logits,
                        sorted_indices,
                    ) = torch.sort(
                        logits,
                        descending=True,
                        dim=-1,
                    )

                    cumulative_probs = torch.cumsum(
                        F.softmax(
                            sorted_logits,
                            dim=-1,
                        ),
                        dim=-1,
                    )

                    sorted_indices_to_remove = (
                        cumulative_probs
                        > top_p
                    )

                    sorted_indices_to_remove[
                        ...,
                        1:,
                    ] = (
                        sorted_indices_to_remove[
                            ...,
                            :-1,
                        ].clone()
                    )

                    sorted_indices_to_remove[
                        ...,
                        0,
                    ] = False

                    sorted_logits = (
                        sorted_logits.masked_fill(
                            sorted_indices_to_remove,
                            float("-inf"),
                        )
                    )

                    logits = torch.full_like(
                        logits,
                        float("-inf"),
                    )

                    logits.scatter_(
                        1,
                        sorted_indices,
                        sorted_logits,
                    )

                probs = F.softmax(
                    logits,
                    dim=-1,
                )

                next_token = torch.multinomial(
                    probs,
                    num_samples=1,
                )

                pad_token = torch.full_like(
                    next_token,
                    self.config.pad_token_id,
                )

                next_token = torch.where(
                    finished.unsqueeze(-1),
                    pad_token,
                    next_token,
                )

                input_ids = torch.cat(
                    [
                        input_ids,
                        next_token,
                    ],
                    dim=1,
                )

                finished = (
                    finished
                    | (
                        next_token.squeeze(1)
                        == eos_token_id
                    )
                )

                if finished.all():
                    break

            return input_ids

        finally:
            self.train(
                was_training
            )