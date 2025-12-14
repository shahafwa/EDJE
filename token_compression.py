import torch
from torch import nn
from transformers.activations import ACT2FN


class TokenCompressionAdapter(nn.Module):

    def __init__(
            self,
            num_compressed_tokens: int,
            hidden_size: int,
            intermediate_size: int,
            output_size: int,
            hidden_act: str,
            num_attention_heads: int,
            layer_norm_eps: float
    ):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, num_compressed_tokens, hidden_size))
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.attention = torch.nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            batch_first=True
        )
        self.layernorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.mlp = MLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act
        )
        self.projection = nn.Linear(hidden_size, output_size)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        batch_size = hidden_state.shape[0]
        query = self.query.repeat(batch_size, 1, 1)
        key = self.key(hidden_state)
        value = self.value(hidden_state)
        hidden_state = self.attention(query, key, value)[0]

        residual = hidden_state
        hidden_state = self.layernorm(hidden_state)
        hidden_state = self.mlp(hidden_state)
        hidden_state = residual + hidden_state

        hidden_state = self.projection(hidden_state)
        return hidden_state


class MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str):
        super().__init__()
        self.activation_fn = ACT2FN[hidden_act]
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states

