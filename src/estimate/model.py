import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal time embeddings for timesteps."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        """
        Args:
            time: (batch_size,) tensor of timesteps
        Returns:
            (batch_size, dim) tensor of embeddings
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: (batch_size, seq_len, d_model)
            key: (batch_size, seq_len, d_model)
            value: (batch_size, seq_len, d_model)
            mask: optional mask
        Returns:
            (batch_size, seq_len, d_model)
        """
        batch_size = query.size(0)

        # Linear projections
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        context = torch.matmul(attn, V)

        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # Final linear layer
        output = self.out_linear(context)

        return output


class CrossAttention(nn.Module):
    """Cross-attention for conditioning."""

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, context):
        """
        Args:
            x: (batch_size, seq_len_x, d_model) - queries (theta tokens)
            context: (batch_size, seq_len_ctx, d_model) - keys/values (y tokens + time)
        Returns:
            (batch_size, seq_len_x, d_model)
        """
        # Cross-attention
        attn_out = self.attention(x, context, context)
        x = self.norm1(x + attn_out)

        # Feed-forward
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)

        return x


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and feed-forward."""

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            (batch_size, seq_len, d_model)
        """
        # Self-attention
        attn_out = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)

        # Feed-forward
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)

        return x


class DiffusionTransformer(nn.Module):
    """Diffusion model with Transformer architecture.

    Input: theta_t (batch_size, n_data_point, param_dim) + y (batch_size, n_data_point, obs_dim) + t
    Output: predicted noise (batch_size, n_data_point, param_dim)

    Each of the n_data_point theta values is treated as a separate token.
    We condition on the corresponding y observations and the timestep t.
    """

    def __init__(
        self,
        obs_dim: int,
        param_dim: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
        n_timesteps: int = 1000
    ):
        """
        Args:
            obs_dim: Dimension of observations y
            param_dim: Dimension of parameters theta
            d_model: Hidden dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
            n_timesteps: Number of diffusion timesteps
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.param_dim = param_dim
        self.d_model = d_model
        self.n_timesteps = n_timesteps

        # Time embeddings (shared across all tokens)
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )

        # Input projections
        # Theta: (batch, n_data_point, param_dim) -> (batch, n_data_point, d_model)
        self.theta_proj = nn.Linear(param_dim, d_model)

        # Y observations: (batch, n_data_point, obs_dim) -> (batch, n_data_point, d_model)
        self.y_proj = nn.Linear(obs_dim, d_model)

        # Learnable positional embeddings for token positions
        # We don't know n_data_point in advance, so we'll use a large max and slice
        self.max_seq_len = 1000
        self.pos_embed = nn.Parameter(torch.randn(1, self.max_seq_len, d_model) * 0.02)

        # Cross-attention layers to condition on y and time
        self.cross_attn_layers = nn.ModuleList([
            CrossAttention(d_model, num_heads, dropout)
            for _ in range(num_layers // 2)
        ])

        # Self-attention transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Output projection back to param_dim
        self.output_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, param_dim)
        )

        # Diffusion schedule (cosine schedule)
        self.register_buffer('betas', self._cosine_beta_schedule(n_timesteps))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('alphas_cumprod_prev',
                            F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0))

        # Variance for sampling
        self.register_buffer('posterior_variance',
                            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))

    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """Cosine schedule as proposed in DDPM improved."""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        # Clip more conservatively to avoid extreme values
        return torch.clip(betas, 0.0001, 0.02)

    def forward(self, theta_t, y, t):
        """
        Predict noise given noisy theta_t, observations y, and timestep t.

        Args:
            theta_t: (batch_size, n_data_point, param_dim) noisy parameters
            y: (batch_size, n_data_point, obs_dim) observations
            t: (batch_size,) timesteps

        Returns:
            epsilon: (batch_size, n_data_point, param_dim) predicted noise
        """
        batch_size, n_data_point, _ = theta_t.shape

        # Time embedding (batch_size, d_model)
        t_emb = self.time_embed(t)  # (batch_size, d_model)
        # Expand to match sequence length: (batch_size, n_data_point, d_model)
        t_emb = t_emb.unsqueeze(1).expand(batch_size, n_data_point, self.d_model)

        # Project y observations: (batch_size, n_data_point, d_model)
        y_tokens = self.y_proj(y)

        # Combine y and time as context for cross-attention
        # (batch_size, n_data_point, d_model)
        context = y_tokens + t_emb

        # Project theta: (batch_size, n_data_point, d_model)
        theta_tokens = self.theta_proj(theta_t)

        # Add positional embeddings
        theta_tokens = theta_tokens + self.pos_embed[:, :n_data_point, :]

        # Apply transformer layers with interleaved cross-attention
        x = theta_tokens
        for i, transformer_layer in enumerate(self.transformer_layers):
            x = transformer_layer(x)

            # Apply cross-attention every other layer
            if i % 2 == 0 and i // 2 < len(self.cross_attn_layers):
                x = self.cross_attn_layers[i // 2](x, context)

        # Output projection: (batch_size, n_data_point, param_dim)
        epsilon = self.output_proj(x)

        return epsilon

    def add_noise(self, theta_0, t, noise=None):
        """Add noise to theta_0 at timestep t.

        Args:
            theta_0: (batch_size, n_data_point, param_dim) clean parameters
            t: (batch_size,) timesteps
            noise: (batch_size, n_data_point, param_dim) optional noise

        Returns:
            theta_t: (batch_size, n_data_point, param_dim) noisy parameters
            noise: (batch_size, n_data_point, param_dim) the noise that was added
        """
        if noise is None:
            noise = torch.randn_like(theta_0)

        # Get alpha values for each sample in batch
        sqrt_alphas_cumprod_t = self.alphas_cumprod[t].sqrt()
        sqrt_one_minus_alphas_cumprod_t = (1.0 - self.alphas_cumprod[t]).sqrt()

        # Reshape to (batch_size, 1, 1) for broadcasting
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.view(-1, 1, 1)

        theta_t = sqrt_alphas_cumprod_t * theta_0 + sqrt_one_minus_alphas_cumprod_t * noise

        return theta_t, noise

    @torch.no_grad()
    def generate(self, y, device='cpu'):
        """Generate theta samples given observations y.

        Args:
            y: (batch_size, n_data_point, obs_dim) observations
            device: device to use

        Returns:
            (batch_size, n_data_point, param_dim) sampled parameters
        """
        batch_size, n_data_point, _ = y.shape
        y = y.to(device)

        # Start with pure noise
        theta_t = torch.randn(batch_size, n_data_point, self.param_dim, device=device)

        # Reverse diffusion process
        for t_idx in reversed(range(self.n_timesteps)):
            t = torch.full((batch_size,), t_idx, device=device, dtype=torch.long)

            # Predict noise
            predicted_noise = self.forward(theta_t, y, t)

            # Get coefficients for this timestep
            alpha_t = self.alphas[t_idx]
            alpha_bar_t = self.alphas_cumprod[t_idx]
            beta_t = self.betas[t_idx]

            # Coefficient for predicted_noise (ensure proper broadcasting)
            coef1 = beta_t / torch.sqrt(1.0 - alpha_bar_t)

            # Mean of p(theta_{t-1} | theta_t) - DDPM formula
            # theta_{t-1} = (1/sqrt(alpha_t)) * (theta_t - (beta_t/sqrt(1-alpha_bar_t)) * eps_theta)
            mean = (1.0 / torch.sqrt(alpha_t)) * (theta_t - coef1 * predicted_noise)

            if t_idx > 0:
                noise = torch.randn_like(theta_t)
                # Use simpler variance: just beta_t instead of posterior variance
                # This is more stable for generation
                sigma_t = torch.sqrt(beta_t)
                theta_t = mean + sigma_t * noise
            else:
                theta_t = mean

            # Clip to prevent explosion (important!)
            theta_t = torch.clamp(theta_t, -10.0, 10.0)

        return theta_t

    @torch.no_grad()
    def sample(self, y, device='cpu'):
        """Alias for generate method.

        Args:
            y: (batch_size, n_data_point, obs_dim) observations
            device: device to use

        Returns:
            (batch_size, n_data_point, param_dim) sampled parameters
        """
        return self.generate(y, device=device)
