# Map Location Prediction

Train a patch-transformer model entirely on the GPU (WebGL2) to predict which map region a cropped image patch belongs to.

---

## Terminology

**Region** — A labeled area of the map. The map is divided into N non-overlapping regions (e.g. 8 or 128). Each region has an integer ID 0…N-1. The model's job is to guess the region from a small image crop.

**Patch / Sample** — A `REGION_SIZE × REGION_SIZE` pixel crop extracted at a random position on the map. Every patch is labeled with the region ID of its center pixel. A patch is the atomic unit of training and inference.

**Token (tile)** — A `PATCH_SIDE × PATCH_SIDE` pixel sub-region within a patch. For `REGION_SIZE=64` and `PATCH_SIDE=8`, each patch is sliced into an 8×8 grid of 64 tokens. Each token is a flat vector of `PATCH_SIDE² × CHANNELS = 192` values. Tokens let the transformer reason about spatial relationships inside the patch.

**Sample** — Same as patch. In the dataset, "samples per region" controls how many patches are collected per region to balance the training distribution.

**Batch** — A fixed number of patches processed together in one forward + backward pass. Using a large batch (e.g. 256) fills the GPU with more work per dispatch, improving utilization. Larger batches also produce more stable gradient estimates.

**Epoch** — One complete pass through the entire training dataset. After each epoch the average loss is recorded. Training typically needs many epochs for the model to converge.

---

## Architecture

The model is a **ViT-style patch transformer** — a Vision Transformer adapted for small image crops.

```
Input patch  [batch, REGION_SIZE²×3]         e.g. [256, 12288] for 64×64
     │
  Patchify   [batch×SEQ_LEN, TOKEN_DIM]           [16384, 192]
     │         split into spatial tokens
  Embed      [batch×SEQ_LEN, D_MODEL]              [16384, 128]
     │         shared linear projection
  Transformer block (see below)
     │
  Mean pool  [batch, D_MODEL]                       [256, 128]
     │         average over all tokens
  Head       [batch, nRegions]                      [256, 128]
     │         linear classification layer
  Cross-entropy loss
```

### Patchification

The flat input `[H·W·C]` per sample is reinterpreted as a grid of non-overlapping spatial tiles:

```
SEQ_LEN  = (REGION_SIZE / PATCH_SIDE)²    tokens per sample
TOKEN_DIM = PATCH_SIDE² × CHANNELS        features per token
```

For `REGION_SIZE=64`, `PATCH_SIDE=8`, `CHANNELS=3`:
- `SEQ_LEN = 64`, `TOKEN_DIM = 192`

This is done on the GPU with a single shader pass — no data is copied to CPU.

### Token Embedding

Each token is linearly projected from `TOKEN_DIM` to `D_MODEL` (shared weights):

$$E = P \cdot W_e + b_e, \quad W_e \in \mathbb{R}^{192 \times 128}$$

### Transformer Block

The transformer block applies multi-head self-attention followed by a feed-forward network, both with residual connections and layer normalization. Crucially, attention is **block-diagonal**: the 64 tokens of each sample only attend to each other, never to tokens from other samples in the batch.

```
h₁  = LayerNorm(x)
attn = MultiHeadAttention(h₁)         per-sample block-diagonal attention
x₁  = x + attn · Wₒ                  residual connection

h₂  = LayerNorm(x₁)
ff   = GELU(h₂ · W₁ + b₁) · W₂ + b₂  feed-forward network
y   = x₁ + ff                         residual connection
```

#### Multi-Head Attention

For each of `N_HEADS` heads with `head_dim = D_MODEL / N_HEADS`:

$$Q_h = E \cdot W_q^h, \quad K_h = E \cdot W_k^h, \quad V_h = E \cdot W_v^h$$

$$\text{Attention}_h = \text{softmax}\!\left(\frac{Q_h K_h^\top}{\sqrt{d_\text{head}}}\right) V_h$$

The `1/√d_head` scale prevents dot products from growing large enough to push softmax into saturation. The outputs of all heads are concatenated and projected:

$$\text{MultiHead}(E) = \text{concat}(\text{Attention}_1, \ldots, \text{Attention}_{N_H}) \cdot W_o$$

#### Layer Normalization

Normalizes each token's feature vector to zero mean and unit variance, then applies learned scale `γ` and shift `β`:

$$\text{LN}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \varepsilon}} \cdot \gamma + \beta$$

This stabilizes training by keeping activations in a consistent range.

#### GELU Activation

Used in the feed-forward network. Smoother than ReLU, empirically better for transformers:

$$\text{GELU}(x) = x \cdot \Phi(x)$$

where $\Phi$ is the standard normal CDF, approximated in the shader as:

$$\text{GELU}(x) \approx 0.5 \cdot x \cdot \left(1 + \tanh\!\left(\sqrt{\tfrac{2}{\pi}}\left(x + 0.044715 x^3\right)\right)\right)$$

### Mean Pooling

After the transformer, the `SEQ_LEN` token vectors per sample are averaged into a single `D_MODEL`-dimensional representation:

$$\text{pool} = \frac{1}{S} \sum_{t=1}^{S} E_t$$

### Classification Head

A final linear layer maps from `D_MODEL` to `nRegions`:

$$\text{logits} = \text{pool} \cdot W_\text{head} + b_\text{head}$$

---

## Training

### Cross-Entropy Loss

For a sample with true region label $y$ and predicted logits $z$:

$$\mathcal{L} = -z_y + \log \sum_k e^{z_k}$$

This is numerically computed as $-z_y + \log\sum_k e^{z_k - z_\text{max}} + z_\text{max}$ to avoid overflow.

### Adam Optimizer

At each step, for each parameter $\theta$ with gradient $g$:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g^2$$
$$\hat{m} = m_t / (1 - \beta_1^t), \quad \hat{v} = v_t / (1 - \beta_2^t)$$
$$\theta_t = \theta_{t-1} - \alpha \cdot \hat{m} / (\sqrt{\hat{v}} + \varepsilon)$$

Default: `α=0.004`, `β₁=0.9`, `β₂=0.999`, `ε=1e-8`.

---

## Model Parameters

| Component | Tensor | Shape | Count |
|---|---|---|---|
| Embed | Wₑ | [192, 128] | 24 576 |
| Embed | bₑ | [1, 128] | 128 |
| Transformer | Wq, Wk, Wv, Wo | [128, 128] × 4 | 65 536 |
| Transformer | W₁ | [128, 256] | 32 768 |
| Transformer | b₁ | [1, 256] | 256 |
| Transformer | W₂ | [256, 128] | 32 768 |
| Transformer | b₂ | [1, 128] | 128 |
| Transformer | γ₁, β₁, γ₂, β₂ | [1, 128] × 4 | 512 |
| Head | W_head | [128, nRegions] | 128 × N |
| Head | b_head | [1, nRegions] | N |
| **Total** | | | **~157K + 129N** |

For 128 regions: ~173K parameters. All arithmetic happens in R32F textures on the GPU.

---

## Saved Model Format

"Download weights" saves a JSON file with both the weights and the architecture:

```json
{
  "architecture": {
    "type": "patch-transformer",
    "nRegions": 128,
    "regionSize": 64,
    "patchSide": 8,
    "seqLen": 64,
    "dModel": 128,
    "nHeads": 4,
    "ffnDim": 256
  },
  "weights": [
    { "rows": 192, "cols": 128, "data": [...] },
    ...
  ]
}
```

"Upload weights" reads this file and automatically adjusts the model's `nRegions` if the saved value differs from the current setting.
