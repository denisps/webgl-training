# Architecture

## Tensor representation
Tensors are `{ texture, rows, cols }` objects backed by `R32F` WebGL2 textures. Texture coordinates follow matrix indexing: pixel `(x, y)` stores element `[y, x]`.

## Render-to-texture compute
Every operation binds a framebuffer, attaches an output texture, sets a fullscreen quad program, and computes results in a fragment shader using `texelFetch`.

## Dependency diagram
`webgl + ops -> layers -> inference`

`webgl + ops + layers + loss + optimizer -> training`

Inference never imports training modules.

## Layer building blocks
Dense layers use GPU matmul, bias add, and activation shaders. Transformer blocks compose layer norm, QKV projections, scaled dot-product attention, residual adds, and feed-forward dense layers.

## Training loop
A trainer builds forward caches, computes loss, runs layer backprop, then updates parameters with an MRT Adam shader.

## Inference pipeline
Inference creates tensors from host data, runs forward-only dense or transformer stacks, reads results if needed, and frees temporary textures.
