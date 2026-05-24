# Development Guide

## Add a new op
1. Write a fragment shader that reads input tensors with `texelFetch`.
2. Add the source to the module `*_SHADERS` export.
3. Add a wrapper function that allocates an output tensor and calls `executePass`.
4. Include the shader object when building the program cache.

## Add a new layer
1. Reuse existing ops or add custom layer-local shaders.
2. Return forward caches needed for backward.
3. Keep parameter ordering stable so trainer and serializer code can flatten tensors predictably.

## Create a new example
1. Create a page under `examples/<name>/index.html`.
2. Build one WebGL program cache at startup.
3. Wire UI controls to dataset generation, trainer calls, and inference helpers.

## Debugging tips
- Use `readTensor` to inspect intermediate tensors.
- Check framebuffer completeness and shader compile errors first.
- Compare a small tensor against a CPU reference before scaling up.

## Performance notes
- Reuse the compiled program cache and quad buffer.
- Keep texture sizes within `MAX_TEXTURE_SIZE`.
- Avoid unnecessary readbacks and repeated shader recompilation.
