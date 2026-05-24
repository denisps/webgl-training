export const QUAD_VERTICES = new Float32Array([
  -1, -1,
   1, -1,
  -1,  1,
   1,  1,
]);

export function createWebGL2Context(canvas) {
  const gl = canvas.getContext('webgl2', {
    alpha: false,
    antialias: false,
    depth: false,
    stencil: false,
    premultipliedAlpha: false,
    preserveDrawingBuffer: false,
  });
  if (!gl) {
    throw new Error('WebGL2 is not available.');
  }
  const ext = gl.getExtension('EXT_color_buffer_float');
  if (!ext) {
    throw new Error('EXT_color_buffer_float is required.');
  }
  gl.disable(gl.BLEND);
  gl.disable(gl.CULL_FACE);
  gl.disable(gl.DEPTH_TEST);
  gl.disable(gl.DITHER);
  return gl;
}

export function createTensor(gl, rows, cols, data) {
  const texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  gl.texImage2D(
    gl.TEXTURE_2D,
    0,
    gl.R32F,
    cols,
    rows,
    0,
    gl.RED,
    gl.FLOAT,
    data ?? null,
  );
  gl.bindTexture(gl.TEXTURE_2D, null);
  return { texture, rows, cols };
}

export function createFramebuffer(gl, tensor) {
  const fb = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tensor.texture, 0);
  const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
  if (status !== gl.FRAMEBUFFER_COMPLETE) {
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.deleteFramebuffer(fb);
    throw new Error(`Framebuffer incomplete: ${status}`);
  }
  return fb;
}

export function bindOutputTensor(gl, fb, tensor) {
  gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tensor.texture, 0);
  gl.viewport(0, 0, tensor.cols, tensor.rows);
}

export function readTensor(gl, tensor) {
  const fb = createFramebuffer(gl, tensor);
  gl.readBuffer(gl.COLOR_ATTACHMENT0);
  const readFormat = gl.getParameter(gl.IMPLEMENTATION_COLOR_READ_FORMAT);
  const readType = gl.getParameter(gl.IMPLEMENTATION_COLOR_READ_TYPE);
  let result;
  if (readFormat === gl.RED && readType === gl.FLOAT) {
    result = new Float32Array(tensor.rows * tensor.cols);
    gl.readPixels(0, 0, tensor.cols, tensor.rows, gl.RED, gl.FLOAT, result);
  } else {
    const rgba = new Float32Array(tensor.rows * tensor.cols * 4);
    gl.readPixels(0, 0, tensor.cols, tensor.rows, gl.RGBA, gl.FLOAT, rgba);
    result = new Float32Array(tensor.rows * tensor.cols);
    for (let index = 0; index < result.length; index += 1) {
      result[index] = rgba[index * 4];
    }
  }
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.deleteFramebuffer(fb);
  return result;
}

export function destroyTensor(gl, tensor) {
  if (tensor?.texture) {
    gl.deleteTexture(tensor.texture);
  }
}
