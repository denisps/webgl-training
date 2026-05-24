import { createTensor, destroyTensor } from '../webgl/context.js';

export const OPTIMIZER_SHADERS = {
  adamUpdate: `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
layout(location = 0) out float o_param;
layout(location = 1) out float o_m;
layout(location = 2) out float o_v;
uniform sampler2D u_Param;
uniform sampler2D u_Grad;
uniform sampler2D u_M;
uniform sampler2D u_V;
uniform float u_Lr;
uniform float u_Beta1;
uniform float u_Beta2;
uniform float u_Eps;
uniform float u_Beta1Pow;
uniform float u_Beta2Pow;
void main() {
  int col = int(gl_FragCoord.x);
  int row = int(gl_FragCoord.y);
  float param = texelFetch(u_Param, ivec2(col, row), 0).r;
  float grad = texelFetch(u_Grad, ivec2(col, row), 0).r;
  float m = texelFetch(u_M, ivec2(col, row), 0).r;
  float v = texelFetch(u_V, ivec2(col, row), 0).r;
  float nextM = u_Beta1 * m + (1.0 - u_Beta1) * grad;
  float nextV = u_Beta2 * v + (1.0 - u_Beta2) * grad * grad;
  float mHat = nextM / max(1.0 - u_Beta1Pow, u_Eps);
  float vHat = nextV / max(1.0 - u_Beta2Pow, u_Eps);
  o_param = param - u_Lr * mHat / (sqrt(vHat) + u_Eps);
  o_m = nextM;
  o_v = nextV;
}`,
};

export function createAdamState(gl, parameterShapes) {
  const m = parameterShapes.map(({ rows, cols }) => createTensor(gl, rows, cols, new Float32Array(rows * cols)));
  const v = parameterShapes.map(({ rows, cols }) => createTensor(gl, rows, cols, new Float32Array(rows * cols)));
  return { m, v, t: 0 };
}

function bindTextureUniform(gl, program, name, texture, unit) {
  const location = gl.getUniformLocation(program, name);
  gl.activeTexture(gl.TEXTURE0 + unit);
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.uniform1i(location, unit);
}

export function adamStep(gl, programs, parameters, gradients, adamState, config) {
  adamState.t += 1;
  const beta1Pow = config.beta1 ** adamState.t;
  const beta2Pow = config.beta2 ** adamState.t;
  const quadBuffer = programs.__quadBuffer;
  const program = programs.adamUpdate;

  for (let index = 0; index < parameters.length; index += 1) {
    const param = parameters[index];
    const grad = gradients[index];
    const m = adamState.m[index];
    const v = adamState.v[index];
    const nextParam = createTensor(gl, param.rows, param.cols);
    const nextM = createTensor(gl, param.rows, param.cols);
    const nextV = createTensor(gl, param.rows, param.cols);

    const framebuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, nextParam.texture, 0);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, nextM.texture, 0);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT2, gl.TEXTURE_2D, nextV.texture, 0);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1, gl.COLOR_ATTACHMENT2]);
    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (status !== gl.FRAMEBUFFER_COMPLETE) {
      throw new Error(`Adam framebuffer incomplete: ${status}`);
    }

    gl.viewport(0, 0, param.cols, param.rows);
    gl.useProgram(program);
    gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

    bindTextureUniform(gl, program, 'u_Param', param.texture, 0);
    bindTextureUniform(gl, program, 'u_Grad', grad.texture, 1);
    bindTextureUniform(gl, program, 'u_M', m.texture, 2);
    bindTextureUniform(gl, program, 'u_V', v.texture, 3);
    gl.uniform1f(gl.getUniformLocation(program, 'u_Lr'), config.lr);
    gl.uniform1f(gl.getUniformLocation(program, 'u_Beta1'), config.beta1);
    gl.uniform1f(gl.getUniformLocation(program, 'u_Beta2'), config.beta2);
    gl.uniform1f(gl.getUniformLocation(program, 'u_Eps'), config.eps);
    gl.uniform1f(gl.getUniformLocation(program, 'u_Beta1Pow'), beta1Pow);
    gl.uniform1f(gl.getUniformLocation(program, 'u_Beta2Pow'), beta2Pow);

    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.disableVertexAttribArray(0);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.deleteFramebuffer(framebuffer);

    destroyTensor(gl, param);
    destroyTensor(gl, m);
    destroyTensor(gl, v);

    parameters[index].texture = nextParam.texture;
    adamState.m[index] = nextM;
    adamState.v[index] = nextV;
  }
}
