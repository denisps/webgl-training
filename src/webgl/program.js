import { QUAD_VERTICES } from './context.js';

export const VERTEX_SHADER_SRC = `#version 300 es
precision highp float;
precision highp int;
layout(location = 0) in vec2 a_position;
void main() {
  gl_Position = vec4(a_position, 0.0, 1.0);
}`;

function compileShader(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(shader) || 'Unknown shader compile error';
    gl.deleteShader(shader);
    throw new Error(log);
  }
  return shader;
}

export function compileProgram(gl, fragmentSrc) {
  const vertexShader = compileShader(gl, gl.VERTEX_SHADER, VERTEX_SHADER_SRC);
  const fragmentShader = compileShader(gl, gl.FRAGMENT_SHADER, fragmentSrc);
  const program = gl.createProgram();
  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.bindAttribLocation(program, 0, 'a_position');
  gl.linkProgram(program);
  gl.deleteShader(vertexShader);
  gl.deleteShader(fragmentShader);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(program) || 'Unknown program link error';
    gl.deleteProgram(program);
    throw new Error(log);
  }
  const uniformTypes = new Map();
  const uniformCount = gl.getProgramParameter(program, gl.ACTIVE_UNIFORMS);
  for (let index = 0; index < uniformCount; index += 1) {
    const info = gl.getActiveUniform(program, index);
    if (info) {
      uniformTypes.set(info.name.replace(/\[0\]$/, ''), info.type);
    }
  }
  program.__uniformTypes = uniformTypes;
  return program;
}

function createQuadBuffer(gl) {
  const buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, QUAD_VERTICES, gl.STATIC_DRAW);
  gl.bindBuffer(gl.ARRAY_BUFFER, null);
  return buffer;
}

export function buildProgramCache(gl, shaderSources) {
  const cache = { __quadBuffer: createQuadBuffer(gl) };
  for (const [name, fragmentSrc] of Object.entries(shaderSources)) {
    cache[name] = compileProgram(gl, fragmentSrc);
  }
  return cache;
}

function setUniform(gl, location, value, textureUnit, type) {
  if (typeof value === 'number') {
    if (type === gl.INT || type === gl.BOOL || type === gl.SAMPLER_2D) {
      gl.uniform1i(location, value);
    } else {
      gl.uniform1f(location, value);
    }
    return textureUnit;
  }
  if (value instanceof Float32Array) {
    switch (type) {
      case gl.FLOAT_VEC2:
        gl.uniform2fv(location, value);
        break;
      case gl.FLOAT_VEC3:
        gl.uniform3fv(location, value);
        break;
      case gl.FLOAT_VEC4:
        gl.uniform4fv(location, value);
        break;
      default:
        gl.uniform1fv(location, value);
        break;
    }
    return textureUnit;
  }
  if (typeof WebGLTexture !== 'undefined' && value instanceof WebGLTexture) {
    gl.activeTexture(gl.TEXTURE0 + textureUnit);
    gl.bindTexture(gl.TEXTURE_2D, value);
    gl.uniform1i(location, textureUnit);
    return textureUnit + 1;
  }
  throw new Error('Unsupported uniform value.');
}

export function executePass(gl, program, uniforms, outputTensor, quadBuffer) {
  const framebuffer = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, outputTensor.texture, 0);
  const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
  if (status !== gl.FRAMEBUFFER_COMPLETE) {
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.deleteFramebuffer(framebuffer);
    throw new Error(`Framebuffer incomplete: ${status}`);
  }
  gl.viewport(0, 0, outputTensor.cols, outputTensor.rows);
  gl.useProgram(program);
  gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

  let textureUnit = 0;
  for (const [name, value] of Object.entries(uniforms)) {
    const location = gl.getUniformLocation(program, name);
    if (location !== null) {
      const type = program.__uniformTypes?.get(name);
      textureUnit = setUniform(gl, location, value, textureUnit, type);
    }
  }

  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  gl.disableVertexAttribArray(0);
  gl.bindBuffer(gl.ARRAY_BUFFER, null);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.deleteFramebuffer(framebuffer);
}
