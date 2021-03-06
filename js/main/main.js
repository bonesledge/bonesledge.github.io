'use strict';

const canvas = document.getElementsByTagName('canvas')[0];
canvas.width = canvas.clientWidth;
canvas.height = canvas.clientHeight;


let config = {
    TEXTURE_DOWNSAMPLE: 1,
    DENSITY_DISSIPATION: 0.98,
    //VELOCITY_DISSIPATION: 0.99,
    VELOCITY_DISSIPATION: 0.999,
    PRESSURE_DISSIPATION: 0.8,
    PRESSURE_ITERATIONS: 25,
    CURL: 30,
    SPLAT_MOUSE_RADIUS: 0.0005,
    SPLAT_RADIUS: 0.000005,
    ORGANISM_COUNT: 25,
    SPECIES_COUNT: 2
}

let pointers = [];
/* organisms are an array of arrays. Each species is its own flock whose velocity vectors are 
impacted by the boid algorithm.
*/
let organisms = [];


const  { gl, ext } = getWebGLContext(canvas);

function initOrganisms () {
    if (organisms.length > 0) {
        return;
    }
    for (let i = 0; i < config.SPECIES_COUNT; i++) {
        organisms[i] = [];
        let color = [Math.random() * 10, Math.random() * 10, Math.random() * 10];
        for (let j = 0; j < config.ORGANISM_COUNT; j++) {
            organisms[i][j] = [
                Math.random() * canvas.width * 0.1, Math.random() * canvas.height * 0.1 // position
                , 0, 0                              // speed
                , 0, 0                               // acceleration
                , color                                 // color
            ];
            organisms[i][j][COHESION_FORCE] = boidConfig.cohesionForce.default;
            organisms[i][j][ALIGNMENT_FORCE] = boidConfig.alignmentForce.default;
            organisms[i][j][SEPARATION_FORCE] = boidConfig.separationForce.default;
            organisms[i][j][SPEED_LIMIT_ROOT] = boidConfig.speedLimitRoot.default;
            organisms[i][j][SPEED_LIMIT] = boidConfig.speedLimit.default;
        }
    }
}

let boidConfig = {

    speedLimitRoot: { default: 10, min: 0, max: 20 },
    accelerationLimitRoot: { default: 4, min: 0, max: 10 },
    speedLimit: { default: 100, min: 0, max: 400 },
    accelerationLimit: { default: 16, min: 0, max: 100 },
    separationDistance: 3600,
    alignmentDistance: 32400,
    cohesionDistance: 32400,
    separationForce: { default: 0.15, min: 0, max: 1 },
    cohesionForce: { default: 0.1, min: 0, max: 1 },
    alignmentForce: { default: 0.7, min: 0, max: 1 },
    attractors: [[
        Infinity // x
        , Infinity // y
        , 150 // dist
        , 0.25 // spd
    ]],
    color: { min: 0, max: 10 }
}

// refer to indexes within each organism's array, not values.
const POSITIONX = 0
const POSITIONY = 1
const SPEEDX = 2
const SPEEDY = 3
const ACCELERATIONX = 4
const ACCELERATIONY = 5
const COLOR_INDEX = 6 
const SPEED_LIMIT = 7
const SPEED_LIMIT_ROOT = 8
const SEPARATION_FORCE = 9
const COHESION_FORCE = 10
const ALIGNMENT_FORCE = 11

function updateOrgPos () {
    for (let i = 0; i < organisms.length; i++) {
        let flock = organisms[i];
        var current = flock.length;


        var sepDist = boidConfig.separationDistance
            , sepForce = boidConfig.separationForce.default
            , cohDist = boidConfig.cohesionDistance
            , cohForce = boidConfig.cohesionForce.default
            , aliDist = boidConfig.alignmentDistance
            , aliForce = boidConfig.alignmentForce.default
            , speedLimit = boidConfig.speedLimit.default
            , speedLimitRoot = boidConfig.speedLimitRoot.default
            , accelerationLimit = boidConfig.accelerationLimit.default
            , accelerationLimitRoot = boidConfig.accelerationLimitRoot.default
            , size = flock.length
            , sforceX, sforceY
            , cforceX, cforceY
            , aforceX, aforceY
            , spareX, spareY
            , attractors = boidConfig.attractors
            , attractorCount = attractors.length
            , attractor
            , distSquared
            , currPos
            , length
            , target
            , ratio

        while (current--) {
            cohForce = flock[current][COHESION_FORCE]
            aliForce = flock[current][ALIGNMENT_FORCE]
            sepForce = flock[current][SEPARATION_FORCE]
            speedLimit = flock[current][SPEED_LIMIT]
            speedLimitRoot = flock[current][SPEED_LIMIT_ROOT]
            sforceX = 0; sforceY = 0
            cforceX = 0; cforceY = 0
            aforceX = 0; aforceY = 0
            currPos = flock[current]
            

            // Attractors
            target = attractorCount
            while (target--) {
                attractor = attractors[target]
                spareX = currPos[0] - attractor[0]
                spareY = currPos[1] - attractor[1]
                distSquared = spareX*spareX + spareY*spareY

                if (distSquared < attractor[2]*attractor[2]) {
                    length = hypot(spareX, spareY)
                    flock[current][SPEEDX] -= (attractor[3] * spareX / length) || 0
                    flock[current][SPEEDY] -= (attractor[3] * spareY / length) || 0
                }
            }

            target = size
            while (target--) {
                if (target === current) continue
                spareX = currPos[0] - flock[target][0]
                spareY = currPos[1] - flock[target][1]
                distSquared = spareX*spareX + spareY*spareY

                if (distSquared < sepDist) {
                    sforceX += spareX
                    sforceY += spareY
                } else {
                    if (distSquared < cohDist) {
                        cforceX += spareX
                        cforceY += spareY
                    }
                    if (distSquared < aliDist) {
                        aforceX += flock[target][SPEEDX]
                        aforceY += flock[target][SPEEDY]
                    }
                }
            }

            // Separation
            length = hypot(sforceX, sforceY)
            flock[current][ACCELERATIONX] += (sepForce * sforceX / length) || 0
            flock[current][ACCELERATIONY] += (sepForce * sforceY / length) || 0
            // Cohesion
            length = hypot(cforceX, cforceY)
            flock[current][ACCELERATIONX] -= (cohForce * cforceX / length) || 0
            flock[current][ACCELERATIONY] -= (cohForce * cforceY / length) || 0
            // Alignment
            length = hypot(aforceX, aforceY)
            flock[current][ACCELERATIONX] -= (aliForce * aforceX / length) || 0
            flock[current][ACCELERATIONY] -= (aliForce * aforceY / length) || 0
        }
        current = size

        // Apply speed/acceleration for
        // this tick
        while (current--) {
            speedLimit = flock[current][SPEED_LIMIT]
            speedLimitRoot = flock[current][SPEED_LIMIT_ROOT]

            if (accelerationLimit) {
                distSquared = flock[current][ACCELERATIONX]*flock[current][ACCELERATIONX] + flock[current][ACCELERATIONY]*flock[current][ACCELERATIONY]
                if (distSquared > accelerationLimit) {
                    ratio = accelerationLimitRoot / hypot(flock[current][ACCELERATIONX], flock[current][ACCELERATIONY])
                    flock[current][ACCELERATIONX] *= ratio
                    flock[current][ACCELERATIONY] *= ratio
                }
            }

            flock[current][SPEEDX] += flock[current][ACCELERATIONX]
            flock[current][SPEEDY] += flock[current][ACCELERATIONY]

            if (speedLimit) {
                distSquared = flock[current][SPEEDX]*flock[current][SPEEDX] + flock[current][SPEEDY]*flock[current][SPEEDY]
                if (distSquared > speedLimit) {
                    ratio = speedLimitRoot / hypot(flock[current][SPEEDX], flock[current][SPEEDY])
                    flock[current][SPEEDX] *= ratio
                    flock[current][SPEEDY] *= ratio
                }
            }

            flock[current][POSITIONX] += flock[current][SPEEDX]
            flock[current][POSITIONY] += flock[current][SPEEDY]
        }
    }

}


// double-dog-leg hypothenuse approximation
// http://forums.parallax.com/discussion/147522/dog-leg-hypotenuse-approximation
function hypot(a, b) {
    a = Math.abs(a)
    b = Math.abs(b)
    var lo = Math.min(a, b)
    var hi = Math.max(a, b)
    return hi + 3 * lo / 32 + Math.max(0, 2 * lo - hi) / 8 + Math.max(0, 4 * lo - hi) / 16
}


function getWebGLContext (canvas) {
    const params = { alpha: false, depth: false, stencil: false, antialias: false };

    let gl = canvas.getContext('webgl2', params);
    const isWebGL2 = !!gl;
    if (!isWebGL2)
        gl = canvas.getContext('webgl', params) || canvas.getContext('experimental-webgl', params);

    let halfFloat;
    let supportLinearFiltering;
    if (isWebGL2) {
        gl.getExtension('EXT_color_buffer_float');
        supportLinearFiltering = gl.getExtension('OES_texture_float_linear');
    } else {
        halfFloat = gl.getExtension('OES_texture_half_float');
        supportLinearFiltering = gl.getExtension('OES_texture_half_float_linear');
    }

    gl.clearColor(0.0, 0.0, 0.0, 1.0);

    const halfFloatTexType = isWebGL2 ? gl.HALF_FLOAT : halfFloat.HALF_FLOAT_OES;
    let formatRGBA;
    let formatRG;
    let formatR;

    if (isWebGL2)
    {
        formatRGBA = getSupportedFormat(gl, gl.RGBA16F, gl.RGBA, halfFloatTexType);
        formatRG = getSupportedFormat(gl, gl.RG16F, gl.RG, halfFloatTexType);
        formatR = getSupportedFormat(gl, gl.R16F, gl.RED, halfFloatTexType);
    }
    else
    {
        formatRGBA = getSupportedFormat(gl, gl.RGBA, gl.RGBA, halfFloatTexType);
        formatRG = getSupportedFormat(gl, gl.RGBA, gl.RGBA, halfFloatTexType);
        formatR = getSupportedFormat(gl, gl.RGBA, gl.RGBA, halfFloatTexType);
    }

    return {
        gl,
        ext: {
            formatRGBA,
            formatRG,
            formatR,
            halfFloatTexType,
            supportLinearFiltering
        }
    };
}

function getSupportedFormat (gl, internalFormat, format, type)
{
    if (!supportRenderTextureFormat(gl, internalFormat, format, type))
    {
        switch (internalFormat)
        {
            case gl.R16F:
                return getSupportedFormat(gl, gl.RG16F, gl.RG, type);
            case gl.RG16F:
                return getSupportedFormat(gl, gl.RGBA16F, gl.RGBA, type);
            default:
                return null;
        }
    }

    return {
        internalFormat,
        format
    }
}

function supportRenderTextureFormat (gl, internalFormat, format, type) {
    let texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, 4, 4, 0, format, type, null);

    let fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);

    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (status != gl.FRAMEBUFFER_COMPLETE)
        return false;
    return true;
}

function pointerPrototype () {
    this.id = -1;
    this.x = 0;
    this.y = 0;
    this.dx = 0;
    this.dy = 0;
    this.down = false;
    this.moved = false;
    this.color = [30, 0, 300];
}

pointers.push(new pointerPrototype());

class GLProgram {
    constructor (vertexShader, fragmentShader) {
        this.uniforms = {};
        this.program = gl.createProgram();

        gl.attachShader(this.program, vertexShader);
        gl.attachShader(this.program, fragmentShader);
        gl.linkProgram(this.program);

        if (!gl.getProgramParameter(this.program, gl.LINK_STATUS))
            throw gl.getProgramInfoLog(this.program);

        const uniformCount = gl.getProgramParameter(this.program, gl.ACTIVE_UNIFORMS);
        for (let i = 0; i < uniformCount; i++) {
            const uniformName = gl.getActiveUniform(this.program, i).name;
            this.uniforms[uniformName] = gl.getUniformLocation(this.program, uniformName);
        }
    }

    bind () {
        gl.useProgram(this.program);
    }
}

function compileShader (type, source) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);

    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS))
        throw gl.getShaderInfoLog(shader);

    return shader;
};

const baseVertexShader = compileShader(gl.VERTEX_SHADER, `
    precision highp float;
    precision mediump sampler2D;

    attribute vec2 aPosition;
    varying vec2 vUv;
    varying vec2 vL;
    varying vec2 vR;
    varying vec2 vT;
    varying vec2 vB;
    uniform vec2 texelSize;

    void main () {
        vUv = aPosition * 0.5 + 0.5;
        vL = vUv - vec2(texelSize.x, 0.0);
        vR = vUv + vec2(texelSize.x, 0.0);
        vT = vUv + vec2(0.0, texelSize.y);
        vB = vUv - vec2(0.0, texelSize.y);
        gl_Position = vec4(aPosition, 0.0, 1.0);
    }
`);

const clearShader = compileShader(gl.FRAGMENT_SHADER, `
    precision highp float;
    precision mediump sampler2D;

    varying vec2 vUv;
    uniform sampler2D uTexture;
    uniform float value;

    void main () {
        gl_FragColor = value * texture2D(uTexture, vUv);
    }
`);

const displayShader = compileShader(gl.FRAGMENT_SHADER, `
    precision highp float;
    precision mediump sampler2D;

    varying vec2 vUv;
    uniform sampler2D uTexture;

    void main () {
        gl_FragColor = texture2D(uTexture, vUv);
    }
`);

const splatShader = compileShader(gl.FRAGMENT_SHADER, `
    precision highp float;
    precision mediump sampler2D;

    varying vec2 vUv;
    uniform sampler2D uTarget;
    uniform float aspectRatio;
    uniform vec3 color;
    uniform vec2 point;
    uniform float radius;

    void main () {
        vec2 p = vUv - point.xy;
        p.x *= aspectRatio;
        vec3 splat = exp(-dot(p, p) / radius) * color;
        vec3 base = texture2D(uTarget, vUv).xyz;
        gl_FragColor = vec4(base + splat, 1.0);
    }
`);

const advectionManualFilteringShader = compileShader(gl.FRAGMENT_SHADER, `
    precision highp float;
    precision mediump sampler2D;

    varying vec2 vUv;
    uniform sampler2D uVelocity;
    uniform sampler2D uSource;
    uniform vec2 texelSize;
    uniform float dt;
    uniform float dissipation;

    vec4 bilerp (in sampler2D sam, in vec2 p) {
        vec4 st;
        st.xy = floor(p - 0.5) + 0.5;
        st.zw = st.xy + 1.0;
        vec4 uv = st * texelSize.xyxy;
        vec4 a = texture2D(sam, uv.xy);
        vec4 b = texture2D(sam, uv.zy);
        vec4 c = texture2D(sam, uv.xw);
        vec4 d = texture2D(sam, uv.zw);
        vec2 f = p - st.xy;
        return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
    }

    void main () {
        vec2 coord = gl_FragCoord.xy - dt * texture2D(uVelocity, vUv).xy;
        gl_FragColor = dissipation * bilerp(uSource, coord);
        gl_FragColor.a = 1.0;
    }
`);

const advectionShader = compileShader(gl.FRAGMENT_SHADER, `
    precision highp float;
    precision mediump sampler2D;

    varying vec2 vUv;
    uniform sampler2D uVelocity;
    uniform sampler2D uSource;
    uniform vec2 texelSize;
    uniform float dt;
    uniform float dissipation;

    void main () {
        vec2 coord = vUv - dt * texture2D(uVelocity, vUv).xy * texelSize;
        gl_FragColor = dissipation * texture2D(uSource, coord);
        gl_FragColor.a = 1.0;
    }
`);

const divergenceShader = compileShader(gl.FRAGMENT_SHADER, `
    precision highp float;
    precision mediump sampler2D;

    varying vec2 vUv;
    varying vec2 vL;
    varying vec2 vR;
    varying vec2 vT;
    varying vec2 vB;
    uniform sampler2D uVelocity;

    vec2 sampleVelocity (in vec2 uv) {
        vec2 multiplier = vec2(1.0, 1.0);
        if (uv.x < 0.0) { uv.x = 0.0; multiplier.x = -1.0; }
        if (uv.x > 1.0) { uv.x = 1.0; multiplier.x = -1.0; }
        if (uv.y < 0.0) { uv.y = 0.0; multiplier.y = -1.0; }
        if (uv.y > 1.0) { uv.y = 1.0; multiplier.y = -1.0; }
        return multiplier * texture2D(uVelocity, uv).xy;
    }

    void main () {
        float L = sampleVelocity(vL).x;
        float R = sampleVelocity(vR).x;
        float T = sampleVelocity(vT).y;
        float B = sampleVelocity(vB).y;
        float div = 0.5 * (R - L + T - B);
        gl_FragColor = vec4(div, 0.0, 0.0, 1.0);
    }
`);

const curlShader = compileShader(gl.FRAGMENT_SHADER, `
    precision highp float;
    precision mediump sampler2D;

    varying vec2 vUv;
    varying vec2 vL;
    varying vec2 vR;
    varying vec2 vT;
    varying vec2 vB;
    uniform sampler2D uVelocity;

    void main () {
        float L = texture2D(uVelocity, vL).y;
        float R = texture2D(uVelocity, vR).y;
        float T = texture2D(uVelocity, vT).x;
        float B = texture2D(uVelocity, vB).x;
        float vorticity = R - L - T + B;
        gl_FragColor = vec4(vorticity, 0.0, 0.0, 1.0);
    }
`);

const vorticityShader = compileShader(gl.FRAGMENT_SHADER, `
    precision highp float;
    precision mediump sampler2D;

    varying vec2 vUv;
    varying vec2 vT;
    varying vec2 vB;
    uniform sampler2D uVelocity;
    uniform sampler2D uCurl;
    uniform float curl;
    uniform float dt;

    void main () {
        float T = texture2D(uCurl, vT).x;
        float B = texture2D(uCurl, vB).x;
        float C = texture2D(uCurl, vUv).x;
        vec2 force = vec2(abs(T) - abs(B), 0.0);
        force *= 1.0 / length(force + 0.00001) * curl * C;
        vec2 vel = texture2D(uVelocity, vUv).xy;
        gl_FragColor = vec4(vel + force * dt, 0.0, 1.0);
    }
`);

const pressureShader = compileShader(gl.FRAGMENT_SHADER, `
    precision highp float;
    precision mediump sampler2D;

    varying vec2 vUv;
    varying vec2 vL;
    varying vec2 vR;
    varying vec2 vT;
    varying vec2 vB;
    uniform sampler2D uPressure;
    uniform sampler2D uDivergence;

    vec2 boundary (in vec2 uv) {
        uv = min(max(uv, 0.0), 1.0);
        return uv;
    }

    void main () {
        float L = texture2D(uPressure, boundary(vL)).x;
        float R = texture2D(uPressure, boundary(vR)).x;
        float T = texture2D(uPressure, boundary(vT)).x;
        float B = texture2D(uPressure, boundary(vB)).x;
        float C = texture2D(uPressure, vUv).x;
        float divergence = texture2D(uDivergence, vUv).x;
        float pressure = (L + R + B + T - divergence) * 0.25;
        gl_FragColor = vec4(pressure, 0.0, 0.0, 1.0);
    }
`);

const gradientSubtractShader = compileShader(gl.FRAGMENT_SHADER, `
    precision highp float;
    precision mediump sampler2D;

    varying vec2 vUv;
    varying vec2 vL;
    varying vec2 vR;
    varying vec2 vT;
    varying vec2 vB;
    uniform sampler2D uPressure;
    uniform sampler2D uVelocity;

    vec2 boundary (in vec2 uv) {
        uv = min(max(uv, 0.0), 1.0);
        return uv;
    }

    void main () {
        float L = texture2D(uPressure, boundary(vL)).x;
        float R = texture2D(uPressure, boundary(vR)).x;
        float T = texture2D(uPressure, boundary(vT)).x;
        float B = texture2D(uPressure, boundary(vB)).x;
        vec2 velocity = texture2D(uVelocity, vUv).xy;
        velocity.xy -= vec2(R - L, T - B);
        gl_FragColor = vec4(velocity, 0.0, 1.0);
    }
`);

let textureWidth;
let textureHeight;
let density;
let velocity;
let divergence;
let curl;
let pressure;
initFramebuffers();

const clearProgram = new GLProgram(baseVertexShader, clearShader);
const displayProgram = new GLProgram(baseVertexShader, displayShader);
const splatProgram = new GLProgram(baseVertexShader, splatShader);
const advectionProgram = new GLProgram(baseVertexShader, ext.supportLinearFiltering ? advectionShader : advectionManualFilteringShader);
const divergenceProgram = new GLProgram(baseVertexShader, divergenceShader);
const curlProgram = new GLProgram(baseVertexShader, curlShader);
const vorticityProgram = new GLProgram(baseVertexShader, vorticityShader);
const pressureProgram = new GLProgram(baseVertexShader, pressureShader);
const gradienSubtractProgram = new GLProgram(baseVertexShader, gradientSubtractShader);

function initFramebuffers () {
    textureWidth = gl.drawingBufferWidth >> config.TEXTURE_DOWNSAMPLE;
    textureHeight = gl.drawingBufferHeight >> config.TEXTURE_DOWNSAMPLE;

    const texType = ext.halfFloatTexType;
    const rgba = ext.formatRGBA;
    const rg   = ext.formatRG;
    const r    = ext.formatR;

    density    = createDoubleFBO(2, textureWidth, textureHeight, rgba.internalFormat, rgba.format, texType, ext.supportLinearFiltering ? gl.LINEAR : gl.NEAREST);
    velocity   = createDoubleFBO(0, textureWidth, textureHeight, rg.internalFormat, rg.format, texType, ext.supportLinearFiltering ? gl.LINEAR : gl.NEAREST);
    divergence = createFBO      (4, textureWidth, textureHeight, r.internalFormat, r.format, texType, gl.NEAREST);
    curl       = createFBO      (5, textureWidth, textureHeight, r.internalFormat, r.format, texType, gl.NEAREST);
    pressure   = createDoubleFBO(6, textureWidth, textureHeight, r.internalFormat, r.format, texType, gl.NEAREST);
}

function createFBO (texId, w, h, internalFormat, format, type, param) {
    gl.activeTexture(gl.TEXTURE0 + texId);
    let texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, param);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, param);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, w, h, 0, format, type, null);

    let fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
    gl.viewport(0, 0, w, h);
    gl.clear(gl.COLOR_BUFFER_BIT);

    return [texture, fbo, texId];
}

function createDoubleFBO (texId, w, h, internalFormat, format, type, param) {
    let fbo1 = createFBO(texId    , w, h, internalFormat, format, type, param);
    let fbo2 = createFBO(texId + 1, w, h, internalFormat, format, type, param);

    return {
        get read () {
            return fbo1;
        },
        get write () {
            return fbo2;
        },
        swap () {
            let temp = fbo1;
            fbo1 = fbo2;
            fbo2 = temp;
        }
    }
}

const blit = (() => {
    gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, -1, 1, 1, 1, 1, -1]), gl.STATIC_DRAW);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, gl.createBuffer());
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array([0, 1, 2, 0, 2, 3]), gl.STATIC_DRAW);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(0);

    return (destination) => {
        gl.bindFramebuffer(gl.FRAMEBUFFER, destination);
        gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);
    }
})();

let gameState = updateSimulation;
let lastTime = Date.now();
let updates = 0;
initOrganisms();
update();

function update () {
    resizeCanvas();
    gameState();
    requestAnimationFrame(update);
}

function updateSimulation () {
    const dt = Math.min((Date.now() - lastTime) / 1000, 0.016);
    lastTime = Date.now();

    gl.viewport(0, 0, textureWidth, textureHeight);

    advectionProgram.bind();
    gl.uniform2f(advectionProgram.uniforms.texelSize, 1.0 / textureWidth, 1.0 / textureHeight);
    gl.uniform1i(advectionProgram.uniforms.uVelocity, velocity.read[2]);
    gl.uniform1i(advectionProgram.uniforms.uSource, velocity.read[2]);
    gl.uniform1f(advectionProgram.uniforms.dt, dt);
    gl.uniform1f(advectionProgram.uniforms.dissipation, config.VELOCITY_DISSIPATION);
    blit(velocity.write[1]);
    velocity.swap();

    gl.uniform1i(advectionProgram.uniforms.uVelocity, velocity.read[2]);
    gl.uniform1i(advectionProgram.uniforms.uSource, density.read[2]);
    gl.uniform1f(advectionProgram.uniforms.dissipation, config.DENSITY_DISSIPATION);
    blit(density.write[1]);
    density.swap();

    for (let i = 0; i < pointers.length; i++) {
        const pointer = pointers[i];
        if (pointer.moved) {
            splat(pointer.x, pointer.y, pointer.dx, pointer.dy, pointer.color, config.SPLAT_MOUSE_RADIUS);
            pointer.moved = false;
        }
    }

    if (updates % 2 == 0) {
        updateOrganisms();
    }

    curlProgram.bind();
    gl.uniform2f(curlProgram.uniforms.texelSize, 1.0 / textureWidth, 1.0 / textureHeight);
    gl.uniform1i(curlProgram.uniforms.uVelocity, velocity.read[2]);
    blit(curl[1]);

    vorticityProgram.bind();
    gl.uniform2f(vorticityProgram.uniforms.texelSize, 1.0 / textureWidth, 1.0 / textureHeight);
    gl.uniform1i(vorticityProgram.uniforms.uVelocity, velocity.read[2]);
    gl.uniform1i(vorticityProgram.uniforms.uCurl, curl[2]);
    gl.uniform1f(vorticityProgram.uniforms.curl, config.CURL);
    gl.uniform1f(vorticityProgram.uniforms.dt, dt);
    blit(velocity.write[1]);
    velocity.swap();

    divergenceProgram.bind();
    gl.uniform2f(divergenceProgram.uniforms.texelSize, 1.0 / textureWidth, 1.0 / textureHeight);
    gl.uniform1i(divergenceProgram.uniforms.uVelocity, velocity.read[2]);
    blit(divergence[1]);

    clearProgram.bind();
    let pressureTexId = pressure.read[2];
    gl.activeTexture(gl.TEXTURE0 + pressureTexId);
    gl.bindTexture(gl.TEXTURE_2D, pressure.read[0]);
    gl.uniform1i(clearProgram.uniforms.uTexture, pressureTexId);
    gl.uniform1f(clearProgram.uniforms.value, config.PRESSURE_DISSIPATION);
    blit(pressure.write[1]);
    pressure.swap();

    pressureProgram.bind();
    gl.uniform2f(pressureProgram.uniforms.texelSize, 1.0 / textureWidth, 1.0 / textureHeight);
    gl.uniform1i(pressureProgram.uniforms.uDivergence, divergence[2]);
    pressureTexId = pressure.read[2];
    gl.uniform1i(pressureProgram.uniforms.uPressure, pressureTexId);
    gl.activeTexture(gl.TEXTURE0 + pressureTexId);
    for (let i = 0; i < config.PRESSURE_ITERATIONS; i++) {
        gl.bindTexture(gl.TEXTURE_2D, pressure.read[0]);
        blit(pressure.write[1]);
        pressure.swap();
    }

    gradienSubtractProgram.bind();
    gl.uniform2f(gradienSubtractProgram.uniforms.texelSize, 1.0 / textureWidth, 1.0 / textureHeight);
    gl.uniform1i(gradienSubtractProgram.uniforms.uPressure, pressure.read[2]);
    gl.uniform1i(gradienSubtractProgram.uniforms.uVelocity, velocity.read[2]);
    blit(velocity.write[1]);
    velocity.swap();

    gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
    displayProgram.bind();
    gl.uniform1i(displayProgram.uniforms.uTexture, density.read[2]);
    blit(null);

    updates++;
}

function updateSpeciateScreen() {
    
}

function updateOrganisms() {
    updateOrgPos();
    for (let i = 0; i < organisms.length; i++) {
        let flock = organisms[i];
        for (let j = 0; j < flock.length; j++) {
            const org = flock[j];

            const splatDx = 1000 * (Math.random() - 0.5);
            const splatDy = 1000 * (Math.random() - 0.5);
            splat(org[0], org[1], splatDx, splatDy, org[COLOR_INDEX], config.SPLAT_RADIUS);

            if (org[0] > canvas.width) {
                org[0] -= canvas.width;
            }
            if (org[1] > canvas.height) {
                org[1] -= canvas.height;
            }
            if (org[0] < 0) {
                org[0] += canvas.width;
            }
            if (org[1] < 0) {
                org[1] += canvas.height;
            }
        }
    }
}


function splat (x, y, dx, dy, color, splatRadius) {
    splatProgram.bind();
    gl.uniform1i(splatProgram.uniforms.uTarget, velocity.read[2]);
    gl.uniform1f(splatProgram.uniforms.aspectRatio, canvas.width / canvas.height);
    gl.uniform2f(splatProgram.uniforms.point, x / canvas.width, 1.0 - y / canvas.height);
    gl.uniform3f(splatProgram.uniforms.color, dx, -dy, 1.0);
    gl.uniform1f(splatProgram.uniforms.radius, splatRadius);
    blit(velocity.write[1]);
    velocity.swap();

    gl.uniform1i(splatProgram.uniforms.uTarget, density.read[2]);
    gl.uniform3f(splatProgram.uniforms.color, color[0] * 0.3, color[1] * 0.3, color[2] * 0.3);
    blit(density.write[1]);
    density.swap();
}

function multipleSplats (amount) {
    for (let i = 0; i < amount; i++) {
        const color = [Math.random() * 10, Math.random() * 10, Math.random() * 10];
        const x = canvas.width * Math.random();
        const y = canvas.height * Math.random();
        const dx = 1000 * (Math.random() - 0.5);
        const dy = 1000 * (Math.random() - 0.5);
        splat(x, y, dx, dy, color, config.SPLAT_RADIUS);
    }
}

function resizeCanvas () {
    if (canvas.width != canvas.clientWidth || canvas.height != canvas.clientHeight) {
        canvas.width = canvas.clientWidth;
        canvas.height = canvas.clientHeight;
        initFramebuffers();
    }
}

let exemplar;

function splitSpecies() {
    organisms.sort((a, b) => (a.length >= b.length));
    const biggestSpecies = organisms.pop();
    const oldSpecies = biggestSpecies.slice(0, biggestSpecies.length / 2);
    const newSpecies = biggestSpecies.slice(biggestSpecies.length / 2, biggestSpecies.length - 1);
    const color = [Math.random() * 10, Math.random() * 10, Math.random() * 10];
    for (let i = 0; i < newSpecies.length; i++) {
        newSpecies[i][COLOR_INDEX] = color;
    }
    organisms.push(oldSpecies, newSpecies); 
    exemplar = newSpecies[0];
}

function getHundredNormalized(value, min, max) {
    const difference = max - min;
    const adjusted = value - min;
    const normalized = 1.0*adjusted/difference;
    return normalized*100;
}

function getHundredDenormalized(value, min, max) {
    const normalized = value/100.0;
    const difference = max - min;
    const adjusted = normalized*difference;
    return normalized + min;
}

function createSpeciateOptions(newSpecies) {
    const speciateOptionsDiv = document.createElement("div");
    speciateOptionsDiv.id = "speciateoptions";
    document.getElementById("wrapper").appendChild(speciateOptionsDiv);
    
    // <style data="test" type="text/css"></style>
    const style = document.createElement("style");
    style.setAttribute("data", "test");
    style.setAttribute("type", "text/css");
    speciateOptionsDiv.appendChild(style);

    const c1 = createLabel("rightlabel", "", "redslider");
    const c2 = createLabel("rightlabel", "", "greenslider");
    const c3 = createLabel("rightlabel", "", "blueslider");
    const bgColor = toColor(exemplar[COLOR_INDEX]);
    c1.style.backgroundColor = bgColor;
    c2.style.backgroundColor = bgColor;
    c3.style.backgroundColor = bgColor;
    createColorSlider("redslider", boidConfig.color.min, boidConfig.color.max, exemplar,
        0, speciateOptionsDiv, "red", c1, c2, c3, c1);
    createColorSlider("greenslider", boidConfig.color.min, boidConfig.color.max, exemplar,
        1, speciateOptionsDiv, "green", c1, c2, c3, c2);
    createColorSlider("blueslider", boidConfig.color.min, boidConfig.color.max, exemplar,
        2, speciateOptionsDiv, "blue", c1, c2, c3, c3);
    createConfigSlider("cohesionslider", boidConfig.cohesionForce.min, boidConfig.cohesionForce.max, 
        COHESION_FORCE, speciateOptionsDiv, "solitary", "social");
    createConfigSlider("alignmentslider", boidConfig.alignmentForce.min, boidConfig.alignmentForce.max,
        ALIGNMENT_FORCE, speciateOptionsDiv, "wiggly", "straight");
    createConfigSlider("speedslider", boidConfig.speedLimit.min, boidConfig.speedLimit.max,
        SPEED_LIMIT, speciateOptionsDiv, "slow", "fast");
}

function toColor(colorList) {
    return "rgb(" + 255*colorList[0] + ", " + 255*colorList[1] + ", " + 255*colorList[2] + ")";
}

function createSlider(id, oninput) {
    //  <input type="range" min="1" max="100" value="50" class="slider" id="myRange">
    const slider = document.createElement("input");
    slider.className = "slider";
    slider.id = id;
    slider.setAttribute("type", "range");
    slider.setAttribute("min", 0);
    slider.setAttribute("max", 100);
    slider.oninput = oninput;
    return slider;
}

function createColorSlider(id, min, max, exemplar, rgbIndex, div, color,
    colorExample1, colorExample2, colorExample3, colorExample) {
    const label = createLabel("leftlabel", color, id);
    label.style.backgroundColor = color;
    
    //  <input type="range" min="1" max="100" value="50" class="slider" id="myRange">
    const slider = createSlider(id, function() {
        exemplar[COLOR_INDEX][rgbIndex] = getHundredDenormalized(this.value, min, max);
        const bgColor = toColor(exemplar[COLOR_INDEX]);
        colorExample1.style.backgroundColor = bgColor;
        colorExample2.style.backgroundColor = bgColor;
        colorExample3.style.backgroundColor = bgColor;
    });
    slider.value = getHundredNormalized(exemplar[COLOR_INDEX][rgbIndex], min, max);
    div.appendChild(slider);    
    div.appendChild(label);
    div.appendChild(colorExample);
}

function createConfigSlider(id, min, max, exemplarIndex, div, leftLabelText, rightLabelText) {
    //  <input type="range" min="1" max="100" value="50" class="slider" id="myRange">
    const slider = createSlider(id, function() {
        exemplar[exemplarIndex] = getHundredDenormalized(this.value, min, max);
    });
    slider.value = getHundredNormalized(exemplar[exemplarIndex], min, max);
    div.appendChild(slider);
    div.appendChild(createLabel("leftlabel", leftLabelText, id));
    div.appendChild(createLabel("rightlabel", rightLabelText, id));
}

function createLabel(className, text, forId) {
    const label = document.createElement("label");
    label.className = className;
    label.setAttribute("for", forId);
    label.innerHTML = text;
    return label;
}

function removeSpeciateOptions() {
    document.getElementById("speciateoptions").remove();
}

function finalizeExemplar() {
    exemplar[SPEED_LIMIT_ROOT] = Math.sqrt(exemplar[SPEED_LIMIT]);
}

function matchSpeciesToExemplar(species) {
    for (let i = 0; i < species.length; i++) {
        for (let j = COLOR_INDEX; j < species[i].length; j++) {
            species[i][j] = exemplar[j];
        }
    }
}

window.addEventListener('mousemove', (e) => {
    pointers[0].moved = pointers[0].down;
    pointers[0].dx = (e.offsetX - pointers[0].x) * 10.0;
    pointers[0].dy = (e.offsetY - pointers[0].y) * 10.0;
    pointers[0].x = e.offsetX;
    pointers[0].y = e.offsetY;
});

window.addEventListener('touchmove', (e) => {
    e.preventDefault();
    const touches = e.targetTouches;
    for (let i = 0; i < touches.length; i++) {
        let pointer = pointers[i];
        pointer.moved = pointer.down;
        pointer.dx = (touches[i].pageX - pointer.x) * 10.0;
        pointer.dy = (touches[i].pageY - pointer.y) * 10.0;
        pointer.x = touches[i].pageX;
        pointer.y = touches[i].pageY;
    }
}, false);

window.addEventListener('mousedown', () => {
    pointers[0].down = true;
    pointers[0].color = [Math.random() + 0.2, Math.random() + 0.2, Math.random() + 0.2];
});

window.addEventListener('touchstart', (e) => {
    e.preventDefault();
    const touches = e.targetTouches;
    for (let i = 0; i < touches.length; i++) {
        if (i >= pointers.length)
            pointers.push(new pointerPrototype());

        pointers[i].id = touches[i].identifier;
        pointers[i].down = true;
        pointers[i].x = touches[i].pageX;
        pointers[i].y = touches[i].pageY;
        pointers[i].color = [Math.random() + 0.2, Math.random() + 0.2, Math.random() + 0.2];
    }
});

window.addEventListener('mouseup', () => {
    pointers[0].down = false;
});

window.addEventListener('touchend', (e) => {
    const touches = e.changedTouches;
    for (let i = 0; i < touches.length; i++)
        for (let j = 0; j < pointers.length; j++)
            if (touches[i].identifier == pointers[j].id)
                pointers[j].down = false;
});

const speciateButton = document.getElementById('speciatebutton');
speciateButton.addEventListener("click", function(){
    if (gameState === updateSimulation) {
        speciateButton.innerHTML = "spawn";
        gameState = updateSpeciateScreen;
        splitSpecies();
        createSpeciateOptions();
    } else {
        finalizeExemplar();
        matchSpeciesToExemplar(organisms[organisms.length - 1]);
        speciateButton.innerHTML = "speciate";
        gameState = updateSimulation;
        removeSpeciateOptions();
    }
});
