"use strict";

import JSZip from "jszip";
import { loadModel, createCanvasRenderer } from "./model_utils.js";

const PIXEL_VALUES = 8;

let logsElement = undefined;

function log(line) {
    if (logsElement === undefined) {
        return;
    }
    logsElement.innerText += line + "\n";
    logsElement.scrollTop = logsElement.scrollHeight;
}

async function main() {
    const adapter = await navigator.gpu?.requestAdapter();
    const device = await adapter?.requestDevice({
        requiredLimits: {
            maxStorageBuffersPerShaderStage: 30,
        },
    });
    if (!device) {
        console.error("WebGPU is not supported in this browser");
        return;
    }

    logsElement = document.getElementById("logs");

    let frames = [];
    let parameters = {};
    let resolution = undefined;

    const zipFileInput = document.getElementById("zip-file");
    const modelFileInput = document.getElementById("model-file");
    const loadButton = document.getElementById("load-button");
    const runButton = document.getElementById("run-button");
    const originalCanvas = document.getElementById("original");
    const predictedCanvas = document.getElementById("predicted");
    const framesLimitInput = document.getElementById("frames-limit");
    const fpsOverlayElement = document.getElementById("fps-overlay");
    const frameTimePlotCanvas = document.getElementById("frame-time-plot");
    const loopInput = document.getElementById("loop");
    const repeatsInput = document.getElementById("repeats");

    runButton.disabled = true;

    loadButton.onclick = async () => {
        if (zipFileInput.files.length === 0) {
            log("Missing data file");
            return;
        }

        loadButton.disabled = true;
        runButton.disabled = true;

        for (const frame of frames) {
            frame.samplePoints.destroy();
        }

        const limit =
            framesLimitInput.value === ""
                ? undefined
                : parseInt(framesLimitInput.value);

        log("Loading ground truth data ...");
        const zipFile = zipFileInput.files[0];
        const data = await readGroundTruthZip(zipFile, device, limit);

        frames = data.frames;
        parameters = data.parameters;
        resolution = parameters.resolution;

        log("Ground truth data loaded");
        log(`Frames: ${frames.length}`);
        log(`Parameters: ${JSON.stringify(parameters, null, 2)}`);

        log("Data ready");
        runButton.disabled = false;
        loadButton.disabled = false;
    };

    runButton.onclick = async () => {
        if (modelFileInput.files.length === 0) {
            log("Missing model file");
            return;
        }

        runButton.disabled = true;
        const numRepeats = parseInt(repeatsInput.value);
        const avgFrameTimes = [];
        for (let i = 0; i < numRepeats; i++) {
            const avgFrameTime = await new Promise((resolve) => {
                runHandler(
                    modelFileInput.files[0],
                    originalCanvas,
                    predictedCanvas,
                    device,
                    resolution,
                    frames,
                    fpsOverlayElement,
                    frameTimePlotCanvas,
                    loopInput,
                    resolve,
                    i,
                    numRepeats,
                );
            });
            avgFrameTimes.push(avgFrameTime);
        }

        if (numRepeats > 1) {
            const n = avgFrameTimes.length;
            const avg = avgFrameTimes.reduce((a, b) => a + b, 0) / n;
            const variance =
                avgFrameTimes.reduce(
                    (acc, t) => acc + Math.pow(t - avg, 2),
                    0,
                ) / n;
            const std = Math.sqrt(variance);
            log(
                `Overall avg. frame time: ${avg.toFixed(2)} ms ` +
                    `(±${std.toFixed(2)} ms)`,
            );
        }

        runButton.disabled = false;
    };
}

window.onload = () => main();

async function readGroundTruthZip(file, device, limit) {
    const zip = await JSZip.loadAsync(file);
    const files = Object.keys(zip.files).sort();
    const frames = [];
    let parameters = {};

    for (const filename of files) {
        if (filename.endsWith(".bin")) {
            if (limit !== undefined && frames.length >= limit) {
                continue;
            }
            const content = await zip.files[filename].async("arraybuffer");
            const data = new Float32Array(content);
            frames.push(createBuffersFromFrame(data, device));
        } else if (filename === "parameters.json") {
            const content = await zip.files[filename].async("string");
            parameters = JSON.parse(content);
        }
        log(`Parsed ${filename}`);
    }

    return { frames, parameters };
}

function createBuffersFromFrame(data, device) {
    const totalPixels = data.length / PIXEL_VALUES;

    // SamplePoint struct: vec3f pos (offset 0) + vec2f dir (offset 16) = 32 bytes
    const samplePoints = new Float32Array(totalPixels * 8);
    const radiance = new Float32Array(totalPixels * 3);

    for (let i = 0; i < totalPixels; i++) {
        // Position (vec3f at offset 0)
        samplePoints[i * 8 + 0] = data[i * PIXEL_VALUES + 0];
        samplePoints[i * 8 + 1] = data[i * PIXEL_VALUES + 1];
        samplePoints[i * 8 + 2] = data[i * PIXEL_VALUES + 2];

        // Direction (vec2f at offset 16 = 4 floats)
        const azimuth = data[i * PIXEL_VALUES + 3];
        const elevation = data[i * PIXEL_VALUES + 4];
        // Normalize azimuth from [-π, π] to [0, 1]
        samplePoints[i * 8 + 4] = (azimuth + Math.PI) / (2.01 * Math.PI);
        // Normalize elevation from [0, π] to [0, 1]
        samplePoints[i * 8 + 5] = elevation / Math.PI;

        radiance[i * 3 + 0] = data[i * PIXEL_VALUES + 5];
        radiance[i * 3 + 1] = data[i * PIXEL_VALUES + 6];
        radiance[i * 3 + 2] = data[i * PIXEL_VALUES + 7];
    }

    const samplePointsBuffer = device.createBuffer({
        size: samplePoints.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(samplePointsBuffer, 0, samplePoints);

    return {
        samplePoints: samplePointsBuffer,
        radiance,
    };
}

function createImage(radiance, resolution) {
    const imageData = new Uint8ClampedArray(resolution * resolution * 4);
    const pixels = radiance.length / 3;

    // Clamp values to [0, 255]
    const clamp = (v) => Math.min(255, Math.max(0, Math.round(v * 255)));

    for (let i = 0; i < pixels; i++) {
        imageData[i * 4 + 0] = clamp(radiance[i * 3 + 0]);
        imageData[i * 4 + 1] = clamp(radiance[i * 3 + 1]);
        imageData[i * 4 + 2] = clamp(radiance[i * 3 + 2]);
        imageData[i * 4 + 3] = 255;
    }

    return new ImageData(imageData, resolution, resolution);
}

function displayFrame(canvas, imageData, resolution) {
    const ctx = canvas.getContext("2d");

    // Scale canvas to display size while keeping internal resolution
    canvas.width = resolution;
    canvas.height = resolution;
    ctx.putImageData(imageData, 0, 0);
}

function drawFrameTimePlot(canvas, frameTimes) {
    // Match canvas internal resolution to display size
    const rect = canvas.getBoundingClientRect();
    if (canvas.width !== rect.width || canvas.height !== rect.height) {
        canvas.width = rect.width;
        canvas.height = rect.height;
    }

    const ctx = canvas.getContext("2d");
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    if (frameTimes.length < 2) return;

    const samples = frameTimes.slice(20);
    if (samples.length < 2) return;

    const times = samples.map((t) => t.time);
    const maxTime = Math.max(...times, 33.33);

    // Draw background
    ctx.fillStyle = "rgba(0, 0, 0, 0.7)";
    ctx.fillRect(0, 0, width, height);

    // Draw 60fps grid line (16.67ms)
    const y60fps = height - (16.67 / maxTime) * height;
    ctx.strokeStyle = "rgba(0, 255, 0, 0.4)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, y60fps);
    ctx.lineTo(width, y60fps);
    ctx.stroke();

    // Draw 30fps grid line (33.33ms)
    if (maxTime > 33.33) {
        const y30fps = height - (33.33 / maxTime) * height;
        ctx.strokeStyle = "rgba(255, 255, 0, 0.4)";
        ctx.beginPath();
        ctx.moveTo(0, y30fps);
        ctx.lineTo(width, y30fps);
        ctx.stroke();
    }

    // Draw frame time line
    ctx.strokeStyle = "#0f0";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    samples.forEach((t, i) => {
        const x = (i / (samples.length - 1)) * width;
        const y = height - (t.time / maxTime) * height;
        if (i === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    });
    ctx.stroke();
}

async function runHandler(
    modelFile,
    originalCanvas,
    predictedCanvas,
    device,
    resolution,
    frames,
    fpsOverlayElement,
    frameTimePlotCanvas,
    loopInput,
    resolve,
    currentRepeat,
    repeats,
) {
    const model = await loadModel(modelFile, device, resolution);
    log(`Loaded model with ${model.numberOfParameters()} parameters`);

    predictedCanvas.width = resolution;
    predictedCanvas.height = resolution;
    const canvasRenderer = await createCanvasRenderer(device, predictedCanvas);

    let frame = 0;
    const frameTimes = [];
    fpsOverlayElement.style.display = "block";
    const render = () => {
        const { samplePoints, radiance } = frames[frame];
        const imageData = createImage(radiance, resolution);
        displayFrame(originalCanvas, imageData, resolution);

        const start = performance.now();
        model.forward(samplePoints, () => {
            const now = performance.now();
            const time = now - start;
            frameTimes.push({ start, time });

            const latest = frameTimes.filter((t) => now - t.start < 500);
            const avg =
                latest.reduce((acc, t) => acc + t.time, 0) / latest.length;
            if (avg > 0) {
                const frameTime = `${avg.toFixed(1)} ms`;
                const fps = (1000 / avg).toFixed(1);
                fpsOverlayElement.innerText = `FPS: ${fps}\nTime: ${frameTime}`;
            } else {
                fpsOverlayElement.innerText = "FPS: /\nTime: /";
            }

            if (repeats > 1) {
                fpsOverlayElement.innerText += `\nRepeat: ${currentRepeat + 1}/${repeats}`;
            }

            drawFrameTimePlot(frameTimePlotCanvas, frameTimes);
        });
        model.renderToCanvas(canvasRenderer);

        frame++;
        if (loopInput.checked) {
            frame = frame % frames.length;
        }
        if (frame < frames.length) {
            requestAnimationFrame(render);
        } else {
            model.destroyBuffers();
            log("Finished inference for all frames");

            const times = frameTimes.map((t) => t.time);
            const avg = times.reduce((acc, t) => acc + t, 0) / times.length;
            log(`Avg. frame time: ${avg.toFixed(2)} ms`);
            log(`Min. frame time: ${Math.min(...times)} ms`);
            log(`Max. frame time: ${Math.max(...times)} ms`);

            resolve(avg);
        }
    };
    render();
}
