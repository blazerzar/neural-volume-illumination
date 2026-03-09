"use strict";

const PIXEL_VALUES = 8;

let logsElement = undefined;

function log(line) {
    if (logsElement === undefined) {
        return;
    }
    logsElement.innerText += line + "\n";
    logsElement.scrollTop = logsElement.scrollHeight;
}

function main() {
    let parameters = {};
    let session = undefined;
    let Xs = [];
    let ys = [];
    let validIndices = [];

    const zipFileInput = document.getElementById("zip-file");
    const modelFileInput = document.getElementById("model-file");
    const loadButton = document.getElementById("load-button");

    logsElement = document.getElementById("logs");

    const modelControls = document.getElementById("model-controls");
    const runButton = document.getElementById("run-button");
    const frameSlider = document.getElementById("frame");

    const originalCanvas = document.getElementById("original");
    const predictedCanvas = document.getElementById("predicted");

    loadButton.onclick = async () => {
        if (
            zipFileInput.files.length === 0 ||
            modelFileInput.files.length === 0
        ) {
            alert("Load data and model");
            return;
        }

        log("Loading model ...");
        const modelFile = modelFileInput.files[0];
        const arrayBuffer = await modelFile.arrayBuffer();
        session = await ort.InferenceSession.create(arrayBuffer, {
            graphOptimizationLevel: "all",
            logSeverityLevel: 0,
            executionProviders: ["webgpu"],
        });
        log("Model loaded");

        log("Loading ground truth data ...");
        const zipFile = zipFileInput.files[0];
        const data = await readGroundTruthZip(zipFile);
        parameters = data.parameters;
        log("Ground truth data loaded");
        log(`Frames: ${data.arrays.length}`);
        log(`Parameters: ${JSON.stringify(parameters, null, 2)}`);

        log("Preprocessing data ...");
        for (let i = 0; i < data.arrays.length; i++) {
            const preprocessed = preprocessFrame(data.arrays[i]);
            Xs.push(preprocessed.X);
            ys.push(preprocessed.y);
            validIndices.push(preprocessed.validIndices);
            log(`Preprocessed frame ${i}`);
        }
        log("Data ready");

        frameSlider.min = 0;
        frameSlider.max = data.arrays.length - 1;
        frameSlider.value = 0;
        modelControls.style.display = "flex";
    };

    runButton.onclick = async () => {
        const index = frameSlider.value;
        const predicted = await predictFrame(
            Xs[index],
            validIndices[index],
            session,
        );

        if (predicted !== undefined) {
            const originalImage = createImageFromValues(
                ys[index],
                validIndices[index],
                parameters.resolution,
            );
            const predictedImage = createImageFromValues(
                predicted,
                validIndices[index],
                parameters.resolution,
            );

            displayFrame(originalCanvas, originalImage, parameters.resolution);
            displayFrame(
                predictedCanvas,
                predictedImage,
                parameters.resolution,
            );
        }
    };
}

window.onload = () => main();

async function readGroundTruthZip(arrayBuffer) {
    const zip = await JSZip.loadAsync(arrayBuffer);
    const files = Object.keys(zip.files).sort();
    const arrays = [];
    let params = {};

    for (const filename of files) {
        if (filename.endsWith(".bin")) {
            const content = await zip.files[filename].async("arraybuffer");
            const float32Data = new Float32Array(content);
            arrays.push(float32Data);
        } else if (filename === "parameters.json") {
            const content = await zip.files[filename].async("string");
            params = JSON.parse(content);
        }
        log(`Parsed ${filename}`);
    }

    return { arrays, parameters: params };
}

function preprocessFrame(float32Data) {
    const totalPixels = float32Data.length / PIXEL_VALUES;
    const validIndices = [];

    // Filter valid pixels (non-zero and non-NaN)
    for (let i = 0; i < totalPixels; i++) {
        const offset = i * PIXEL_VALUES;
        let isValid = true;
        let hasNonZero = false;

        for (let j = 0; j < PIXEL_VALUES; j++) {
            const val = float32Data[offset + j];
            if (Number.isNaN(val)) {
                isValid = false;
                break;
            }
            if (val !== 0) {
                hasNonZero = true;
            }
        }

        if (isValid && hasNonZero) {
            validIndices.push(i);
        }
    }

    // Extract and normalize features
    const X = new Float32Array(validIndices.length * 5);
    const y = new Float32Array(validIndices.length * 3);

    for (let i = 0; i < validIndices.length; i++) {
        const srcOffset = validIndices[i] * PIXEL_VALUES;
        const destOffset = i * 5;
        const destOffsetY = i * 3;

        // Features (x, y, z, azimuth, elevation)
        X[destOffset + 0] = float32Data[srcOffset + 0]; // x
        X[destOffset + 1] = float32Data[srcOffset + 1]; // y
        X[destOffset + 2] = float32Data[srcOffset + 2]; // z
        // Normalize azimuth to [0, 1]
        X[destOffset + 3] =
            (float32Data[srcOffset + 3] + Math.PI) / (2.01 * Math.PI);
        // Normalize elevation to [0, 1]
        X[destOffset + 4] = float32Data[srcOffset + 4] / Math.PI;

        // Target values (r, g, b)
        y[destOffsetY + 0] = float32Data[srcOffset + 5];
        y[destOffsetY + 1] = float32Data[srcOffset + 6];
        y[destOffsetY + 2] = float32Data[srcOffset + 7];
    }

    const gpuBuffer = ort.env.webgpu.device.createBuffer({
        size: X.byteLength,
        usage:
            GPUBufferUsage.STORAGE |
            GPUBufferUsage.COPY_DST |
            GPUBufferUsage.COPY_SRC,
    });

    ort.env.webgpu.device.queue.writeBuffer(gpuBuffer, 0, X);
    const tensor = ort.Tensor.fromGpuBuffer(gpuBuffer, {
        dataType: "float32",
        dims: [validIndices.length, 5],
    });

    return { X: tensor, y, validIndices };
}

async function predictFrame(X, validIndices, session) {
    const inputName = session.inputNames[0];
    const outputName = session.outputNames[0];

    try {
        const results = await session.run({ [inputName]: X });
        const output = results[outputName];

        const predictionData = output.data.slice(0, validIndices.length * 3);
        return new Float32Array(predictionData);
    } catch (error) {
        console.error(error);
    }
}

function createImageFromValues(values, validIndices, resolution) {
    const imageData = new Uint8ClampedArray(resolution * resolution * 4);

    for (let i = 0; i < validIndices.length; i++) {
        const pixelIndex = validIndices[i];
        const destOffset = pixelIndex * 4;

        // Clamp values to [0, 255]
        imageData[destOffset + 0] = Math.min(
            255,
            Math.max(0, Math.round(values[i * 3 + 0] * 255)),
        );
        imageData[destOffset + 1] = Math.min(
            255,
            Math.max(0, Math.round(values[i * 3 + 1] * 255)),
        );
        imageData[destOffset + 2] = Math.min(
            255,
            Math.max(0, Math.round(values[i * 3 + 2] * 255)),
        );
        imageData[destOffset + 3] = 255;
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
