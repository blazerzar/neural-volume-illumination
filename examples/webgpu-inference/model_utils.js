"use strict";

import JSZip from "jszip";
import { RadianceFieldNetwork } from "./radiance_field_network.js";

export async function loadModel(modelFile, device, resolution) {
    const { positionTables, directionTables, fcWeights, fcBiases, metadata } =
        await readModelFile(modelFile, device);
    const modelArgs = metadata["model_args"];

    const response = await fetch("/radiance_field_network.wgsl");
    const shader = preprocessShader(await response.text(), modelArgs);

    const model = new RadianceFieldNetwork({
        positionTables,
        directionTables,
        fcWeights,
        fcBiases,
        device,
        modelArgs,
        resolution,
        shader,
    });
    return model;
}

function preprocessShader(shader, modelArgs) {
    const L = modelArgs["levels"];
    const F = modelArgs["feature_dim"];

    const mappings = {
        "1/* embedding size */": L * F * 2,
        "2/* vec embedding size */": (L * F * 2) / 4,
    };

    for (const [key, value] of Object.entries(mappings)) {
        shader = shader.replaceAll(key, value);
    }

    return shader;
}

async function readModelFile(modelFile, device) {
    const zip = await JSZip.loadAsync(modelFile);

    const metadataString = await zip.files["metadata.json"].async("string");
    const metadata = JSON.parse(metadataString);

    let positionTables;
    let directionTables;
    for (const param of ["position", "direction"]) {
        let tables = [];
        let tablesSizeBytes = 0;

        for (let level = 0; level < metadata["model_args"]["levels"]; level++) {
            const name = `${param}_encoding_tables_${level}_weight`;
            const content = await zip.files[name + ".bin"].async("arraybuffer");
            const array = new Float32Array(content);
            assert(array.byteLength === metadata[name]["total_bytes"]);
            assert(array.length === metadata[name]["num_elements"]);

            tables.push(array);
            tablesSizeBytes += array.byteLength;
        }

        const buffer = device.createBuffer({
            label: `${param} encoding tables`,
            size: tablesSizeBytes,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        let offset = 0;
        for (const table of tables) {
            device.queue.writeBuffer(buffer, offset, table);
            offset += table.byteLength;
        }

        if (param === "position") {
            positionTables = buffer;
        } else {
            directionTables = buffer;
        }
    }

    let fcWeights;
    let fcBiases;
    for (const param of ["weight", "bias"]) {
        let weights = [];
        let weightsSizeBytes = 0;

        for (let layer = 0; layer < metadata["model_args"]["layers"]; layer++) {
            const name = `fc_${layer}_${param}`;
            const content = await zip.files[name + ".bin"].async("arraybuffer");
            let array = new Float32Array(content);
            assert(array.byteLength === metadata[name]["total_bytes"]);
            assert(array.length === metadata[name]["num_elements"]);

            if (param === "weight") {
                const [outFeatures, inFeatures] = metadata[name]["size"];
                array = padWeightRows(array, inFeatures, outFeatures);
            } else {
                array = padBias(array);
            }

            weights.push(array);
            weightsSizeBytes += array.byteLength;
        }

        const buffer = device.createBuffer({
            label: `fc ${param}`,
            size: weightsSizeBytes,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        let offset = 0;
        for (const weight of weights) {
            device.queue.writeBuffer(buffer, offset, weight);
            offset += weight.byteLength;
        }

        if (param === "weight") {
            fcWeights = buffer;
        } else {
            fcBiases = buffer;
        }
    }

    return { positionTables, directionTables, fcWeights, fcBiases, metadata };
}

export async function createCanvasRenderer(device, canvas) {
    const context = canvas.getContext("webgpu");
    const format = navigator.gpu.getPreferredCanvasFormat();

    context.configure({ device, format });

    const response = await fetch("/canvas_renderer.wgsl");
    const shader = await response.text();

    const module = device.createShaderModule({
        label: "canvas renderer shader module",
        code: shader,
    });

    const pipeline = device.createRenderPipeline({
        label: "canvas renderer pipeline",
        layout: "auto",
        vertex: {
            module: module,
        },
        fragment: {
            module: module,
            targets: [{ format }],
        },
        primitive: {
            topology: "triangle-list",
        },
    });

    const createBindGroup = (inputTextureView) => {
        return device.createBindGroup({
            label: "canvas renderer bind group",
            layout: pipeline.getBindGroupLayout(0),
            entries: [{ binding: 0, resource: inputTextureView }],
        });
    };

    return {
        context,
        pipeline,
        createBindGroup,
    };
}

function assert(condition, message) {
    if (!condition) {
        throw new Error(message || "Assertion failed");
    }
}

function padWeightRows(array, inFeatures, outFeatures) {
    const paddedInFeatures = Math.ceil(inFeatures / 4) * 4;
    const padded = new Float32Array(outFeatures * paddedInFeatures);

    for (let row = 0; row < outFeatures; row++) {
        const srcOffset = row * inFeatures;
        const dstOffset = row * paddedInFeatures;
        padded.set(
            array.subarray(srcOffset, srcOffset + inFeatures),
            dstOffset,
        );
    }
    return padded;
}

function padBias(array) {
    const paddedLen = Math.ceil(array.length / 4) * 4;
    const padded = new Float32Array(paddedLen);
    padded.set(array);
    return padded;
}
