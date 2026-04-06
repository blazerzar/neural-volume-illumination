"use strict";

export class RadianceFieldNetwork {
    constructor({
        positionTables,
        directionTables,
        fcWeights,
        fcBiases,
        device,
        modelArgs,
        resolution,
        shader,
    }) {
        this.positionTables = positionTables;
        this.directionTables = directionTables;
        this.fcWeights = fcWeights;
        this.fcBiases = fcBiases;
        this.device = device;
        this.modelArgs = modelArgs;
        this.resolution = resolution;

        this._initializeUniforms();
        this._initializeForwardPassBuffers();
        this._createPrograms(shader);
    }

    _getWorkgroupCount() {
        return [
            Math.ceil(
                (this.resolution * this.resolution) /
                    (this.constants["WORKGROUP_SIZE"] *
                        this.constants["PIXELS_PER_THREAD"]),
            ),
        ];
    }

    _initializeUniforms() {
        const L = this.modelArgs["levels"];
        const T = 2 ** this.modelArgs["hash_table_size"];
        const F = this.modelArgs["feature_dim"];

        const posGridSizesArray = createGridSizes(
            this.modelArgs["pos_coarse_res"],
            this.modelArgs["pos_fine_res"],
            L,
        );
        const dirGridSizesArray = createGridSizes(
            this.modelArgs["dir_coarse_res"],
            this.modelArgs["dir_fine_res"],
            L,
        );

        this.constants = {
            ...getComputeConfig(),
            RESOLUTION: this.resolution,
            LEVELS: L,
            HASH_TABLE_SIZE: T,
            FEATURE_DIM: F,
            LAYERS: this.modelArgs["layers"],
            HIDDEN_DIM: this.modelArgs["hidden_dim"],
            POS_FIRST_HASH_LEVEL: findFirstHashLevel(posGridSizesArray, T, 3),
            DIR_FIRST_HASH_LEVEL: findFirstHashLevel(dirGridSizesArray, T, 2),
        };

        this.posGridSizes = this.device.createBuffer({
            label: "radiance field network position grid sizes",
            size: L * 4,
            usage:
                GPUBufferUsage.STORAGE |
                GPUBufferUsage.UNIFORM |
                GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.posGridSizes, 0, posGridSizesArray);

        this.dirGridSizes = this.device.createBuffer({
            label: "radiance field network direction grid sizes",
            size: L * 4,
            usage:
                GPUBufferUsage.STORAGE |
                GPUBufferUsage.UNIFORM |
                GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.dirGridSizes, 0, dirGridSizesArray);

        const posTableOffsetsArray = createTableOffsets(
            posGridSizesArray,
            T,
            this.constants["POS_FIRST_HASH_LEVEL"],
            3,
        );
        const dirTableOffsetsArray = createTableOffsets(
            dirGridSizesArray,
            T,
            this.constants["DIR_FIRST_HASH_LEVEL"],
            2,
        );

        this.posTableOffsets = this.device.createBuffer({
            label: "radiance field network position table offsets",
            size: L * 4,
            usage:
                GPUBufferUsage.STORAGE |
                GPUBufferUsage.UNIFORM |
                GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(
            this.posTableOffsets,
            0,
            posTableOffsetsArray,
        );

        this.dirTableOffsets = this.device.createBuffer({
            label: "radiance field network direction table offsets",
            size: L * 4,
            usage:
                GPUBufferUsage.STORAGE |
                GPUBufferUsage.UNIFORM |
                GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(
            this.dirTableOffsets,
            0,
            dirTableOffsetsArray,
        );
    }

    _initializeForwardPassBuffers() {
        const L = this.modelArgs["levels"];
        const F = this.modelArgs["feature_dim"];

        this.outputTexture = this.device.createTexture({
            label: "radiance field network output texture",
            size: [this.resolution, this.resolution],
            format: "rgba32float",
            usage:
                GPUTextureUsage.STORAGE_BINDING |
                GPUTextureUsage.TEXTURE_BINDING,
        });
        this.outputView = this.outputTexture.createView();
        this.embeddingsBuffer = this.device.createBuffer({
            label: "radiance field network embeddings buffer",
            // 2 ... position and direction, 4 ... f32 size
            size: this.resolution * this.resolution * L * F * 2 * 4,
            usage: GPUBufferUsage.STORAGE,
        });
    }

    _createPrograms(shader) {
        const module = this.device.createShaderModule({
            label: "radiance field network shader module",
            code: shader,
        });

        this.pipeline = this.device.createComputePipeline({
            label: "radiance field network pipeline",
            layout: "auto",
            compute: {
                module: module,
                constants: this.constants,
            },
        });

        this.uniformsBindGroup = this.device.createBindGroup({
            label: "radiance field network uniforms bind group",
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.posGridSizes } },
                { binding: 1, resource: { buffer: this.dirGridSizes } },
                { binding: 2, resource: { buffer: this.posTableOffsets } },
                { binding: 3, resource: { buffer: this.dirTableOffsets } },
            ],
        });

        this.modelBindGroup = this.device.createBindGroup({
            label: "radiance field network model bind group",
            layout: this.pipeline.getBindGroupLayout(1),
            entries: [
                {
                    binding: 0,
                    resource: { buffer: this.positionTables },
                },
                {
                    binding: 1,
                    resource: { buffer: this.directionTables },
                },
                {
                    binding: 2,
                    resource: { buffer: this.fcWeights },
                },
                {
                    binding: 3,
                    resource: { buffer: this.fcBiases },
                },
            ],
        });
    }

    _createForwardPassBindGroup(samplePoints) {
        return this.device.createBindGroup({
            label: "radiance field network forward pass bind group",
            layout: this.pipeline.getBindGroupLayout(2),
            entries: [
                { binding: 0, resource: { buffer: samplePoints } },
                { binding: 1, resource: this.outputView },
                { binding: 2, resource: { buffer: this.embeddingsBuffer } },
            ],
        });
    }

    forward(samplePoints, doneCallback) {
        const forwardPassBindGroup =
            this._createForwardPassBindGroup(samplePoints);

        const encoder = this.device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, this.uniformsBindGroup);
        pass.setBindGroup(1, this.modelBindGroup);
        pass.setBindGroup(2, forwardPassBindGroup);
        pass.dispatchWorkgroups(...this._getWorkgroupCount());
        pass.end();

        this.device.queue.submit([encoder.finish()]);
        this.device.queue.onSubmittedWorkDone().then(doneCallback);
    }

    numberOfParameters() {
        let parameters = 0;
        parameters += this.positionTables.size / 4;
        parameters += this.directionTables.size / 4;
        parameters += this.fcWeights.size / 4;
        parameters += this.fcBiases.size / 4;
        return parameters;
    }

    destroyBuffers() {
        this.positionTables.destroy();
        this.directionTables.destroy();
        this.fcWeights.destroy();
        this.fcBiases.destroy();
        this.posGridSizes.destroy();
        this.dirGridSizes.destroy();
        this.posTableOffsets.destroy();
        this.dirTableOffsets.destroy();
        this.outputTexture.destroy();
        this.embeddingsBuffer.destroy();
    }

    renderToCanvas(canvasRenderer) {
        const canvasTexture = canvasRenderer.context.getCurrentTexture();

        const encoder = this.device.createCommandEncoder();
        const pass = encoder.beginRenderPass({
            colorAttachments: [
                {
                    view: canvasTexture.createView(),
                    clearValue: [0, 0, 0, 1],
                    loadOp: "clear",
                    storeOp: "store",
                },
            ],
        });

        pass.setPipeline(canvasRenderer.pipeline);
        pass.setBindGroup(0, canvasRenderer.createBindGroup(this.outputView));
        pass.draw(6);
        pass.end();

        this.device.queue.submit([encoder.finish()]);
    }
}

function createGridSizes(coarseRes, fineRes, levels) {
    const b = Math.exp(
        (Math.log(fineRes) - Math.log(coarseRes)) / (levels - 1),
    );
    const grids = new Uint32Array(levels);
    for (let i = 0; i < levels; i++) {
        grids[i] = Math.floor(coarseRes * b ** i);
    }
    return grids;
}

function findFirstHashLevel(gridSizes, hashTableSize, dimensions) {
    for (let i = 0; i < gridSizes.length; i++) {
        const gridSize = gridSizes[i];
        const tableSize = Math.pow(gridSize + 1, dimensions);
        if (tableSize > hashTableSize) {
            return i;
        }
    }
    return gridSizes.length;
}

function createTableOffsets(
    gridSizes,
    hashTableSize,
    firstHashLevel,
    dimensions,
) {
    const offsets = new Uint32Array(gridSizes.length);
    let offset = 0;
    for (let i = 0; i < gridSizes.length; i++) {
        offsets[i] = offset;
        if (i < firstHashLevel) {
            // One-to-one indexing: table size is (gridSize + 1)^dimensions
            offset += Math.pow(gridSizes[i] + 1, dimensions);
        } else {
            // Hash indexing: table size is hashTableSize
            offset += hashTableSize;
        }
    }
    return offsets;
}

function getComputeConfig() {
    const defaultWorkgroup = 64;
    const defaultPixels = 1;

    const parseValue = (id, defaultValue) => {
        const val = document.getElementById(id).value;
        if (val === "") return defaultValue;
        const parsed = parseInt(val, 10);
        return isNaN(parsed) ? defaultValue : parsed;
    };

    return {
        WORKGROUP_SIZE: parseValue("workgroup-size", defaultWorkgroup),
        PIXELS_PER_THREAD: parseValue("pixels-thread", defaultPixels),
    };
}
