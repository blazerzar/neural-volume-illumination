override WORKGROUP_SIZE: u32;
override PIXELS_PER_THREAD: u32;
override RESOLUTION: u32;
override LEVELS: u32;           // L
override HASH_TABLE_SIZE: u32;  // T
override FEATURE_DIM: u32;      // F
override LAYERS: u32;
override HIDDEN_DIM: u32;

// Uniforms
@group(0) @binding(0) var<storage, read> posGridSizes: array<u32>;  // N_l
@group(0) @binding(1) var<storage, read> dirGridSizes: array<u32>;  // N_l

// Position and direciton encoding tables
@group(1) @binding(0) var<storage, read> posTables: array<f32>;
@group(1) @binding(1) var<storage, read> dirTables: array<f32>;

// FC layers weights and biases
@group(1) @binding(2) var<storage, read> fcWeights: array<vec4f>;
@group(1) @binding(3) var<storage, read> fcBiases: array<vec4f>;

// Input and output
struct SamplePoint {
    pos: vec3f,
    dir: vec2f,
};

@group(2) @binding(0) var<storage, read> samplePoints: array<SamplePoint>;
@group(2) @binding(1) var output: texture_storage_2d<rgba32float, write>;

fn oneToOnePosition(corner: vec3f, gridSize: u32) -> u32 {
    let res = f32(gridSize + 1u);
    return u32(corner.x * res * res + corner.y * res + corner.z);
}

fn hashPosition(corner: vec3f, gridSize: u32) -> u32 {
    return u32(corner.x + corner.y * 2654435761 + corner.z * 805459861) % HASH_TABLE_SIZE;
}

fn encodePosition(
    position: vec3f, output: ptr<function, array<f32, 1/* embedding size */>>,
) {
    var tableOffset = 0u;

    for (var level = 0u; level < LEVELS; level++) {
        let gridSize = posGridSizes[level];

        let cornerTLB = floor(position * f32(gridSize));
        let cornerTLF = cornerTLB + vec3f(0, 0, 1);
        let cornerTRB = cornerTLB + vec3f(0, 1, 0);
        let cornerTRF = cornerTLB + vec3f(0, 1, 1);
        let cornerBLB = cornerTLB + vec3f(1, 0, 0);
        let cornerBLF = cornerTLB + vec3f(1, 0, 1);
        let cornerBRB = cornerTLB + vec3f(1, 1, 0);
        let cornerBRF = cornerTLB + vec3f(1, 1, 1);

        var tableSize = (gridSize + 1) * (gridSize + 1) * (gridSize + 1);

        var indexTLB: u32;
        var indexTLF: u32;
        var indexTRB: u32;
        var indexTRF: u32;
        var indexBLB: u32;
        var indexBLF: u32;
        var indexBRB: u32;
        var indexBRF: u32;
        if tableSize <= HASH_TABLE_SIZE {
            indexTLB = oneToOnePosition(cornerTLB, gridSize);
            indexTLF = oneToOnePosition(cornerTLF, gridSize);
            indexTRB = oneToOnePosition(cornerTRB, gridSize);
            indexTRF = oneToOnePosition(cornerTRF, gridSize);
            indexBLB = oneToOnePosition(cornerBLB, gridSize);
            indexBLF = oneToOnePosition(cornerBLF, gridSize);
            indexBRB = oneToOnePosition(cornerBRB, gridSize);
            indexBRF = oneToOnePosition(cornerBRF, gridSize);
        } else {
            indexTLB = hashPosition(cornerTLB, gridSize);
            indexTLF = hashPosition(cornerTLF, gridSize);
            indexTRB = hashPosition(cornerTRB, gridSize);
            indexTRF = hashPosition(cornerTRF, gridSize);
            indexBLB = hashPosition(cornerBLB, gridSize);
            indexBLF = hashPosition(cornerBLF, gridSize);
            indexBRB = hashPosition(cornerBRB, gridSize);
            indexBRF = hashPosition(cornerBRF, gridSize);

            tableSize = HASH_TABLE_SIZE;
        }

        let w = position * f32(gridSize) - vec3f(cornerTLB);

        let weightTLB = (1 - w.x) * (1 - w.y) * (1 - w.z);  // (0,0,0)
        let weightTLF = (1 - w.x) * (1 - w.y) * w.z;        // (0,0,1)
        let weightTRB = (1 - w.x) * w.y * (1 - w.z);        // (0,1,0)
        let weightTRF = (1 - w.x) * w.y * w.z;              // (0,1,1)
        let weightBLB = w.x * (1 - w.y) * (1 - w.z);        // (1,0,0)
        let weightBLF = w.x * (1 - w.y) * w.z;              // (1,0,1)
        let weightBRB = w.x * w.y * (1 - w.z);              // (1,1,0)
        let weightBRF = w.x * w.y * w.z;                    // (1,1,1)

        let offset = level * FEATURE_DIM;
        for (var f = 0u; f < FEATURE_DIM; f++) {
            let embTLB = posTables[tableOffset + indexTLB * FEATURE_DIM + f];
            let embTLF = posTables[tableOffset + indexTLF * FEATURE_DIM + f];
            let embTRB = posTables[tableOffset + indexTRB * FEATURE_DIM + f];
            let embTRF = posTables[tableOffset + indexTRF * FEATURE_DIM + f];
            let embBLB = posTables[tableOffset + indexBLB * FEATURE_DIM + f];
            let embBLF = posTables[tableOffset + indexBLF * FEATURE_DIM + f];
            let embBRB = posTables[tableOffset + indexBRB * FEATURE_DIM + f];
            let embBRF = posTables[tableOffset + indexBRF * FEATURE_DIM + f];
            let value = embTLB * weightTLB + embTLF * weightTLF + embTRB * weightTRB + embTRF * weightTRF + embBLB * weightBLB + embBLF * weightBLF + embBRB * weightBRB + embBRF * weightBRF;
            (*output)[offset + f] = value;
        }

        tableOffset = tableOffset + tableSize * FEATURE_DIM;
    }
}

fn oneToOneDirection(corner: vec2f, gridSize: u32) -> u32 {
    let res = f32(gridSize + 1u);
    return u32(corner.x * res + corner.y);
}

fn hashDirection(corner: vec2f, gridSize: u32) -> u32 {
    return u32(corner.x + corner.y * 2654435761) % HASH_TABLE_SIZE;
}

fn encodeDirection(
    direction: vec2f, output: ptr<function, array<f32, 1/* embedding size */>>
) {
    var tableOffset = 0u;

    for (var level = 0u; level < LEVELS; level++) {
        let gridSize = dirGridSizes[level];

        let cornerTL = floor(direction * f32(gridSize));
        let cornerTR = cornerTL + vec2f(0, 1);
        let cornerBL = cornerTL + vec2f(1, 0);
        let cornerBR = cornerTL + vec2f(1, 1);

        var tableSize = (gridSize + 1) * (gridSize + 1);

        var indexTL: u32;
        var indexTR: u32;
        var indexBL: u32;
        var indexBR: u32;
        if tableSize <= HASH_TABLE_SIZE {
            indexTL = oneToOneDirection(cornerTL, gridSize);
            indexTR = oneToOneDirection(cornerTR, gridSize);
            indexBL = oneToOneDirection(cornerBL, gridSize);
            indexBR = oneToOneDirection(cornerBR, gridSize);
        } else {
            indexTL = hashDirection(cornerTL, gridSize);
            indexTR = hashDirection(cornerTR, gridSize);
            indexBL = hashDirection(cornerBL, gridSize);
            indexBR = hashDirection(cornerBR, gridSize);

            tableSize = HASH_TABLE_SIZE;
        }

        let w = direction * f32(gridSize) - vec2f(cornerTL);

        let weightTL = (1 - w.x) * (1 - w.y);  // (0,0)
        let weightTR = (1 - w.x) * w.y;        // (0,1)
        let weightBL = w.x * (1 - w.y);        // (1,0)
        let weightBR = w.x * w.y;              // (1,1)

        let offset = (LEVELS + level) * FEATURE_DIM;
        for (var f = 0u; f < FEATURE_DIM; f++) {
            let embTL = dirTables[tableOffset + indexTL * FEATURE_DIM + f];
            let embTR = dirTables[tableOffset + indexTR * FEATURE_DIM + f];
            let embBL = dirTables[tableOffset + indexBL * FEATURE_DIM + f];
            let embBR = dirTables[tableOffset + indexBR * FEATURE_DIM + f];
            let value = embTL * weightTL + embTR * weightTR + embBL * weightBL + embBR * weightBR;
            (*output)[offset + f] = value;
        }

        tableOffset = tableOffset + tableSize * FEATURE_DIM;
    }
}

fn multiLayerPerceptron(
    embeddings: ptr<function, array<vec4f, 2/* vec embedding size */>>
) -> vec3f {
    var intermediateEven: array<vec4f, 4>;
    var intermediateOdd: array<vec4f, 4>;

    let inputSize = LEVELS * FEATURE_DIM * 2 / 4;
    let hiddenSize = HIDDEN_DIM / 4;
    let vecSize = 4u;

    var weightOffset = 0u;
    var biasOffset = 0u;

    for (var i = 0u; i < hiddenSize; i++) {
        var a = vec4f(0);
        for (var k = 0u; k < vecSize; k++) {
            for (var j = 0u; j < inputSize; j++) {
                let w = fcWeights[(i * vecSize + k) * inputSize + j];
                a[k] = a[k] + dot((*embeddings)[j], w,);
            }
        }
        intermediateEven[i] = max(a + fcBiases[i], vec4f(0));
    }
    weightOffset = weightOffset + inputSize * HIDDEN_DIM;
    biasOffset = biasOffset + hiddenSize;

    var in = intermediateEven;
    var out = intermediateOdd;
    var tmp = in;

    for (var layer = 0u; layer < LAYERS - 2; layer++) {
        for (var i = 0u; i < hiddenSize; i++) {
            var a = vec4f(0);
            for (var k = 0u; k < vecSize; k++) {
                for (var j = 0u; j < hiddenSize; j++) {
                    let w = fcWeights[weightOffset + (i * vecSize + k) * hiddenSize + j];
                    a[k] = a[k] + dot(in[j], w);
                }
            }
            out[i] = max(a + fcBiases[biasOffset + i], vec4f(0));
        }
        weightOffset = weightOffset + hiddenSize * HIDDEN_DIM;
        biasOffset = biasOffset + hiddenSize;

        tmp = in;
        in = out;
        out = tmp;
    }

    var color: vec3f;
    for (var i = 0u; i < 3; i++) {
        var a = 0.0;
        for (var j = 0u; j < hiddenSize; j++) {
            let w = fcWeights[weightOffset + i * hiddenSize + j];
            a = a + dot(in[j], w);
        }
        let value = min(max(a + fcBiases[biasOffset + i / 4][i], 0), 1);
        color[i] = value;
    }

    return color;
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn forward(
    @builtin(global_invocation_id) globalId: vec3u,
    @builtin(local_invocation_index) localIndex: u32,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    var pixelIndex = globalId.x;
    let pixels = RESOLUTION * RESOLUTION;
    let embeddingVecSize = LEVELS * FEATURE_DIM * 2 / 4;

    while pixelIndex < pixels {
        let input = samplePoints[pixelIndex];
        let x = pixelIndex % RESOLUTION;
        let y = pixelIndex / RESOLUTION;

        if all(input.pos == vec3f(0)) {
            textureStore(output, vec2u(x, y), vec4f(0, 0, 0.5, 1));
            pixelIndex = pixelIndex + num_workgroups.x;
            continue;
        }

        var embeddings: array<f32, 1/* embedding size */>;
        encodePosition(input.pos, &embeddings);
        encodeDirection(input.dir, &embeddings);

        // Pack embeddings into array of vectors
        var embeddingsVec: array<vec4f, 2/* vec embedding size */>;
        for (var i = 0u; i < embeddingVecSize; i++) {
            embeddingsVec[i].x = embeddings[i * 4 + 0];
            embeddingsVec[i].y = embeddings[i * 4 + 1];
            embeddingsVec[i].z = embeddings[i * 4 + 2];
            embeddingsVec[i].w = embeddings[i * 4 + 3];
        }

        let color = multiLayerPerceptron(&embeddingsVec);

        textureStore(output, vec2u(x, y), vec4f(color, 1.0));
        pixelIndex = pixelIndex + num_workgroups.x;
    }
}
