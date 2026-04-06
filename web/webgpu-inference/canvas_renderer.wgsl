// Canvas renderer shaders for GPU-only rendering
// Renders a texture to the canvas using two triangles (fullscreen quad)

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
}

@vertex
fn vert(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
    // Two triangles forming a quad
    var positions = array<vec2f, 6>(
        vec2f(-1.0, -1.0),  // bottom-left
        vec2f(-1.0,  1.0),  // top-left
        vec2f( 1.0,  1.0),  // top-right
        vec2f(-1.0, -1.0),  // bottom-left
        vec2f( 1.0,  1.0),  // top-right
        vec2f( 1.0, -1.0),  // bottom-right
    );

    let pos = positions[vertexIndex];
    var output: VertexOutput;
    output.position = vec4f(pos, 0.0, 1.0);
    let uv = vec2f(pos.x, -pos.y);
    output.uv = (uv + 1.0) * 0.5;  // Convert [-1,1] to [0,1]
    return output;
}

@group(0) @binding(0) var inputTexture: texture_2d<f32>;

@fragment
fn frag(output: VertexOutput) -> @location(0) vec4f {
    let dims = textureDimensions(inputTexture);
    let texel = textureLoad(inputTexture, vec2i(output.uv * vec2f(dims)), 0);
    return vec4f(texel.rgb, 1.0);
}
