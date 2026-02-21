# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a master's thesis project for "Neural caching of dynamic volume illumination" - a volumetric path tracing framework that combines real-time rendering with neural network-based caching. The project consists of:

1. **VPT Framework** (`vpt/`) - A WebGL 2/WebGPU-based volumetric path tracing framework with no external dependencies (except express for serving)
2. **Examples** (`examples/`) - Experimental neural network implementations using TensorFlow.js

## Build and Run Commands

**IMPORTANT:** This project uses `nodemon` for automatic rebuilding during development. Changes to source files will be automatically built. Do NOT manually run build commands - nodemon handles this.

### VPT Framework

```bash
# Start development server with auto-rebuild (nodemon)
nodemon

# Manual build (ONLY if nodemon is not running)
cd vpt
npm run build

# Clean build artifacts
make clean
```

The build process:
- Copies JS/HTML/CSS/JSON files to `build/static/`
- Parses and bundles GLSL/WGSL shaders into `build/shaders.json` and `build/mixins.json`
- Bundles WGSL shaders into `build/shaders-wgsl.json` and `build/mixins-wgsl.json`

### Examples (TensorFlow.js)

The examples use TensorFlow.js with WebGPU backend and can be run by opening the HTML file in a browser with WebGPU support.

## Architecture

### VPT Framework Structure

```
vpt/
├── src/
│   ├── js/                      # JavaScript modules
│   │   ├── Application.js       # Main entry point, coordinates all components
│   │   ├── RenderingContext.js  # WebGL 2 rendering context
│   │   ├── WebGPURenderingContext.js  # WebGPU rendering context
│   │   ├── Volume.js            # 3D volume texture management
│   │   ├── Node.js              # Scene graph nodes
│   │   ├── PerspectiveCamera.js # Camera component
│   │   ├── Transform.js         # Transformation matrices
│   │   ├── renderers/           # Volume rendering algorithms
│   │   │   ├── AbstractRenderer.js
│   │   │   ├── MIPRenderer.js   # Maximum Intensity Projection
│   │   │   ├── MCMRenderer.js   # Monte Carlo Path Tracing
│   │   │   ├── EAMRenderer.js   # Emission-Absorption
│   │   │   └── ... (8 renderers total)
│   │   ├── tonemappers/         # HDR tone mapping algorithms
│   │   │   ├── AbstractToneMapper.js
│   │   │   ├── ReinhardToneMapper.js
│   │   │   ├── AcesToneMapper.js
│   │   │   └── ... (10 tone mappers)
│   │   ├── dialogs/             # UI dialogs (MainDialog, VolumeLoadDialog, etc.)
│   │   ├── ui/                  # Custom UI components
│   │   ├── readers/             # Volume data format readers (BVP, RAW, ZIP)
│   │   ├── loaders/             # Data loading (Ajax, Blob)
│   │   └── animators/           # Camera animations
│   ├── glsl/                    # GLSL shaders (WebGL 2)
│   │   ├── mixins/              # Reusable GLSL code snippets
│   │   │   ├── random/          # Random number generators and distributions
│   │   │   ├── hash/            # Hash functions
│   │   │   └── distribution/    # Sampling distributions
│   │   └── renderers/           # Renderer-specific shaders
│   ├── wgsl/                    # WGSL shaders (WebGPU)
│   │   └── [mirrors glsl structure]
│   └── index.html
├── bin/                         # Build tools and servers
│   ├── packer                   # Custom bundler
│   ├── server-express           # Express server
│   └── watcher                  # File watcher for dev mode
└── build/                       # Generated build output
```

### Key Design Patterns

**Renderer Architecture:**
- `AbstractRenderer` (vpt/src/js/renderers/AbstractRenderer.js) - Base class for all rendering algorithms
- Three-buffer system: `_frameBuffer` (generate), `_accumulationBuffer` (integrate), `_renderBuffer` (display)
- Supports progressive rendering through accumulation buffer swapping
- Each renderer extends `PropertyBag` for runtime-tweakable parameters

**Shader System:**
- Modular GLSL/WGSL with mixins for code reuse
- Shaders fetched at runtime as JSON (`shaders.json`, `mixins.json`)
- Mixins organized by category: random number generation, sampling distributions, color space conversions
- Custom packer tool parses `#include` directives and bundles shaders

**Volume Loading:**
- `Volume` class manages 3D texture uploads
- `Reader` classes handle different volume formats
- Block-based loading with progress events for large datasets
- Supports multiple modalities per volume

**Tone Mapping:**
- `AbstractToneMapper` base class with factory pattern
- 10 different tone mapping algorithms (Reinhard, ACES, etc.)
- Applied in final rendering stage

### WebGPU Support

The framework includes WebGPU variants of most components:
- `WebGPURenderingContext` - WebGPU rendering context
- `WebGPUVolume` - WebGPU volume texture management
- `WebGPU*Renderer` classes - WebGPU compute shader renderers
- `WebGPU*ToneMapper` classes - WebGPU tone mapping

WebGPU is currently the default (see `Application.js:31`).

## Data Flow

1. **Application.js** initializes the rendering context and UI
2. User loads volume data via `VolumeLoadDialog`
3. `Reader` reads metadata and blocks from volume file
4. `Volume` class uploads data to 3D texture
5. `Renderer` selected via `RendererFactory` based on user choice
6. `RenderingContext.render()` loop:
   - Calls `renderer._generateFrame()` - ray marching/volume integration
   - Calls `renderer._integrateFrame()` - accumulates samples for progressive refinement
   - Calls `renderer._renderFrame()` - applies tone mapping and displays
7. `ToneMapper` applies HDR to LDR conversion

## Common Tasks

### Adding a New Renderer

1. Create `src/js/renderers/MyRenderer.js` extending `AbstractRenderer`
2. Implement `_generateFrame()`, `_integrateFrame()`, `_resetFrame()`, `_renderFrame()`
3. Create corresponding `src/glsl/renderers/MyRenderer.glsl`
4. Add to `src/js/renderers/RendererFactory.js`
5. Add to `src/js/dialogs/MainDialog/MainDialog.js` renderer options

### Adding a New Tone Mapper

1. Create `src/js/tonemappers/MyToneMapper.js` extending `AbstractToneMapper`
2. Implement tone mapping in shader
3. Add to `src/js/tonemappers/ToneMapperFactory.js`
4. Add to UI in `MainDialog.js`

### Shader Development

- Edit GLSL/WGSL files in `vpt/src/glsl/` or `vpt/src/wgsl/`
- nodemon automatically rebuilds on file changes (configured in `nodemon.json`)
- Use mixins from `src/glsl/mixins/` or `src/wgsl/mixins/` for common functionality
- Random number generation: `#include "mixins/random/hash/pcg.glsl"`
- Distributions: `#include "mixins/random/distribution/sphere.glsl"`

## Dependencies

- **Node.js** 12.x (for build tools only)
- **Express** (dev server)
- **No runtime dependencies** - pure WebGL 2/WebGPU
- **gl-matrix** (bundled in `src/lib/`)

## License

VPT Framework: GNU General Public License v3 (see vpt/LICENSE)
