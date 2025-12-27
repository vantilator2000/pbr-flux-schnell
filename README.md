# PBR Texture Generator v2

Generate seamless PBR (Physically Based Rendering) texture maps from text descriptions with **AI-powered depth estimation**.

## AI Models Used

| Model | Purpose | License |
|-------|---------|---------|
| [Playground v2.5](https://huggingface.co/playgroundai/playground-v2.5-1024px-aesthetic) | Texture generation | Apache 2.0 |
| [Intel DPT-Large](https://huggingface.co/Intel/dpt-large) | Depth/height estimation | Apache 2.0 |

## Features

- **AI Height Maps**: Uses Intel DPT-Large for accurate depth estimation from textures
- **Derived PBR Maps**: Normal, AO, roughness, and emissive computed from AI-estimated height
- **Hybrid Normals**: Blend height-based (smooth) with color-based (detailed) normals
- **Emissive Detection**: Auto-detect glowing/neon areas for emissive maps
- **Seamless Tiling**: Edge blending with cosine interpolation
- **OpenGL/DirectX**: Choose normal map format for your engine
- **16-bit Height**: Optional high-precision height map export

## Output

Generates **7 texture maps** from a single prompt:

| Output | Description |
|--------|-------------|
| **color.png** | Base color/albedo texture |
| **height.png** | AI-estimated height/displacement map |
| **normal.png** | Normal map (OpenGL or DirectX format) |
| **roughness.png** | Surface roughness (black=smooth, white=rough) |
| **ao.png** | Ambient occlusion for soft shadows |
| **emissive.png** | Glow/emission map (bright saturated areas) |
| **grid.png** | 3x2 preview of all maps |

## Usage

```python
import replicate

output = replicate.run(
    "vantilator2000/pbr-playground",
    input={
        "prompt": "seamless cyberpunk circuit board, neon blue and purple, glowing lines",
        "negative_prompt": "blurry, text, watermark",
        "resolution": 1024,
        "tiling_strength": 0.5,
        "num_steps": 25,
        "normal_strength": 1.0,
        "normal_detail": 0.25,
        "normal_format": "opengl",
        "height_contrast": 1.0,
        "ao_strength": 1.0,
        "seed": 42
    }
)

# output[0] = color
# output[1] = height
# output[2] = normal
# output[3] = roughness
# output[4] = ao
# output[5] = emissive
# output[6] = grid
```

## Parameters

### Generation

| Parameter | Default | Description |
|-----------|---------|-------------|
| `input_image` | None | Upload your own texture to generate PBR maps |
| `prompt` | - | Text description of the texture |
| `negative_prompt` | "" | Things to avoid in the texture |
| `resolution` | 1024 | Output size (512 or 1024) |
| `num_steps` | 25 | Inference steps (1-50, higher=better quality) |
| `seed` | -1 | Random seed (-1 for random) |

### Tiling

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tiling_strength` | 0.5 | Seamless tiling blend (0-1, higher=smoother edges) |

### Height Map

| Parameter | Default | Description |
|-----------|---------|-------------|
| `height_contrast` | 1.0 | Height map contrast (0.5-3.0) |
| `height_gamma` | 1.0 | Height gamma (>1 darkens, <1 lightens) |
| `suppress_scene_depth` | true | Remove large-scale depth variations |
| `output_16bit_height` | false | Export height as 16-bit PNG (higher precision) |

### Normal Map

| Parameter | Default | Description |
|-----------|---------|-------------|
| `normal_strength` | 1.0 | Normal intensity (0.1-5.0) |
| `normal_detail` | 0.25 | Blend color-based detail (0=height only, 1=max detail) |
| `normal_format` | opengl | "opengl" (Y+ up) or "directx" (Y- down) |

### Ambient Occlusion

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ao_strength` | 1.0 | AO intensity (0-2.0) |
| `ao_radius` | 8.0 | Shadow spread radius (1-32) |

### Roughness

| Parameter | Default | Description |
|-----------|---------|-------------|
| `roughness_contrast` | 1.0 | Roughness contrast (0.5-2.0) |
| `roughness_base` | 0.5 | Base roughness level (0-1) |

## Two Modes

### 1. Generate from Prompt
Leave `input_image` empty and provide a text prompt. The AI generates the texture and all PBR maps.

### 2. Generate from Image
Upload your own texture image - the model estimates height using AI and generates all PBR maps from it.

## Normal Map Formats

| Format | Y Direction | Use With |
|--------|-------------|----------|
| **OpenGL** | Y+ up | Blender, Unreal Engine, Godot |
| **DirectX** | Y- down | Unity, 3ds Max |

## Example Prompts

- `seamless dark wood texture, oak, detailed grain`
- `seamless marble texture, white with grey veins`
- `seamless grass texture, top-down view, lawn`
- `seamless concrete texture, weathered, cracks`
- `seamless metal texture, brushed steel`
- `seamless cyberpunk circuit board, neon blue and purple, glowing lines`
- `seamless stone wall texture, medieval castle`
- `seamless leather texture, brown, worn`

## Tips

1. **Include "seamless"** in your prompt for better tiling
2. **Use "top-down view"** for floor/ground textures
3. **Increase `num_steps`** (30-40) for higher quality
4. **Increase `normal_strength`** (1.5-2.0) for more pronounced bumps
5. **Adjust `normal_detail`** (0.3-0.5) for more surface texture in normals
6. **Set `ao_radius`** higher for softer, more spread shadows
7. **Use negative_prompt** to avoid unwanted elements like "blurry, text, watermark"

## Technical Details

### Depth Estimation
Uses Intel DPT-Large (Dense Prediction Transformer) for monocular depth estimation. The depth is converted to a height map with optional scene-depth suppression to preserve texture detail.

### Normal Generation
Normals are computed using Sobel operators on the height map. Optionally blends in high-frequency detail from the color image for more organic results.

### Emissive Detection
Analyzes HSV color space to detect bright, saturated areas (neon colors glow more than white).

### Seamless Tiling
Uses edge blending with cosine interpolation - blends opposite borders together for seamless wrapping without center artifacts.

## Use Cases

- Game development (Unity, Unreal Engine, Godot)
- 3D rendering (Blender, Maya, 3ds Max)
- Architectural visualization
- Product design
- Digital art
- VFX and film production

## License

This project uses models with Apache 2.0 licenses, suitable for commercial use.
