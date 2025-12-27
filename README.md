# PBR Texture Generator v2

Generate seamless PBR (Physically Based Rendering) texture maps from text descriptions with **AI-powered depth estimation**.

Powered by **Playground v2.5** for texture generation and **Intel DPT-Large** for depth/height estimation.

## What's New in v2

- **AI Height Maps**: Uses Intel DPT-Large for accurate depth estimation
- **Derived PBR Maps**: Normal, AO, and roughness are computed from the AI-estimated height
- **Better Quality**: Height-based normal maps capture true surface detail
- **More Controls**: Adjust height contrast, normal strength, AO radius, and more
- **OpenGL/DirectX**: Choose normal map format for your engine

## Output

This model generates 6 texture maps from a single prompt:

| Output | Description |
|--------|-------------|
| **color.png** | Base color/albedo texture |
| **height.png** | AI-estimated height/displacement map |
| **normal.png** | Normal map derived from height (OpenGL or DirectX) |
| **roughness.png** | Surface roughness (black=smooth, white=rough) |
| **ao.png** | Ambient occlusion for soft shadows |
| **grid.png** | 3x2 preview of all maps |

## Usage

```python
import replicate

output = replicate.run(
    "vantilator2000/pbr-playground",
    input={
        "prompt": "seamless red brick wall texture, weathered, detailed",
        "negative_prompt": "blurry, text, watermark",
        "resolution": 1024,
        "tiling_strength": 0.5,
        "num_steps": 25,
        "normal_strength": 1.0,
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
# output[5] = grid
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
| `tiling_strength` | 0.5 | Seamless tiling blend (0-1) |

### Height Map

| Parameter | Default | Description |
|-----------|---------|-------------|
| `height_contrast` | 1.0 | Height map contrast (0.5-3.0) |
| `height_gamma` | 1.0 | Height gamma (>1 darkens, <1 lightens) |
| `suppress_scene_depth` | true | Remove large-scale depth variations |
| `output_16bit_height` | false | Export height as 16-bit PNG |

### Normal Map

| Parameter | Default | Description |
|-----------|---------|-------------|
| `normal_strength` | 1.0 | Normal intensity (0.1-5.0) |
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
- `seamless fabric texture, blue denim, detailed`
- `seamless stone wall texture, medieval castle`
- `seamless leather texture, brown, worn`

## Tips

1. **Include "seamless"** in your prompt for better tiling
2. **Use "top-down view"** for floor/ground textures
3. **Increase `num_steps`** (30-40) for higher quality
4. **Increase `normal_strength`** (1.5-2.0) for more pronounced bumps
5. **Adjust `height_contrast`** to control depth intensity
6. **Set `ao_radius`** higher for softer, more spread shadows
7. **Use negative_prompt** to avoid unwanted elements

## Use Cases

- Game development (Unity, Unreal Engine, Godot)
- 3D rendering (Blender, Maya, 3ds Max)
- Architectural visualization
- Product design
- Digital art

## Models

- **Texture Generation**: [Playground v2.5](https://huggingface.co/playgroundai/playground-v2.5-1024px-aesthetic)
- **Depth Estimation**: [Intel DPT-Large](https://huggingface.co/Intel/dpt-large)
