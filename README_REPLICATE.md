# PBR Texture Generator

Generate seamless PBR (Physically Based Rendering) texture maps from text descriptions.

## Output

This model generates 4 texture maps from a single prompt:

| Map | Description |
|-----|-------------|
| **Diffuse** | Base color/albedo texture |
| **Normal** | Surface detail for lighting |
| **Roughness** | Surface smoothness (black=smooth, white=rough) |
| **AO** | Ambient occlusion for soft shadows |

## Usage

```python
import replicate

output = replicate.run(
    "vantilator2000/pbr-texture-gen",
    input={
        "prompt": "seamless red brick wall texture, weathered, detailed",
        "resolution": 1024,
        "tiling_strength": 0.5,
        "num_steps": 8,
        "seed": 42
    }
)

# output.diffuse, output.normal, output.roughness, output.ao
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `prompt` | - | Text description of the texture |
| `resolution` | 512 | Output size (512 or 1024) |
| `tiling_strength` | 0.5 | Seamless tiling blend (0-1) |
| `num_steps` | 8 | Inference steps (1-30, higher=better quality) |
| `seed` | -1 | Random seed (-1 for random) |

## Example Prompts

- `seamless dark wood texture, oak, detailed grain`
- `seamless marble texture, white with grey veins`
- `seamless grass texture, top-down view, lawn`
- `seamless concrete texture, weathered, cracks`
- `seamless metal texture, brushed steel`
- `seamless fabric texture, blue denim, detailed`

## Tips

1. **Include "seamless"** in your prompt for better tiling
2. **Use "top-down view"** for floor/ground textures
3. **Increase `num_steps`** (12-20) for higher quality
4. **Set `tiling_strength`** to 0.7+ for perfect seamless edges

## Use Cases

- Game development (Unity, Unreal Engine)
- 3D rendering (Blender, Maya, 3ds Max)
- Architectural visualization
- Product design
- Digital art

## Model

Powered by [Segmind SSD-1B](https://huggingface.co/segmind/SSD-1B) - a distilled version of SDXL optimized for speed and quality.
