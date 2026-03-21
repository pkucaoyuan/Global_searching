# Vectorize Color Images

Auto-vectorize color PNG/JPG images using Inkscape with iterative optimization to preserve all details and colors.

## Description

This skill automatically converts bitmap images to true vector PDFs while:
- Preserving all colors and gradients
- Maintaining fine details and text clarity
- Iteratively trying different parameters until quality requirements are met
- Generating watermark-free vector output

## Usage

```bash
/vectorize-color <image-path> [--target-colors N] [--max-attempts N]
```

## Parameters

- `image-path`: Path to PNG or JPG file
- `--target-colors`: Minimum color count to preserve (default: auto-detect)
- `--max-attempts`: Maximum optimization iterations (default: 5)

## Examples

```bash
# Single image
/vectorize-color Setting.png

# Batch process
/vectorize-color Setting.png comparison.png two-stage.png training_pipeline.jpg

# Custom parameters
/vectorize-color Setting.png --target-colors 16 --max-attempts 10
```

## Output

- `<name>_vector.svg`: Optimized vector file
- `<name>_vector.pdf`: Print-ready PDF
- `<name>_quality_report.txt`: Quality metrics

## Quality Metrics

The skill evaluates:
1. ✅ True vector (no embedded bitmaps)
2. ✅ Color preservation (vs original)
3. ✅ File size optimization
4. ✅ Path smoothness
5. ✅ Detail retention

## Implementation

See: `scripts/vectorization/auto_vectorize.py`
