# Enhance Image Quality

Improve image resolution and quality for academic paper figures using various upscaling and optimization techniques.

## Description

This skill enhances image quality by:
- Upscaling low-resolution images to higher DPI (300+ for print)
- Sharpening blurry images while preserving details
- Optimizing file size without quality loss
- Converting between formats (PNG, JPG, PDF) with quality control

## Usage

```bash
/enhance-image-quality <image-path> [options]
```

## Parameters

- `image-path`: Path to image file(s) - supports PNG, JPG, JPEG, PDF
- `--dpi N`: Target DPI (default: 300 for print quality)
- `--scale N`: Scale factor (e.g., 2 for 2x resolution)
- `--sharpen`: Apply sharpening filter
- `--denoise`: Apply noise reduction
- `--output-format`: Output format (png, jpg, pdf)
- `--quality N`: JPEG quality 1-100 (default: 95)

## Examples

```bash
# Single image - double resolution
/enhance-image-quality Setting.png --scale 2

# Set specific DPI for print
/enhance-image-quality figure1.png --dpi 300

# Batch process all figures
/enhance-image-quality paper/figures/*.png --dpi 300 --sharpen

# Convert to high-quality PDF
/enhance-image-quality comparison.png --output-format pdf --dpi 300

# Full optimization pipeline
/enhance-image-quality training_pipeline.jpg --scale 2 --sharpen --denoise --dpi 300
```

## Workflow

### Step 1: Analyze Source Image

Read the input image and report:
- Current resolution (width x height)
- Current DPI (if available)
- File size
- Color depth
- Format

### Step 2: Determine Enhancement Strategy

Based on parameters and source analysis:

| Source State | Strategy |
|--------------|----------|
| Low DPI (<150) | Upscale with Lanczos/bicubic interpolation |
| Blurry | Apply unsharp mask |
| Noisy | Apply bilateral filter (edge-preserving) |
| JPEG artifacts | Denoise before upscaling |

### Step 3: Apply Enhancements

Use ImageMagick or Python PIL for processing:

**ImageMagick (preferred for quality)**:
```bash
# Upscale with Lanczos
convert input.png -filter Lanczos -resize 200% output.png

# Set DPI
convert input.png -density 300 -units PixelsPerInch output.png

# Sharpen
convert input.png -sharpen 0x1.0 output.png

# Full pipeline
convert input.png \
    -filter Lanczos -resize 200% \
    -sharpen 0x0.5 \
    -density 300 -units PixelsPerInch \
    output.png
```

**Python PIL alternative**:
```python
from PIL import Image, ImageFilter, ImageEnhance

img = Image.open("input.png")

# Upscale with high-quality resampling
new_size = (img.width * 2, img.height * 2)
img = img.resize(new_size, Image.LANCZOS)

# Sharpen
img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))

# Set DPI and save
img.save("output.png", dpi=(300, 300))
```

### Step 4: Quality Verification

After enhancement, verify:
- [ ] Resolution meets target
- [ ] DPI is set correctly (use `identify -verbose` or `exiftool`)
- [ ] No visible artifacts introduced
- [ ] File size is reasonable
- [ ] Colors preserved accurately

### Step 5: Generate Report

Output quality comparison:
```
Image Quality Report: figure1.png
================================
           Before    After     Change
Resolution: 800x600   1600x1200  +100%
DPI:        72        300        +317%
File Size:  245 KB    890 KB     +263%
Sharpness:  Applied
Denoise:    N/A
```

## Output

- `<name>_enhanced.<ext>`: Enhanced image file
- `<name>_quality_report.txt`: Before/after metrics (optional)

## Tool Requirements

Ensure one of these is available:
- **ImageMagick**: `brew install imagemagick` (macOS) / `apt install imagemagick` (Linux)
- **Python PIL**: `pip install Pillow`

Verify installation:
```bash
convert --version  # ImageMagick
python -c "from PIL import Image; print('PIL OK')"
```

## Quality Guidelines for Academic Papers

| Use Case | Recommended DPI | Format |
|----------|-----------------|--------|
| Screen/Web | 72-150 | PNG |
| Print (figures) | 300 | PNG/PDF |
| Print (photos) | 300-600 | JPG (q95) |
| Vector diagrams | N/A | PDF/SVG |

## Notes

- For true vector quality, use `/vectorize-color` skill instead
- Upscaling cannot recover lost detail - start with highest available resolution
- JPEG compression introduces artifacts - prefer PNG for intermediate steps
- PDF output embeds image at specified DPI for consistent print quality
