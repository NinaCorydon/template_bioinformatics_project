# Hyperspectral plate analysis

Run the plate analysis from the repository root:

```powershell
python -m pip install -r requirements.txt
python scripts/main.py
```

The script loads:

- `data/cleaned/cropped_plate_image.jpg`
- `data/raw/plate information/plate_design.csv`
- `data/raw/spectral data/plate_cube.hdr`
- `data/raw/spectral data/plate_cube.raw`

Reusable functions are in `scripts/Python functions/`. The analysis writes the
mean spectrum table, PCA scores, detected-well overlay, and plots to
`results/hyperspectral_analysis/`.

The ENVI loader expects the generated cube format: float32 data, BIP
interleave, with the associated wavelengths in the header. Well detection
uses Hough circles followed by row/column clustering; if detection fails,
adjust the circle settings in `detect_plate_wells()` in
`scripts/Python functions/image_processing.py`.

The generated `plate_cube.raw` is about 458 MB and is ignored by Git because
it exceeds GitHub's regular 100 MB file limit. Keep a local copy at the path
above, or arrange to provide it through Git LFS or another data store before
running the analysis on a fresh clone.