from __future__ import annotations

import sys
from pathlib import Path

FUNCTIONS_DIR = Path(__file__).resolve().parent / "Python functions"
sys.path.insert(0, str(FUNCTIONS_DIR))

from envi_io import load_envi_cube
from image_processing import detect_plate_wells, load_rgb_image
from plate_design import load_plate_design
from plotting import plot_mean_spectra, plot_pca_scores, save_well_overlay
from spectral_analysis import calculate_well_spectra, project_spectra_pca


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    image_path = project_root / "data" / "cleaned" / "cropped_plate_image.jpg"
    design_path = (
        project_root / "data" / "raw" / "plate information" / "plate_design.csv"
    )
    header_path = project_root / "data" / "raw" / "spectral data" / "plate_cube.hdr"
    raw_path = project_root / "data" / "raw" / "spectral data" / "plate_cube.raw"
    output_dir = project_root / "results" / "hyperspectral_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    image = load_rgb_image(image_path)
    plate_design = load_plate_design(design_path)
    wells = detect_plate_wells(image, n_rows=8, n_columns=12)
    cube, wavelengths = load_envi_cube(header_path, raw_path)

    if image.shape[:2] != cube.shape[:2]:
        raise ValueError(
            "The cropped plate image and hyperspectral cube must have matching "
            f"row/column dimensions; got {image.shape[:2]} and {cube.shape[:2]}."
        )

    well_spectra = calculate_well_spectra(cube, wavelengths, wells)
    analysis = well_spectra.merge(
        plate_design,
        on="well",
        how="left",
        validate="one_to_one",
    )
    if analysis["composition"].isna().any():
        missing = analysis.loc[analysis["composition"].isna(), "well"].tolist()
        raise ValueError(f"No plate-design composition found for wells: {missing}")

    scores, pca = project_spectra_pca(analysis, wavelengths)
    analysis = analysis.merge(scores, on="well", validate="one_to_one")

    analysis.to_csv(output_dir / "well_mean_spectra.csv", index=False)
    scores.merge(
        analysis[["well", "composition"]],
        on="well",
        validate="one_to_one",
    ).to_csv(output_dir / "pca_scores.csv", index=False)

    save_well_overlay(image, wells, output_dir / "detected_wells.png")
    plot_mean_spectra(
        analysis,
        wavelengths,
        output_dir / "mean_spectra_by_composition.png",
    )
    plot_pca_scores(
        analysis,
        pca,
        output_dir / "pca_by_composition.png",
    )

    print(f"Loaded cube with shape {cube.shape} and {wavelengths.size} wavelengths.")
    print(f"Analyzed {len(analysis)} wells.")
    print(f"PC explained variance: {pca.explained_variance_ratio_}")
    print(f"Saved tables and plots to {output_dir}")


if __name__ == "__main__":
    main()
