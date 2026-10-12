from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def calculate_well_spectra(
    cube: np.ndarray,
    wavelengths: np.ndarray,
    wells: pd.DataFrame,
) -> pd.DataFrame:
    """Average all finite hyperspectral pixel spectra inside each well circle."""
    if cube.ndim != 3:
        raise ValueError("Expected cube dimensions in (rows, columns, bands) order.")
    if cube.shape[2] != wavelengths.size:
        raise ValueError("The wavelength count must match the cube's band count.")
    if not {"well", "x", "y", "radius"}.issubset(wells.columns):
        raise ValueError("Well table must include 'well', 'x', 'y', and 'radius'.")

    height, width, _ = cube.shape
    row_grid, column_grid = np.ogrid[:height, :width]
    records: list[dict[str, float | int | str]] = []

    for well in wells.itertuples(index=False):
        x = float(well.x)
        y = float(well.y)
        radius = float(well.radius)
        if radius <= 0:
            raise ValueError(f"Well {well.well} has a non-positive radius.")

        mask = (column_grid - x) ** 2 + (row_grid - y) ** 2 <= radius**2
        spectra = np.asarray(cube[mask])
        valid = np.isfinite(spectra).all(axis=1)
        spectra = spectra[valid]
        if spectra.shape[0] == 0:
            raise ValueError(f"No finite cube pixels found inside well {well.well}.")

        record: dict[str, float | int | str] = {
            "well": str(well.well),
            "pixel_count": int(spectra.shape[0]),
        }
        record.update(
            {float(wavelength): float(value) for wavelength, value in zip(wavelengths, spectra.mean(axis=0))}
        )
        records.append(record)

    return pd.DataFrame.from_records(records)


def project_spectra_pca(
    analysis: pd.DataFrame,
    wavelengths: np.ndarray,
) -> tuple[pd.DataFrame, PCA]:
    """Fit two-component PCA to mean spectra and return labeled well scores."""
    spectral_columns = [float(wavelength) for wavelength in wavelengths]
    missing = [column for column in spectral_columns if column not in analysis.columns]
    if missing:
        raise ValueError(f"Analysis table is missing wavelength columns: {missing}")

    spectra = analysis.loc[:, spectral_columns].to_numpy(dtype=float)
    valid = np.isfinite(spectra).all(axis=1)
    if valid.sum() < 2:
        raise ValueError("PCA requires at least two wells with complete spectra.")
    if spectra.shape[1] < 2:
        raise ValueError("PCA requires at least two wavelength bands.")

    pca = PCA(n_components=2)
    projected = pca.fit_transform(spectra[valid])
    scores = pd.DataFrame(
        {
            "well": analysis.loc[valid, "well"].to_numpy(),
            "PC1": projected[:, 0],
            "PC2": projected[:, 1],
        }
    )
    return scores, pca
