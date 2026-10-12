from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_well_overlay(
    image_rgb: np.ndarray,
    wells: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Save an image showing the inferred well centers and labels."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image_rgb)
    for well in wells.itertuples(index=False):
        circle = plt.Circle(
            (well.x, well.y),
            well.radius,
            fill=False,
            color="yellow",
            linewidth=0.8,
        )
        ax.add_patch(circle)
        ax.text(
            well.x,
            well.y,
            well.well,
            color="white",
            fontsize=5,
            ha="center",
            va="center",
            bbox={"facecolor": "black", "alpha": 0.45, "pad": 1},
        )
    ax.set_title("Detected plate wells")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_mean_spectra(
    analysis: pd.DataFrame,
    wavelengths: np.ndarray,
    output_path: str | Path,
) -> None:
    """Plot mean hyperspectral signatures grouped by plate composition."""
    spectra = analysis.loc[:, wavelengths]
    grouped = spectra.assign(composition=analysis["composition"]).groupby(
        "composition",
        sort=False,
    )

    fig, ax = plt.subplots(figsize=(12, 7))
    for composition, group in grouped:
        ax.plot(wavelengths, group.mean(axis=0).to_numpy(), label=composition)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Mean synthetic signal")
    ax.set_title("Mean hyperspectral signature by plate composition")
    ax.grid(True, alpha=0.25)
    ax.legend(title="Composition", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_pca_scores(
    analysis: pd.DataFrame,
    pca: object,
    output_path: str | Path,
) -> None:
    """Plot well PCA scores with color and labels indicating plate composition."""
    compositions = analysis["composition"].drop_duplicates().tolist()
    color_map = plt.get_cmap("tab20", max(len(compositions), 1))
    fig, ax = plt.subplots(figsize=(12, 8))

    for index, composition in enumerate(compositions):
        selected = analysis["composition"] == composition
        ax.scatter(
            analysis.loc[selected, "PC1"],
            analysis.loc[selected, "PC2"],
            label=composition,
            color=color_map(index),
            s=55,
            edgecolors="black",
            linewidths=0.35,
            alpha=0.85,
        )

    for row in analysis.itertuples(index=False):
        ax.annotate(
            row.well,
            (row.PC1, row.PC2),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=5,
        )

    variance = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC 1 ({variance[0]:.1%} variance)")
    ax.set_ylabel(f"PC 2 ({variance[1]:.1%} variance)")
    ax.set_title("PCA of mean well spectra, colored by plate composition")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Composition", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
