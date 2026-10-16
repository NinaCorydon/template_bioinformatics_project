from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

STRAIN_NAMES = {
    "GCA_000011845.1": "CNRZ1066",
    "GCA_000014485.1": "LMD-9",
    "GCA_000698885.1": "ASCC1275",
    "GCA_000056065.1": "L. bulgaricus",
    "GCA_000236475.1": "L. lactis A76",
}


def load_plate_design(csv_path: str | Path) -> pd.DataFrame:
    """Load strain percentages and create a readable composition label per well."""
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Plate design CSV not found: {csv_path}")

    plate = pd.read_csv(csv_path)
    if "well" not in plate.columns:
        raise ValueError("Plate design CSV must contain a 'well' column.")

    strain_columns = [column for column in plate.columns if column.startswith("GCA_")]
    if not strain_columns:
        raise ValueError("Plate design CSV has no strain columns beginning with 'GCA_'.")

    plate["well"] = plate["well"].astype(str).str.strip().str.upper()
    if plate["well"].duplicated().any():
        duplicates = plate.loc[plate["well"].duplicated(), "well"].tolist()
        raise ValueError(f"Plate design contains duplicate well IDs: {duplicates}")

    plate[strain_columns] = plate[strain_columns].apply(pd.to_numeric, errors="raise")
    strain_values = plate[strain_columns]
    if strain_values.isna().any().any():
        raise ValueError("Strain percentage columns cannot contain missing values.")
    if ((strain_values < 0) | (strain_values > 100)).any().any():
        raise ValueError("Strain percentages must be between 0 and 100.")

    totals = strain_values.sum(axis=1)
    valid_totals = np.isclose(totals, 0) | np.isclose(totals, 100)
    if not valid_totals.all():
        invalid_wells = plate.loc[~valid_totals, "well"].tolist()
        raise ValueError(
            "Each well's strain percentages must sum to 0 (blank) or 100; "
            f"invalid wells: {invalid_wells}"
        )

    def composition_for_row(row: pd.Series) -> str:
        components = [
            f"{STRAIN_NAMES.get(column, column)} {row[column]:g}%"
            for column in strain_columns
            if row[column] > 0
        ]
        return "Blank" if not components else " + ".join(components)

    plate["composition"] = plate.apply(composition_for_row, axis=1)
    return plate
