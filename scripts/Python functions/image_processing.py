from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def load_rgb_image(image_path: str | Path) -> np.ndarray:
    """Load an image from disk and return it as an RGB uint8 array."""
    image_path = Path(image_path)
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not load plate image: {image_path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def detect_plate_wells(
    image_rgb: np.ndarray,
    n_rows: int = 8,
    n_columns: int = 12,
) -> pd.DataFrame:
    """Detect well circles and map their centers to standard plate well names."""
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Expected an RGB image with three color channels.")
    if image_rgb.dtype != np.uint8:
        raise ValueError("Expected the RGB image to have uint8 pixel values.")
    if n_rows > 26 or min(n_rows, n_columns) < 1:
        raise ValueError("Plate dimensions must be positive and row count at most 26.")

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    detected = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=45,
        param1=50,
        param2=15,
        minRadius=25,
        maxRadius=30,
    )
    if detected is None:
        raise ValueError(
            "No wells were detected in the plate image. Check the image and "
            "adjust the HoughCircles settings in detect_plate_wells()."
        )

    circles = np.rint(detected[0]).astype(int)
    if circles.shape[0] < max(n_rows, n_columns):
        raise ValueError(
            f"Detected only {circles.shape[0]} well circles; not enough to infer "
            f"a {n_rows}-by-{n_columns} plate grid."
        )

    # Discard duplicate detections around one physical well, retaining the larger circle.
    unique_circles: list[np.ndarray] = []
    duplicate_distance = 45 * 0.55
    for circle in sorted(circles, key=lambda item: item[2], reverse=True):
        x, y, _ = circle
        if all(
            (x - kept[0]) ** 2 + (y - kept[1]) ** 2 >= duplicate_distance**2
            for kept in unique_circles
        ):
            unique_circles.append(circle)
    circles = np.asarray(unique_circles)

    x_values = circles[:, 0].reshape(-1, 1)
    y_values = circles[:, 1].reshape(-1, 1)
    if np.unique(x_values).size < n_columns or np.unique(y_values).size < n_rows:
        raise ValueError(
            "Could not identify enough distinct well columns and rows. "
            "Adjust the circle detection settings."
        )

    x_model = KMeans(n_clusters=n_columns, random_state=0, n_init=10).fit(x_values)
    y_model = KMeans(n_clusters=n_rows, random_state=0, n_init=10).fit(y_values)
    x_order = np.argsort(x_model.cluster_centers_.ravel())
    y_order = np.argsort(y_model.cluster_centers_.ravel())
    column_number = {cluster: index + 1 for index, cluster in enumerate(x_order)}
    row_letter = {cluster: chr(ord("A") + index) for index, cluster in enumerate(y_order)}

    detected_wells = pd.DataFrame(
        {
            "x_cluster": x_model.labels_,
            "y_cluster": y_model.labels_,
            "x": circles[:, 0],
            "y": circles[:, 1],
            "radius": circles[:, 2],
        }
    )
    detected_wells["well"] = [
        f"{row_letter[row]}{column_number[column]}"
        for column, row in zip(
            detected_wells["x_cluster"],
            detected_wells["y_cluster"],
        )
    ]

    # Estimate a complete regular grid, including any circle missed by Hough detection.
    default_radius = float(np.median(circles[:, 2]))
    wells: list[dict[str, float | str]] = []
    for row_cluster in y_order:
        row_letter_value = row_letter[row_cluster]
        row_circles = detected_wells[detected_wells["y_cluster"] == row_cluster]
        for column_cluster in x_order:
            column_number_value = column_number[column_cluster]
            cell = row_circles[row_circles["x_cluster"] == column_cluster]
            radius = float(cell["radius"].median()) if not cell.empty else default_radius
            wells.append(
                {
                    "well": f"{row_letter_value}{column_number_value}",
                    "x": float(x_model.cluster_centers_[column_cluster, 0]),
                    "y": float(y_model.cluster_centers_[row_cluster, 0]),
                    "radius": radius,
                }
            )

    return pd.DataFrame(wells)
