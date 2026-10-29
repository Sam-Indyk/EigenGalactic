"""
Export the eigengalaxy basis for the interactive web demo on samuelindyk.com.

Reuses the data loader and configuration from Eigengalactic.py, fits the same
IncrementalPCA model, then writes:

  web_export/eigengalaxies.json  metadata (sizes, explained variance, offsets)
  web_export/eigengalaxies.bin   mean (float32), components (int8, per-component
                                 scale), and held-out test galaxies (uint8)
  web_export/*.png               the same diagnostic plots the main script makes

Usage:  python export_web.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm

from Eigengalactic import (
    BATCH_SIZE,
    IMAGE_SIZE,
    NUM_COMPONENTS,
    NUM_TRAINING_IMAGES,
    load_galaxy_zoo_data,
)

OUT_DIR = "web_export"
NUM_TEST_IMAGES = 48   # held out from training, shipped to the browser to reconstruct
SEED = 0


def main():
    data = load_galaxy_zoo_data(IMAGE_SIZE, NUM_TRAINING_IMAGES)
    if data is None or len(data) == 0:
        raise SystemExit("No images were loaded.")

    # Drop images that failed to load (the loader zero-fills them).
    data = data[data.max(axis=1) > 0]

    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(data))
    test = data[order[:NUM_TEST_IMAGES]]
    train = data[order[NUM_TEST_IMAGES:]]

    pca = IncrementalPCA(n_components=NUM_COMPONENTS, batch_size=BATCH_SIZE)
    for i in tqdm(range(0, len(train), BATCH_SIZE), desc="Training PCA model"):
        batch = train[i:i + BATCH_SIZE]
        if len(batch) >= NUM_COMPONENTS:  # partial_fit needs at least n_components rows
            pca.partial_fit(batch)

    os.makedirs(OUT_DIR, exist_ok=True)

    # Quantise each component to int8 with its own scale (~4x smaller, visually lossless).
    comps = pca.components_.astype(np.float32)
    scales = np.abs(comps).max(axis=1) / 127.0
    q = np.round(comps / scales[:, None]).astype(np.int8)

    mean = pca.mean_.astype(np.float32)
    test_u8 = np.clip(np.round(test), 0, 255).astype(np.uint8)

    with open(os.path.join(OUT_DIR, "eigengalaxies.bin"), "wb") as f:
        f.write(mean.tobytes())
        f.write(q.tobytes())
        f.write(test_u8.tobytes())

    d = IMAGE_SIZE[0] * IMAGE_SIZE[1]
    meta = {
        "source": "Galaxy Zoo 2 (galaxy-datasets), grayscale, LANCZOS-resized",
        "image_size": list(IMAGE_SIZE),
        "n_train": int(len(train)),
        "n_components": int(NUM_COMPONENTS),
        "n_test": int(len(test_u8)),
        "explained_variance_ratio": [round(float(v), 6) for v in pca.explained_variance_ratio_],
        "component_scales": [float(s) for s in scales],
        "layout": {
            "mean": {"offset": 0, "dtype": "float32", "count": d},
            "components": {"offset": 4 * d, "dtype": "int8", "count": NUM_COMPONENTS * d},
            "test": {"offset": 4 * d + NUM_COMPONENTS * d, "dtype": "uint8", "count": len(test_u8) * d},
        },
    }
    with open(os.path.join(OUT_DIR, "eigengalaxies.json"), "w") as f:
        json.dump(meta, f, indent=1)

    # Diagnostics, matching the main script's figures.
    plt.figure(figsize=(4, 4))
    plt.imshow(mean.reshape(IMAGE_SIZE), cmap="gray")
    plt.axis("off")
    plt.savefig(os.path.join(OUT_DIR, "mean_galaxy.png"), bbox_inches="tight", dpi=120)
    plt.close()

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(comps[i].reshape(IMAGE_SIZE), cmap="gray")
        ax.set_title(f"Component {i + 1}")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "eigengalaxies.png"), dpi=120)
    plt.close()

    total = float(np.sum(pca.explained_variance_ratio_))
    print(f"Trained on {len(train)} images; first component explains "
          f"{pca.explained_variance_ratio_[0] * 100:.2f}%, all {NUM_COMPONENTS} explain {total * 100:.2f}%")
    print(f"Wrote {OUT_DIR}/eigengalaxies.bin ({os.path.getsize(os.path.join(OUT_DIR, 'eigengalaxies.bin')) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
