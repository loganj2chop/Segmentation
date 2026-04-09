# Kidney Ultrasound Segmentation Model

A pretrained U-Net model for segmenting kidneys in ultrasound images. The model accepts **256×256 single-channel (grayscale) images** and outputs a binary kidney mask.

---

## Model Architecture

- **Architecture:** U-Net ([segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch))
- **Encoder:** ResNet-34 (pretrained on ImageNet)
- **Input:** 1-channel grayscale, 256×256
- **Output:** Binary kidney mask, 256×256
- **Threshold:** 0.5 (sigmoid output)

---

## Requirements

```bash
pip install torch segmentation-models-pytorch numpy scikit-image tqdm pydicom gcsfs
```

Or with conda:

```bash
conda run --no-capture-output -n MAG3 python your_script.py
```

---

## Preprocessing Notes

Regardless of your source format (DICOM, PNG, JPEG, NumPy array), the model expects:

- Images resized to **256×256**
- **Single channel** (grayscale). RGB/YBR images must be converted using BT.601 luma weights: `0.299·R + 0.587·G + 0.114·B`
- Pixel values normalized to **[0, 1]** (divide by 255 if your array is uint8)
- A **channel dimension** added: shape should be `(N, 1, 256, 256)` for a batch of N images

**Recommended crop** before resizing (applied in the full pipeline examples below):

| Side   | Amount |
|--------|--------|
| Top    | 10%    |
| Bottom | 10%    |
| Left   | 5%     |
| Right  | 5%     |

This crop removes probe labels and UI overlays common in ultrasound frames. It is a suggestion — adjust to your institution's scanner layout as needed.

---

## Example 1: NumPy Array Input (Simplest Case)

Use this as your starting point if you have already converted your images to a NumPy array. The array is assumed to already be 256×256. Crop and normalization behavior can be adjusted to suit your data.

```python
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from tqdm import tqdm

print("Starting underlay generation...")

# =========================================================
# Load Ultrasound Images
# Shape expected: (N, 256, 256) — grayscale, already resized
# =========================================================
image_array = np.load('ultrasound_RPA_images_256.npy')
print("Original image array shape:", image_array.shape)

images = torch.tensor(image_array, dtype=torch.float32)

# Add channel dimension if needed → (N, 1, 256, 256)
if images.ndim == 3:
    images = images.unsqueeze(1)
print("Tensor shape:", images.shape)

# =========================================================
# Dataset
# =========================================================
class KidneyImageDataset(Dataset):
    def __init__(self, images):
        self.images = images
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        return self.images[idx]

dataset = KidneyImageDataset(images)
dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

# =========================================================
# Load Model
# =========================================================
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=1,
    classes=1
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Running on device:", device)

model.load_state_dict(torch.load('model_pretrained.pth', map_location=device))
model = model.to(device)
model.eval()

# =========================================================
# Generate Kidney Masks
# The "underlay" is the original image multiplied by the
# predicted kidney mask — background pixels become 0.
# =========================================================
all_underlays = []
with torch.no_grad():
    for batch in tqdm(dataloader):
        batch = batch.to(device)

        # Normalize to [0, 1] if not already
        if batch.max().item() > 1.0:
            batch = batch / 255.0

        outputs = model(batch)
        preds = torch.sigmoid(outputs)
        preds = (preds > 0.5).float()

        # Multiply original image by predicted kidney mask
        underlay = batch * preds
        all_underlays.append(underlay.cpu())

# =========================================================
# Save
# Output shape: (N, 256, 256)
# =========================================================
underlays_tensor = torch.cat(all_underlays, dim=0)
underlays_np = underlays_tensor.squeeze(1).numpy()
print("Underlay shape:", underlays_np.shape)

np.save("ultrasound_RPA_underlays_256.npy", underlays_np)
print("Saved ultrasound_underlays_256.npy")
```

**Output:** `ultrasound_RPA_underlays_256.npy` — shape `(N, 256, 256)`, kidney region preserved, background zeroed.

---

## Example 2: Full DICOM Pipeline (Manifest-Driven)

This is the full production pipeline. It reads a CSV manifest of DICOM file paths, processes them fold-by-fold using GroupKFold cross-validation, and writes outputs directly to disk as memory-mapped `.npy` files to avoid OOM on large datasets.

Supports both local/NFS paths and Google Cloud Storage (`gs://`) paths.

### Input CSV Format

The manifest CSV must contain these columns:

| Column | Description |
|--------|-------------|
| `study_id` | Patient/study identifier (used for group-based splitting) |
| `dcmfile` | Full path to the DICOM file (local or `gs://`) |
| `Bad` | Quality flag |
| `new-old` | Scan timepoint label |
| `function` | Functional label |

### Running a Single Fold

```bash
# Run fold 1 of 5
python train.py --fold 1

# With custom paths
python train.py --fold 2 \
    --csv /path/to/manifest.csv \
    --output-dir /path/to/output \
    --parts 5
```

### Slurm / Batch Array

To run all 5 folds in parallel via a job array, use the provided shell runner:

```bash
#!/bin/bash
# submit_all_folds.sh
# Submit as: sbatch --array=1-5 submit_all_folds.sh
PYSCRIPT="train.py --fold $SLURM_ARRAY_TASK_ID"
conda run --no-capture-output -n MAG3 python $PYSCRIPT
```

Or run a single fold directly:

```bash
#!/bin/bash
# run_fold.sh  (SPLIT_ID is set by your scheduler or manually)
PYSCRIPT="train.py --fold $SPLIT_ID"
conda run --no-capture-output -n MAG3 python $PYSCRIPT
```

### Output Files

For each fold and split, the following files are written to `--output-dir`:

```
fold{N}_train_image_{part}.npy   # Cropped grayscale images,  shape (M, 256, 256), float32
fold{N}_train_mask_{part}.npy    # Binary kidney masks,        shape (M, 256, 256), float32
fold{N}_train_ids_{part}.csv     # Metadata rows for this part

fold{N}_test_image_{part}.npy
fold{N}_test_mask_{part}.npy
fold{N}_test_ids_{part}.csv
```

### DICOM Handling Notes

- Multi-frame DICOMs: only the **first frame** is used
- RGB/YBR pixel arrays are converted to grayscale automatically
- NFS/stale file handle errors are retried up to 5 times with exponential backoff
- GCS paths (`gs://...`) are read via `gcsfs` using application default credentials

### Preprocessing Applied in the Pipeline

```
1. Read DICOM pixel array
2. Convert to grayscale if RGB/YBR
3. Take first frame if multi-frame
4. Resize to 256×256
5. Crop: 10% top, 10% bottom, 5% left, 5% right → resize back to 256×256
6. Normalize to [0, 1]
7. Run model inference
8. Threshold sigmoid output at 0.5 → binary mask
```

---

## Adapting to Other Input Formats

If your institution starts from a different format, convert to a `(N, 256, 256)` float32 NumPy array first, then use Example 1.

| Source Format | Suggested Conversion |
|---------------|----------------------|
| PNG / JPEG | `PIL.Image.open()` → `np.array()` → grayscale → resize |
| DICOM | `pydicom.dcmread().pixel_array` → grayscale → resize (see Example 2) |
| NIfTI / CT | `SimpleITK.ReadImage()` → `GetArrayFromImage()` → slice selection → resize |
| NumPy array (not 256×256) | `skimage.transform.resize(arr, (N, 256, 256))` |

The only hard requirements are **256×256**, **single channel**, and **float32 normalized to [0, 1]**.

---

## File Summary

| File | Description |
|------|-------------|
| `model_pretrained.pth` | Pretrained model weights |
| `train.py` | Full DICOM pipeline (Example 2) |
| `run_fold.sh` | Shell runner for one fold |

---

## Citation / Contact

If you use this model, please contact the originating institution for attribution details.