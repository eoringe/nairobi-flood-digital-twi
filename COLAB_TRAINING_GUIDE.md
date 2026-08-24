# Training on Google Colab

Guide to training the flood segmentation model on Google Colab's free GPU instead of your local laptop.

## Quick Summary

- **GPU:** Free T4 (NVIDIA, 16GB VRAM) or P100 (if you get lucky)
- **Training time:** ~4 hours on T4
- **Cost:** Free
- **Data:** Upload once to Google Drive, reuse for all training runs

---

## Step 1: Prepare Your Data (One-Time Setup)

### Option A: Upload Dataset to Google Drive (Recommended)

1. **Create a Google Drive folder:**
   - Go to [drive.google.com](https://drive.google.com)
   - Create folder: `My Drive > nairobi-flood-data`

2. **Upload the training dataset:**
   ```bash
   # On your laptop, install rclone or use Drive web UI
   # File: data/processed/arrays/segmentation_train_dataset.npz (6.1 GB)
   # Drag & drop into: My Drive/nairobi-flood-data/
   ```

3. **Share the folder (get shareable link):**
   - Right-click folder → Share
   - Change to "Anyone with the link"
   - Copy link (you'll need it in Colab)

### Option B: Use Direct Download (Faster, No Storage Limit)

If your codebase is on GitHub:
```bash
# In Colab, just clone directly:
git clone https://github.com/YOUR_USERNAME/nairobi-flood-digital-twi.git
```

---

## Step 2: Create Colab Notebook

### Quick Start (Copy & Paste)

Go to [colab.research.google.com](https://colab.research.google.com) and create a new notebook. **Copy this entire code below into a single cell:**

```python
# ============================================================
# Nairobi Flood Segmentation Training on Google Colab
# ============================================================

# 1. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Clone repository (or use local if you uploaded)
import os
os.chdir('/content')
!git clone https://github.com/YOUR_USERNAME/nairobi-flood-digital-twi.git
os.chdir('nairobi-flood-digital-twi')

# 3. Install dependencies
!pip install -q torch torchvision torchaudio numpy scipy scikit-learn tqdm

# 4. Link dataset (assuming it's in Drive)
!mkdir -p data/processed/arrays
!cp /content/drive/MyDrive/nairobi-flood-data/segmentation_train_dataset.npz data/processed/arrays/

# Verify dataset is there
import os
dataset_path = 'data/processed/arrays/segmentation_train_dataset.npz'
print(f"Dataset exists: {os.path.exists(dataset_path)}")
print(f"Dataset size: {os.path.getsize(dataset_path) / 1e9:.2f} GB")

# 5. Check GPU
!nvidia-smi

# 6. Train the model
import subprocess
result = subprocess.run(['python', '-m', 'src.models.train_segmentation'], 
                       capture_output=False)
print(f"Training exit code: {result.returncode}")

# 7. Download trained model
!cp models/time_series/segmentation_model.pth /content/drive/MyDrive/nairobi-flood-data/
!cp models/time_series/segmentation_metrics.json /content/drive/MyDrive/nairobi-flood-data/
print("✓ Model and metrics saved to Google Drive")
```

---

## Step 3: Run Training

1. **In Colab:** Click the play button (▶) on the cell
2. **Authenticate:** When prompted, allow Colab to access Google Drive
3. **Wait:** Training runs (~4 hours on T4)
   - Monitor in real-time in the cell output
   - Shows epoch progress, validation metrics, best F1 score
4. **Download:** Model automatically saved to your Drive

---

## Step 4: Download Trained Model to Laptop

Once training completes:

```bash
# Option 1: Download from Google Drive web UI
# Go to My Drive > nairobi-flood-data > segmentation_model.pth
# Right-click > Download

# Option 2: Use rclone (faster for large files)
rclone copy gdrive:nairobi-flood-data/segmentation_model.pth ./models/time_series/
```

---

## Advanced: Full Colab Notebook Template

If you want a more structured notebook (better for iterating), create a Colab file with these cells:

### Cell 1: Setup
```python
# Setup GPU and environment
from google.colab import drive
import os
import subprocess

drive.mount('/content/drive')
os.chdir('/content')

# Check GPU
!nvidia-smi
```

### Cell 2: Clone Codebase
```python
# Clone repo (or mount if already uploaded)
if not os.path.exists('nairobi-flood-digital-twi'):
    !git clone https://github.com/YOUR_USERNAME/nairobi-flood-digital-twi.git
    
os.chdir('nairobi-flood-digital-twi')
print("Repository cloned successfully")
```

### Cell 3: Install Dependencies
```python
# Install required packages
!pip install -q torch torchvision torchaudio
!pip install -q numpy scipy scikit-learn tqdm matplotlib
!pip install -q python-dotenv

print("✓ Dependencies installed")
```

### Cell 4: Prepare Dataset
```python
# Create directories
import os
os.makedirs('data/processed/arrays', exist_ok=True)

# Copy dataset from Drive
drive_path = '/content/drive/MyDrive/nairobi-flood-data/segmentation_train_dataset.npz'
local_path = 'data/processed/arrays/segmentation_train_dataset.npz'

if os.path.exists(drive_path):
    !cp "{drive_path}" "{local_path}"
    print(f"✓ Dataset copied ({os.path.getsize(local_path)/1e9:.1f} GB)")
else:
    print("⚠ Dataset not found in Drive. Upload segmentation_train_dataset.npz first.")
```

### Cell 5: Train Model
```python
# Run training
import subprocess
result = subprocess.run(
    ['python', '-m', 'src.models.train_segmentation'],
    cwd='/content/nairobi-flood-digital-twi',
    capture_output=False
)
print(f"\nTraining completed with exit code: {result.returncode}")
```

### Cell 6: Save Results
```python
# Copy trained model back to Drive
import shutil
import os

drive_out = '/content/drive/MyDrive/nairobi-flood-data/outputs'
os.makedirs(drive_out, exist_ok=True)

# Copy model
shutil.copy(
    'models/time_series/segmentation_model.pth',
    f'{drive_out}/segmentation_model.pth'
)

# Copy metrics
shutil.copy(
    'models/time_series/segmentation_metrics.json',
    f'{drive_out}/segmentation_metrics.json'
)

print(f"✓ Model saved to {drive_out}")
```

---

## Troubleshooting

### "Dataset not found"
- Ensure `segmentation_train_dataset.npz` is uploaded to Google Drive
- Check path: `My Drive > nairobi-flood-data > segmentation_train_dataset.npz`

### "CUDA out of memory"
- Reduce batch size in `src/models/train_segmentation.py` line 55
- Change: `BATCH_SIZE = 8` → `BATCH_SIZE = 4`
- Or use CPU (slower but works)

### "Disconnected after 12 hours"
- Colab times out after ~12 hours of inactivity
- Set up auto-save (Colab settings → check "Automatically restart runtime")
- Or train in multiple shorter sessions (model checkpoints every epoch)

### "Git repo not found"
- If your repo is private, use GitHub token:
  ```python
  !git clone https://YOUR_TOKEN@github.com/YOUR_USERNAME/nairobi-flood-digital-twi.git
  ```

### "Module not found" errors
- Ensure your codebase has `src/` directory structure
- Add to path in Colab: `import sys; sys.path.insert(0, '/content/nairobi-flood-digital-twi')`

---

## Performance Tips

1. **Use GPU Runtime:** Colab > Settings > Runtime type > GPU (T4 recommended)
2. **Close other notebooks:** Frees GPU memory
3. **Restart runtime if memory leaks:** Runtime > Restart runtime
4. **Monitor training:** Look for validation F1 plateau (usually epoch 30–40)
5. **Save checkpoint frequently:** Model auto-saves best validation F1

---

## Expected Output

Once training completes, you'll see:

```
[TRAIN] Starting training for 50 epochs...
[E01] train_loss=0.4521 val_loss=0.3892 val_iou=0.0234 val_f1=0.0456
[CKPT] Saved best model (F1=0.0456) to models/time_series/segmentation_model.pth
...
[E50] train_loss=0.1823 val_loss=0.2105 val_iou=0.5234 val_f1=0.6823
[TEST] Test IoU=0.5190, F1=0.6801, Precision=0.7245, Recall=0.6412

[SAVE] Training history → models/time_series/segmentation_metrics.json
```

**Target:** `val_f1 > 0.60` at epoch 50

---

## Next Steps After Training

1. **Download model:**
   ```bash
   # From Google Drive
   gdown <FILE_ID> -O models/time_series/segmentation_model.pth
   ```

2. **Inspect metrics:**
   ```bash
   cat models/time_series/segmentation_metrics.json | jq '.test_metrics'
   ```

3. **Load and validate locally:**
   ```python
   import torch
   from src.models.segmentation import UNet
   
   model = UNet(in_channels=14, out_channels=1).eval()
   model.load_state_dict(torch.load('models/time_series/segmentation_model.pth'))
   ```

---

## Colab Advantages vs Laptop

| Aspect | Colab | Laptop |
|--------|-------|--------|
| **GPU** | Free T4 (16GB VRAM) | Your GPU (if any) |
| **Speed** | ~4 hrs (T4) | ~24 hrs (CPU) |
| **Cost** | Free | Electricity |
| **Availability** | Always on (12h limit) | Only when laptop is on |
| **Data transfer** | Upload once to Drive | Local disk |

---

**Ready?** Create a Colab notebook and copy the code above. Takes 5 minutes to set up, then just click play!
