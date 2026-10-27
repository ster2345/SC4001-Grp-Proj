# SC4001-Grp-Proj

 **RESEARCH QN:** Does CLIP’s advantage over a fine-tuned CNN in detecting images from unseen generators hold once JPEG and image-size shortcuts are removed from the training data?

## 1. Overview

We train detectors on Stable Diffusion v1.4 images only and test them on three unseen generators: ADM, BigGAN and Midjourney. Data comes from GenImage (Zhu et al., 2023). We compare two detectors, a fine-tuned ResNet-50 and a linear probe on frozen CLIP ViT-B/32 features, on two versions of the same images: the original data and a bias-controlled ("cleaned") version where format, size and JPEG quality are matched between real and fake. This gives a 2×2 grid (detector × data version), evaluated per generator.

## 2. Environment

All work runs in Google Colab (Python 3.13). A T4 GPU is needed for CLIP feature extraction and model training.

```
pip install -r requirements.txt
```

Versions used: torch 2.11.0 (CUDA 13.0), transformers 5.20.2, pillow 11.3.0, numpy 2.1.3. Also pandas, pyyaml, tqdm, scikit-learn (versions not pinned). Colab's defaults change over time, so later sessions may differ. Rebuilding data from scratch also needs 7-Zip (`apt-get install -y p7zip-full`).

## 3. How to run it (step by step)

### 3.1 One-time setup (each person)

1. In Google Drive, add a shortcut to the shared **SC4001 Project** folder in My Drive and rename it exactly `sc4001_project`.
2. Open `notebooks/00_setup_colab.ipynb` and **save your own copy to your Drive**. Do not co-edit one shared notebook.
3. Optional, only for rebuilding data: add a shortcut to the GenImage Drive folder (https://drive.google.com/drive/folders/1jGt10bwTbhEZuGXLyvrCuxOI0cBqQ1FS) and name it `GenImage_src`.

### 3.2 Every session

1. Set the runtime to **T4 GPU** (Runtime, Change runtime type) if you will extract features or train.
2. Run the notebook's startup cells top to bottom:
   - mount Drive, clone or pull the repo, `pip install -r requirements.txt`
   - unpack `raw_downloads/original_subset.tar` to `/content/data/original`
   - unpack `raw_downloads/cleaned_subset.tar` to `/content/data/cleaned`
   - check that both folders contain **5,500** files and that `splits/splits.csv` exists
3. `git pull` before working.

Colab's `/content` is wiped when the session ends. Drive is the permanent copy, so save features, checkpoints and result tables to Drive.

### 3.3 Reproduce the main results

1. **Check the data:** `python src/check_splits.py --csv splits/splits.csv --root /content/data` should print `ALL CHECKS PASSED: 5500 images`.
2. **Load the CLIP features** (already extracted; no need to re-run CLIP):
   ```python
   import numpy as np
   z = np.load("/content/drive/MyDrive/sc4001_project/features/clip_vitb32/cleaned.npz")
   X, y = z["feats"], z["labels"]       # (5500, 512) float32, 0 real / 1 fake
   split, gen, src = z["splits"], z["generators"], z["sources"]
   ```
   Do the same with `original.npz`. L2-normalise features before training.
3. **Train the CLIP linear probe** on `split == "train"`, tune on `"val"`, once per data version (original, cleaned).
4. **Fine-tune the ResNet-50** on the same train and val images, once per data version, using the images under `/content/data/original` or `/content/data/cleaned` and the paths in `splits.csv`. Fix the seeds and save checkpoints to Drive.
5. **Evaluate on each unseen generator** G (adm, biggan, midjourney): fakes with `generator == G` plus reals with `generator == "real"` and `source == G`. Report accuracy and AUC for all four cells of the 2×2 grid.
6. **Robustness and analysis:** apply JPEG compression and resizing at test time, build the results tables and plots, and collect failure cases with example images.

Rules:
- `splits/splits.csv` comes from git. **Never regenerate it.** If `git pull` fails with "untracked working tree files would be overwritten: splits/splits.csv", delete the local copy and pull again.
- Do not rename files in `data/`.
- Settings live in `config.yaml` only.

## 4. Data

| In `sc4001_project/` on Drive | Contents |
|---|---|
| `raw_downloads/original_subset.tar` | 5,500 original images (PNG fakes, JPEG reals) as in GenImage |
| `raw_downloads/cleaned_subset.tar` | The same 5,500 images after bias control |
| `raw_downloads/*_list.txt` | 7-Zip file listings of the four GenImage archives |
| `raw_downloads/stats_original.csv`, `stats_cleaned.csv` | Format, size and JPEG quantisation statistics |
| `features/clip_vitb32/original.npz`, `cleaned.npz` | Frozen CLIP ViT-B/32 features |
| `splits.csv` | Copy of `splits/splits.csv` from git |

**Sample:** SD 1.4 gives 1,250 fakes + 1,250 reals (from `train/`). ADM, BigGAN and Midjourney give 500 fakes + 500 reals each (from `val/`). Total 5,500 images, sampled uniformly at random with seed 42.

### Splits

`splits/splits.csv` columns: `image_id, path_original, path_cleaned, label, generator, source, md5, split`. `label`: 0 = real, 1 = fake. `source`: the GenImage archive the file came from.

| Split | Content |
|---|---|
| train | SD 1.4: 1,000 fake + 1,000 real |
| val | SD 1.4: 250 fake + 250 real |
| test | ADM, BigGAN, Midjourney: 500 fake each, plus 1,500 reals (500 per archive) |

Test reals have `generator == "real"`, so use the `source` column to match them to a test generator. MD5 hashing found no duplicates. `check_splits.py` asserts that no image ID, path or hash appears in two splits and that train and val contain only SD 1.4 fakes. MD5 only catches byte-identical files.

### Data audit

Every fake is a PNG at one fixed, generator-specific size (BigGAN 128, ADM 256, SD 1.4 512, Midjourney 1024). Every real is a JPEG with 819 distinct sizes (mostly 500×375) and mean quantisation 8.17. In the original data, format and size therefore separate real from fake perfectly.

### Bias-controlled ("cleaned") version

Every image, real and fake, goes through one identical pipeline: centre-crop to a square, bicubic resize to 256×256, JPEG quality 95 (settings in `config.yaml`). All 5,500 outputs were verified as JPEG, 256×256, with identical quantisation (mean 5.77).

This is our own recipe. It is related to, but not a replication of, the Unbiased GenImage protocol (Grommelt et al., 2024), which selects reals by size and JPEG quality and re-encodes fakes to match. Limitations: BigGAN is upsampled 2× and Midjourney downsampled 4×, ADM fakes are only recompressed, and reals are JPEG-compressed twice. Format, size and compression cues are removed, but resampling traces may remain.

### CLIP features

Frozen CLIP ViT-B/32 (`openai/clip-vit-base-patch32`). Keys in each `.npz`: `feats, labels, image_ids, generators, sources, splits, paths`. Rows follow `splits.csv` order, and the original and cleaned files have identical rows. Features are unnormalised. CLIP's own preprocessing (resize to 224, centre-crop) is applied to both versions.

## 5. Models and experiments

- **CLIP linear probe:** a linear classifier on L2-normalised frozen CLIP ViT-B/32 features, trained on the SD 1.4 train split and tuned on val.
- **ResNet-50:** fine-tuned on the same SD 1.4 train and val images, with fixed seeds and checkpoints saved to Drive (Colab sessions can disconnect).
- **2×2 grid:** each detector is trained once on the original data and once on the cleaned data, so the only difference between the two training sets is the cleaning.
- **Seeds and ablations:** repeated runs with different seeds measure variation, and ablations test the effect of individual choices.

## 6. Evaluation and analysis

- **Metrics:** accuracy and AUC, computed per unseen generator (ADM, BigGAN, Midjourney) as described in Section 3.3. Test data is never used for tuning.
- **Cross-generator results table:** detector × data version × generator.
- **Robustness:** a pipeline that applies JPEG compression and resizing to test images, and compares the drop in performance for each detector and data version.
- **Analysis:** plots, failure cases with example images, and a discussion of why the results came out as they did.

## 7. Rebuilding the data from scratch

Only needed if the tars or features are lost. Run in Colab in `/content/repo`, after the setup in Section 3.1.

**1. List the archive contents** (GenImage is split multi-volume zips; 7z reads the index without extracting):

```
cd /content/drive/MyDrive/GenImage_src/ADM && 7z l imagenet_ai_0508_adm.zip > /content/adm_list.txt
```

Repeat for SD 1.4 (`stable_diffusion_v_1_4/imagenet_ai_0419_sdv4.zip`), BigGAN (`BigGAN/imagenet_ai_0419_biggan.zip`) and Midjourney (`Midjourney/imagenet_midjourney.zip`), writing `sd14_list.txt`, `biggan_list.txt` and `midjourney_list.txt`.

**2. Sample images:**

```
A=/content/drive/MyDrive/GenImage_src
python src/extract_subset_7z.py --archive "$A/ADM/imagenet_ai_0508_adm.zip" --listing /content/adm_list.txt --tag adm --part val --n 500
python src/extract_subset_7z.py --archive "$A/BigGAN/imagenet_ai_0419_biggan.zip" --listing /content/biggan_list.txt --tag biggan --part val --n 500
python src/extract_subset_7z.py --archive "$A/Midjourney/imagenet_midjourney.zip" --listing /content/midjourney_list.txt --tag midjourney --part val --n 500
python src/extract_subset_7z.py --archive "$A/stable_diffusion_v_1_4/imagenet_ai_0419_sdv4.zip" --listing /content/sd14_list.txt --tag sd14 --part train --n 1250
tar -cf /content/drive/MyDrive/sc4001_project/raw_downloads/original_subset.tar -C /content/data original
```

Output goes to `/content/data/original/{real,sd14,adm,biggan,midjourney}/`. The SD 1.4 run takes about 35 minutes.

**3. Inspect:** `python src/inspect_data.py --root /content/data/original --out stats.csv`

**4. Splits** (already done and frozen; shown for the record, do not re-run):

```
python src/dedupe_and_split.py --root /content/data
python src/check_splits.py --csv splits/splits.csv --root /content/data
```

**5. Cleaned version:**

```
python src/clean_images.py --root /content/data
python src/clean_images.py --root /content/data --verify
python src/inspect_data.py --root /content/data/cleaned --out stats_cleaned.csv
tar -cf /content/drive/MyDrive/sc4001_project/raw_downloads/cleaned_subset.tar -C /content/data cleaned
```

**6. CLIP features** (GPU; saves shards so it can resume after a disconnect):

```
python src/extract_clip.py --version original
python src/extract_clip.py --version cleaned
python src/extract_clip.py --version original --verify
python src/extract_clip.py --version cleaned --verify
```

Each verify should end with `VERIFY PASSED ... (5500, 512)`.


## 8. Repository layout

```
config.yaml                     all settings (seed, sample sizes, cleaning recipe, CLIP model)
requirements.txt
splits/splits.csv               FROZEN train/val/test split
src/
  common.py                     config loading, seed, image listing
  extract_subset_7z.py          sample images from the GenImage archives
  inspect_data.py               format / size / JPEG quantisation statistics
  dedupe_and_split.py           hash files, build splits.csv (run once)
  check_splits.py               proves no file appears in two splits
  clean_images.py               bias-controlled version of the data
  extract_clip.py               frozen CLIP features
  (model training and evaluation code added by Members B and C)
notebooks/00_setup_colab.ipynb  startup notebook
```

## 9. Work split

- **Esther** data download and sampling, splits, bias-controlled data, CLIP features, dataset statistics, README.
- **Jing Ni:** ResNet-50 and CLIP linear-probe training, seeds, checkpoints, the 2×2 grid, ablations.
- **Nicole:** metrics, cross-generator tables, JPEG/resize robustness tests, plots, failure analysis.
- **All:** AI Use Statement, reading and editing the whole report.