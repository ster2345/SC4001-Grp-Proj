# SC4001-Grp-Proj

## Data preparation

All steps run in Google Colab. Settings (seed, sample sizes, cleaning recipe, CLIP model) are in `config.yaml`.

### Dependencies
`pip install -r requirements.txt`. Extraction also needs 7-Zip (`apt-get install -y p7zip-full`).
Versions used: torch 2.11.0 (CUDA 13.0), transformers 5.20.2, pillow 11.3.0, numpy 2.1.3 (Google Colab, Python 3.13). Also: pandas, pyyaml, tqdm, scikit-learn (versions not recorded; run pip freeze if the group wants them listed).

### Setup
1. Add a shortcut to the shared Drive folder in My Drive, named exactly `sc4001_project`.
2. Add a shortcut to the GenImage Google Drive folder
   (https://drive.google.com/drive/folders/1jGt10bwTbhEZuGXLyvrCuxOI0cBqQ1FS), named `GenImage_src`.

### Quick start (no need to rebuild data)
Open `notebooks/00_setup_colab.ipynb`, save a copy to your own Drive, and run the startup cells.
They unpack `original_subset.tar` and `cleaned_subset.tar` from Drive and print 5500 for each.
CLIP features are already saved in `features/clip_vitb32/{original,cleaned}.npz`.
`splits/splits.csv` is frozen in git. Do not regenerate it.

### Rebuilding from scratch
**1. List the archive contents** (GenImage is split multi-volume zips; 7z reads the index without extracting):
```
cd <GenImage_src>/ADM && 7z l imagenet_ai_0508_adm.zip > /content/adm_list.txt
```
Repeat for SD 1.4 (`imagenet_ai_0419_sdv4.zip`), BigGAN (`imagenet_ai_0419_biggan.zip`), Midjourney (`imagenet_midjourney.zip`).

**2. Sample images** (random, seed 42; fakes from `ai/`, reals from `nature/`):
```
python src/extract_subset_7z.py --archive "<path>/ADM/imagenet_ai_0508_adm.zip" --listing /content/adm_list.txt --tag adm --part val --n 500
python src/extract_subset_7z.py --archive "<path>/Midjourney/imagenet_midjourney.zip" --listing /content/midjourney_list.txt --tag midjourney --part val --n 500
python src/extract_subset_7z.py --archive "<path>/BigGAN/imagenet_ai_0419_biggan.zip" --listing /content/biggan_list.txt --tag biggan --part val --n 500
python src/extract_subset_7z.py --archive "<path>/stable_diffusion_v_1_4/imagenet_ai_0419_sdv4.zip" --listing /content/sd14_list.txt --tag sd14 --part train --n 1250
```
Output: `/content/data/original/{real,sd14,adm,biggan,midjourney}/` (5,500 images).

**3. Inspect** formats, sizes and JPEG quantisation:
```
python src/inspect_data.py --root /content/data/original --out stats.csv
```

**4. Build splits and prove no overlap** (run once; the committed `splits/splits.csv` is the reference):
```
python src/dedupe_and_split.py --root /content/data
python src/check_splits.py --csv splits/splits.csv --root /content/data
```
Train: SD 1.4, 1000 real + 1000 fake. Val: 250 + 250. Test: 500 fakes per unseen generator (ADM, BigGAN, Midjourney) plus 500 reals from the same archive.
For a test generator G, use fakes with `generator == G` and reals with `generator == "real"` and `source == G`.

**5. Bias-controlled version** (centre-crop to square, bicubic resize to 256, JPEG quality 95, identical for real and fake):
```
python src/clean_images.py --root /content/data
python src/clean_images.py --root /content/data --verify
```

**6. CLIP features** (frozen ViT-B/32; needs a GPU; saves shards so it can resume):
```
python src/extract_clip.py --version original
python src/extract_clip.py --version cleaned
python src/extract_clip.py --version original --verify
python src/extract_clip.py --version cleaned --verify
```
Each `.npz` holds `feats` (5500x512, unnormalised), `labels` (0 real, 1 fake), `image_ids`, `generators`, `sources`, `splits`, `paths`, in `splits.csv` row order.
L2-normalise features before training a linear probe.