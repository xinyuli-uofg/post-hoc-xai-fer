# post-hoc-xai-fer
A post-hoc XAI toolkits for generating model explaination for FER

Supported methods:
- LIME
- SHAP (abs / signed / posneg)
- Integrated Gradients (pixel IG or LayerIG)
- SmoothGrad
- Occlusion Sensitivity
- Grad-CAM
- LayerCAM
- LRP (Captum)
- AGCM ante-hoc interpretability


---

## 1) Online documented instruction

Will be online soon

---

## 2) Prerequisites

### 2.1 Python packages

Install the Python packages 
```bash
pip install -U shap lime scikit-image captum opencv-python pillow numpy torch torchvision
````

Notes:

* `torch` / `torchvision` should match your CUDA setup.
* `opencv-python` is required for reading/writing overlays (`cv2`).
* `captum` is required for IG, Occlusion, Grad-CAM, and LRP.
* `shap`, `lime`, `scikit-image` are required for SHAP and LIME.

AGCM-based interpretability (optional)

* If you want to run the AGCM-based interpretability components used in this project (rather than adapting the runners to your own model), you can install the dependencies and (optionally) download the pretrained weights from the AGCM project:

  * [https://github.com/xinyuli-uofg/agcm](https://github.com/xinyuli-uofg/agcm)
* This step is optional if you only want to use the post-hoc XAI scripts with your own model and you have already satisfied the package requirements above.

---

## 3) Configure paths

Open `xai_core.py` and set these paths to match your environment:

```python
DATA_DIR         = '/path/to/your/test_dataset'
BASE_OUTPUT_DIR  = '/path/to/your/output/dir'
MODEL_CHECKPOINT = '/path/to/your/model_checkpoint'
VAL_PKL          = '/path/to/your/validation_pkl' # for concept-based interpretability only
```

* `DATA_DIR`: folder containing `.jpg` images to process.
* `BASE_OUTPUT_DIR`: root folder where results will be written.
* `MODEL_CHECKPOINT`: model checkpoint to load (`.ckpt` or `.pt`).
* `VAL_PKL`: AffectNet metadata file used to map image paths to GT labels (optional but recommended for correct folder naming).

You can override `DATA_DIR` and `BASE_OUTPUT_DIR` at runtime via CLI:

* `--data_dir`
* `--out_root`

---

## 4) Output layout

For each input image `IMG.jpg`, outputs are written under:

```
<out_root>/
  <stem>_<gt_label_lower>/
    results.txt
    images/
      <stem>.jpg
      <stem>_au_GT.jpg
      <stem>_au_pred.jpg
      <stem>_norm_pred.jpg
      <stem>_norm_GT.jpg
      <stem>_<method>.jpg
```

If you run with `--no_core`, the scripts skip the “core outputs” stage and will place results in a fallback folder:

```
<out_root>/
  <stem>_unknown/
    images/
      <stem>_<method>.jpg
```

---

## 5) Running the scripts

All scripts share common arguments:

* `--data_dir`: folder containing `.jpg` files
* `--out_root`: output root directory
* `--no_core`: skip the original (core) AU/attention visualizations

### 5.1 Core outputs only (original visualizations)

```bash
python run_core_outputs.py --data_dir /path/to/images --out_root /path/to/out
```

### 5.2 LIME

```bash
python run_lime.py --data_dir /path/to/images --out_root /path/to/out --lime_samples 600
```

Optional:

* `--seg slic` or `--seg quickshift`
* `--kernel_width 0.25`

Example:

```bash
python run_lime.py --data_dir /path/to/images --out_root /path/to/out --seg slic --lime_samples 1000 --kernel_width 0.25
```

### 5.3 SHAP

```bash
python run_shap.py --data_dir /path/to/images --out_root /path/to/out --shap_evals 300 --shap_mode abs
```

Modes:

* `--shap_mode abs` (single file: `_shap.jpg`)
* `--shap_mode signed` (single file: `_shap_signed.jpg`)
* `--shap_mode posneg` (two files: `_shap_pos.jpg`, `_shap_neg.jpg`)

Attribute to logits (instead of softmax probability):

```bash
python run_shap.py --data_dir /path/to/images --out_root /path/to/out --shap_logits
```

### 5.4 Integrated Gradients (pixel IG)

```bash
python run_ig.py --data_dir /path/to/images --out_root /path/to/out --ig_steps 50
```

### 5.5 Integrated Gradients (LayerIG at a layer)

Use `--ig_layer` to run LayerIntegratedGradients at a specific layer. For ViT/MAE, a common choice is:

* `pre_concept_model.patch_embed.proj`

Example:

```bash
python run_ig.py --data_dir /path/to/images --out_root /path/to/out \
  --ig_layer pre_concept_model.patch_embed.proj \
  --ig_steps 50 \
  --ig_internal_bs 8 \
  --ig_nt_samples 4
```

Optional:

* `--ig_attr_to_input` to attribute to the layer input (often best for conv/patch-proj layers)
* `--patch 16` (patch pooling/upsample; set `--patch 0` to disable)
* `--p_low 80 --p_high 99.5 --gamma 0.9` contrast controls

### 5.6 SmoothGrad

```bash
python run_smoothgrad.py --data_dir /path/to/images --out_root /path/to/out \
  --smoothgrad_samples 25 --smoothgrad_sigma 0.15 --batch_size 8 --agg abs
```

Optional:

* `--agg abs | sq | raw | gi`
* `--patch 16` (set `--patch 0` to disable)
* `--p_low`, `--p_high`, `--gamma` contrast controls

### 5.7 Occlusion Sensitivity

```bash
python run_occlusion.py --data_dir /path/to/images --out_root /path/to/out --occ_kernel 21 --occ_stride 10
```

### 5.8 Grad-CAM

If your model contains a Conv2d at the patch embedding (ViT/MAE), explicitly specify it:

```bash
python run_gradcam.py --data_dir /path/to/images --out_root /path/to/out \
  --gradcam_layer pre_concept_model.patch_embed.proj
```

If omitted, the script attempts to auto-select a ViT/MAE patch embedding conv, then falls back to the last Conv2d under `pre_concept_model`.

### 5.9 LayerCAM

Recommended for ViT/MAE patch embedding conv:

```bash
python run_layercam.py --data_dir /path/to/images --out_root /path/to/out \
  --layercam_layer pre_concept_model.patch_embed.proj
```

### 5.10 LRP (Captum)

LRP sometimes requires wrapping reused modules (e.g., head activation reused multiple times). Defaults are provided:

```bash
python run_lrp.py --data_dir /path/to/images --out_root /path/to/out \
  --lrp_rule epsilon --lrp_eps 1e-6
```

If you need different wrap paths:

```bash
python run_lrp.py --data_dir /path/to/images --out_root /path/to/out \
  --wrap_paths head.relu head.sigmoid head.fc \
  --wrap_calls 1024
```

To disable AMP:

```bash
python run_lrp.py --data_dir /path/to/images --out_root /path/to/out --no_amp
```

---

## 6) Typical workflows

### Workflow A: Generate core outputs + one method

Example: core + LayerCAM

```bash
python run_layercam.py --data_dir /path/to/images --out_root /path/to/out \
  --layercam_layer pre_concept_model.patch_embed.proj
```

### Workflow B: Generate only one method (skip core outputs)

Example: only SHAP

```bash
python run_shap.py --data_dir /path/to/images --out_root /path/to/out --no_core --shap_mode abs
```

### Workflow C: Run multiple methods

Run separate scripts one after another, reusing the same output folder:

```bash
python run_core_outputs.py --data_dir /path/to/images --out_root /path/to/out
python run_shap.py         --data_dir /path/to/images --out_root /path/to/out --shap_mode posneg
python run_gradcam.py      --data_dir /path/to/images --out_root /path/to/out --gradcam_layer pre_concept_model.patch_embed.proj
python run_ig.py           --data_dir /path/to/images --out_root /path/to/out --ig_layer pre_concept_model.patch_embed.proj
```

---

## 7) Troubleshooting

### 7.1 Import errors for project modules (cem/xtools)

* Verify `sys.path.append(...)` in `xai_core.py` points to the correct repository root.
* Confirm that `agcem_test_affectnet_ROI.py`, `xtools/`, and `cem/` are importable under that path.

### 7.2 CUDA out of memory

* Reduce batch-like parameters:

  * SHAP: reduce `--shap_evals`
  * LIME: reduce `--lime_samples`
  * IG: reduce `--ig_steps`, reduce `--ig_internal_bs`, reduce `--ig_nt_samples`
  * SmoothGrad: reduce `--smoothgrad_samples`, reduce `--batch_size`
  * Occlusion: increase `--occ_stride`, reduce `--occ_kernel`
* Close other GPU jobs.
* Consider running `--no_core` to skip the repeated-128 forward used by the original visualization pipeline.

### 7.3 Grad-CAM cannot find a Conv2d

* Provide `--gradcam_layer pre_concept_model.patch_embed.proj` for ViT/MAE-style backbones.
* If your model uses a different naming, inspect your model and pass the correct dotted path.

### 7.4 LRP failures due to module reuse

* Adjust `--wrap_paths` to target the specific reused modules in your model.
* Increase `--wrap_calls` if it reports exceeding `max_calls`.
* Keep LRP on GPU in this setup; moving LRP to CPU may fail depending on the Captum version and model internals.

---




