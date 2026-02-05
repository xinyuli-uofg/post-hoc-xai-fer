

import os, sys, cv2, shutil, pickle, argparse, datetime
import numpy as np
from glob import glob
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

sys.path.append("/path/to/this_project")

from xtools.xy_draw_au_map import (
    show_GT_au_heatmap_for_image, visualize_patch_based_heatmap,
    visualize_weighted_heatmaps, visualize_GT_weighted_heatmaps
)

from agcem_test_affectnet_ROI import load_config
from xtools.affectnet_loader_ROI import generate_data
from cem.train.training import construct_model

DATA_DIR         = '/path/to/your/test_dataset'
BASE_OUTPUT_DIR  = '/path/to/your/output/dir'
MODEL_CHECKPOINT = '/path/to/your/model_checkpoint'
VAL_PKL          = '/path/to/your/validation_pkl' # for concept-based interpretability only

CONCEPT_MAP = [
    "Upper:FAU1","Upper:FAU2","Upper:FAU4","Upper:FAU5","Upper:FAU6","Upper:FAU7",
    "Lower:FAU9","Lower:FAU10","Lower:FAU12","Lower:FAU14","Lower:FAU15","Lower:FAU17",
    "Lower:FAU20","Lower:FAU23","Lower:FAU25","Lower:FAU26","Lower:FAU28","Upper:FAU45",
]
LABEL_MAP = ["Neutral","Happiness","Sadness","Surprise","Fear","Disgust","Anger","Contempt"]

# ── transforms ────────────────────────────────────────────────────────────────
tfm = T.Compose([
    T.Resize([224, 224]),
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

def au_score_table(scores, concept_map, n_rows=None):
    scores = np.asarray(scores).flatten()
    order  = np.argsort(scores)[::-1]
    if n_rows is not None:
        order = order[:n_rows]
    w = max(len(concept_map[i]) for i in order) + 2
    lines = [f"{'AU name'.ljust(w)}| score", '-'*(w+7)]
    lines += [f"{concept_map[i].ljust(w)}| {scores[i]:.4f}" for i in order]
    lines.append('-'*(w+7))
    return lines

def _to_uint8_hwc(arr):
    arr = np.asarray(arr)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255)
        if arr.max() <= 1.0:
            arr = (arr * 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
    return arr

def overlay_and_save(raw_bgr, heat_0_1, save_path, alpha_raw=0.6, alpha_heat=0.4, cmap=cv2.COLORMAP_JET):
    heat = (np.clip(heat_0_1, 0, 1) * 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cmap)
    out  = cv2.addWeighted(raw_bgr, alpha_raw, heat, alpha_heat, 0)
    cv2.imwrite(save_path, out)

def _predict_logits_tensor(model, x):
    # x: [B,3,224,224] (normalized)
    c_sem, _, y_pred, _, _ = model(x)
    return y_pred

def resolve_module_by_name(root: nn.Module, dotted: str) -> nn.Module:
    cur = root
    for part in dotted.split('.'):
        if not hasattr(cur, part):
            raise AttributeError(f"Module path not found: {dotted}")
        cur = getattr(cur, part)
    if not isinstance(cur, nn.Module):
        raise AttributeError(f"Resolved object is not a Module: {dotted}")
    return cur


# ───────────────────────────────────────────────────────────────────────────────
# Model / data
# ───────────────────────────────────────────────────────────────────────────────

def build_model():
    config = load_config()
    config['x2c_used_pretrain_path'] = "/path/to/your/trained/model"

    train_dl, val_dl, test_dl, imbalance, (n_concepts, n_tasks) = generate_data(
        config=config, seed=42, output_dataset_vars=True
    )
    print("n_concepts:", n_concepts, "n_tasks:", n_tasks)

    model = construct_model(
        n_concepts, n_tasks, config, imbalance=imbalance,
        task_class_weights=config.get('task_class_weights', None),
    )

    if MODEL_CHECKPOINT.endswith('.pt'):
        print("Resume Training from pt:", MODEL_CHECKPOINT)
        model.load_state_dict(torch.load(MODEL_CHECKPOINT, map_location='cpu'))
    elif MODEL_CHECKPOINT.endswith('.ckpt'):
        print("Resume Training from ckpt:", MODEL_CHECKPOINT)
        ckpt = torch.load(MODEL_CHECKPOINT, map_location='cpu')
        state = ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt))
        model.load_state_dict(state)

    model.freeze()
    model.to(device).eval()
    return model, config

def load_val_meta(val_pkl=VAL_PKL, data_dir=DATA_DIR):
    with open(val_pkl, 'rb') as f:
        val_aff_data = pickle.load(f)
    path2meta = {}
    for d in val_aff_data:
        key = d['img_path']
        path2meta[key] = d
        path2meta.setdefault(os.path.join(data_dir, os.path.basename(key)), d)
    return path2meta


# ───────────────────────────────────────────────────────────────────────────────
# Core per-image outputs (your original forward + concept/spatial maps)
# ───────────────────────────────────────────────────────────────────────────────

def run_core_outputs(model, img_path, path2meta, out_root):
    stem = os.path.splitext(os.path.basename(img_path))[0]
    meta = path2meta.get(img_path, None)
    gt_label = LABEL_MAP[meta['class_label']] if meta else 'unknown'

    out_dir = ensure_dir(os.path.join(out_root, f"{stem}_{gt_label.lower()}"))
    img_dir = ensure_dir(os.path.join(out_dir, 'images'))
    log_path = os.path.join(out_dir, 'results.txt')

    def log(*args):
        msg = ' '.join(map(str, args))
        print(msg)
        with open(log_path, 'a') as f:
            f.write(msg + '\n')

    if not os.path.isfile(log_path):
        log("#"*60)
        log(f"Image : {img_path}")
        log(f"GT    : {gt_label}")
        log("#"*60)

    with torch.no_grad():
        inp = tfm(Image.open(img_path).convert('RGB')).unsqueeze(0).repeat(128,1,1,1).to(device)
        c_sem, _, y_pred, spatial_map, _ = model(inp)

    pred_idx = int(y_pred[0].argmax().item())
    pred_cls = LABEL_MAP[pred_idx]
    scores_np = c_sem[0].detach().cpu().numpy()

    log("Pred class :", pred_cls)
    log("Pred AU table:")
    for l in au_score_table(scores_np, CONCEPT_MAP, n_rows=None):
        log(l)

    # Visualizations
    save_gt      = os.path.join(img_dir, f'{stem}_au_GT.jpg')
    save_pred    = os.path.join(img_dir, f'{stem}_au_pred.jpg')
    save_norm    = os.path.join(img_dir, f'{stem}_norm_pred.jpg')
    save_norm_gt = os.path.join(img_dir, f'{stem}_norm_GT.jpg')

    au_maps_gt = show_GT_au_heatmap_for_image(img_path, save_gt, label=False)
    att_map    = spatial_map[0].detach().cpu().numpy()
    raw_cv     = cv2.imread(img_path)
    visualize_patch_based_heatmap(raw_cv, att_map, save_pred, normalize=True)

    scores_np_w = scores_np.copy()
    scores_np_w[scores_np_w <= 0.5] = 0.0
    visualize_weighted_heatmaps(raw_cv, att_map, scores_np_w, save_norm, normalize=True)

    try:
        visualize_GT_weighted_heatmaps(img_path, save_norm_gt, au_maps_patch_level=au_maps_gt)
    except FileNotFoundError:
        log("AU CSV not found – skipped GT-weighted heat-map for", img_path)

    raw_copy = os.path.join(img_dir, f'{stem}.jpg')
    if not os.path.isfile(raw_copy):
        shutil.copy(img_path, raw_copy)
    log("Original images saved to", img_dir)
    log("")

    return {
        "stem": stem,
        "gt_label": gt_label,
        "pred_idx": pred_idx,
        "pred_cls": pred_cls,
        "scores_np": scores_np,
        "out_dir": out_dir,
        "img_dir": img_dir,
        "log_path": log_path,
    }


def iter_images(data_dir):
    return sorted(glob(os.path.join(data_dir, "*.jpg")))


def add_common_cli(parser: argparse.ArgumentParser):
    parser.add_argument('--data_dir', default=DATA_DIR, help='Folder of images')
    parser.add_argument('--out_root', default=BASE_OUTPUT_DIR, help='Root output dir')
    parser.add_argument('--no_core', action='store_true',
                        help="Skip the core outputs (AU/attention maps).")
    return parser
