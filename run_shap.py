
import os, datetime
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as F
import shap

from xai_core import (
    add_common_cli, build_model, load_val_meta, ensure_dir, iter_images,
    run_core_outputs, tfm, device, _to_uint8_hwc, _predict_logits_tensor
)

def _predict_proba_from_hw3_np(model, imgs_hw3):
    if isinstance(imgs_hw3, np.ndarray):
        batch = [imgs_hw3] if imgs_hw3.ndim == 3 else list(imgs_hw3)
    else:
        batch = list(imgs_hw3)
    with torch.no_grad():
        xs = torch.stack([tfm(Image.fromarray(_to_uint8_hwc(x))) for x in batch]).to(device)
        logits = _predict_logits_tensor(model, xs)
        return F.softmax(logits, dim=1).cpu().numpy()

def _predict_logits_from_hw3_np(model, imgs_hw3):
    if isinstance(imgs_hw3, np.ndarray):
        batch = [imgs_hw3] if imgs_hw3.ndim == 3 else list(imgs_hw3)
    else:
        batch = list(imgs_hw3)
    with torch.no_grad():
        xs = torch.stack([tfm(Image.fromarray(_to_uint8_hwc(x))) for x in batch]).to(device)
        logits = _predict_logits_tensor(model, xs)
        return logits.detach().cpu().numpy()

def run_shap(model, img_path, target_idx, max_evals=300, shap_mode='abs', use_logits=False):
    img_rgb = np.array(Image.open(img_path).convert('RGB'))
    H, W = img_rgb.shape[:2]
    raw_bgr = cv2.imread(img_path)

    masker = shap.maskers.Image("blur(64,64)", (H, W, 3))
    predict_fn = (lambda xs: _predict_logits_from_hw3_np(model, xs)) if use_logits \
                 else (lambda xs: _predict_proba_from_hw3_np(model, xs))
    explainer = shap.Explainer(predict_fn, masker)

    try:
        selector = shap.Explanation.indices([int(target_idx)])
    except Exception:
        selector = shap.Explanation.argsort.flip[:1]

    sv = explainer(img_rgb[None, ...], max_evals=int(max_evals), outputs=selector)

    vals = sv.values
    if isinstance(vals, list):
        vals = vals[0]
    if vals.ndim == 5:
        vals = vals[..., 0]
    vals = vals[0]  # [H,W,3]

    saved_paths = []

    if shap_mode == 'abs':
        heat = np.mean(np.abs(vals), axis=2)
        heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
        saved_paths.append(("shap.jpg", heat, cv2.COLORMAP_JET))

    elif shap_mode == 'signed':
        signed = np.mean(vals, axis=2)
        m = np.max(np.abs(signed)) + 1e-8
        signed = signed / m
        signed = cv2.resize(signed, (raw_bgr.shape[1], raw_bgr.shape[0]), interpolation=cv2.INTER_CUBIC)
        pos = np.clip(signed, 0, 1)
        neg = np.clip(-signed, 0, 1)

        # merge into one RGB blend (hot + cool)
        pos_u8 = (pos * 255).astype(np.uint8)
        neg_u8 = (neg * 255).astype(np.uint8)
        pos_cm = cv2.applyColorMap(pos_u8, cv2.COLORMAP_HOT)
        neg_cm = cv2.applyColorMap(neg_u8, cv2.COLORMAP_COOL)
        blend = np.clip(pos_cm.astype(np.int16) + neg_cm.astype(np.int16), 0, 255).astype(np.uint8)

        out = cv2.addWeighted(raw_bgr, 0.6, blend, 0.4, 0)
        return {"shap_signed.jpg": out}

    elif shap_mode == 'posneg':
        signed = np.mean(vals, axis=2)
        signed = cv2.resize(signed, (raw_bgr.shape[1], raw_bgr.shape[0]), interpolation=cv2.INTER_CUBIC)

        pos = np.maximum(signed, 0.0)
        neg = np.maximum(-signed, 0.0)

        if pos.max() > 0:
            pos = (pos - pos.min()) / (pos.max() - pos.min() + 1e-8)
        if neg.max() > 0:
            neg = (neg - neg.min()) / (neg.max() - neg.min() + 1e-8)

        out_pos = cv2.addWeighted(raw_bgr, 0.6, cv2.applyColorMap((pos*255).astype(np.uint8), cv2.COLORMAP_HOT), 0.4, 0)
        out_neg = cv2.addWeighted(raw_bgr, 0.6, cv2.applyColorMap((neg*255).astype(np.uint8), cv2.COLORMAP_COOL), 0.4, 0)
        return {"shap_pos.jpg": out_pos, "shap_neg.jpg": out_neg}

    else:
        raise ValueError("Unknown shap_mode")

    # abs mode rendering
    out_map = {}
    for name, heat01, cmap in saved_paths:
        heat_u8 = (cv2.resize(heat01, (raw_bgr.shape[1], raw_bgr.shape[0])) * 255).astype(np.uint8)
        cm = cv2.applyColorMap(heat_u8, cmap)
        out = cv2.addWeighted(raw_bgr, 0.6, cm, 0.4, 0)
        out_map[name] = out
    return out_map

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run SHAP only.")
    add_common_cli(parser)
    parser.add_argument('--shap_evals', type=int, default=300)
    parser.add_argument('--shap_mode', choices=['abs','signed','posneg'], default='abs')
    parser.add_argument('--shap_logits', action='store_true')
    args = parser.parse_args()

    model, _ = build_model()
    path2meta = load_val_meta(data_dir=args.data_dir)

    imgs = iter_images(args.data_dir)
    print(f"\nProcessing {len(imgs)} images...\n")
    t0 = datetime.datetime.now()

    ensure_dir(args.out_root)
    for p in imgs:
        info = None
        if not args.no_core:
            info = run_core_outputs(model, p, path2meta, args.out_root)
            pred_idx, img_dir, stem = info["pred_idx"], info["img_dir"], info["stem"]
        else:
            x = tfm(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                pred_idx = int(_predict_logits_tensor(model, x).argmax(1).item())
            img_dir = ensure_dir(os.path.join(args.out_root, os.path.splitext(os.path.basename(p))[0] + "_unknown", "images"))
            stem = os.path.splitext(os.path.basename(p))[0]

        outs = run_shap(model, p, target_idx=pred_idx, max_evals=args.shap_evals,
                        shap_mode=args.shap_mode, use_logits=args.shap_logits)

        for fname, out_img in outs.items():
            save_to = os.path.join(img_dir, f"{stem}_{fname}")
            cv2.imwrite(save_to, out_img)
            print("SHAP saved to", save_to)

    print("Runtime:", datetime.datetime.now() - t0)

if __name__ == "__main__":
    main()
