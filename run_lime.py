

import os, shutil, datetime
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as F

from lime import lime_image
from skimage.segmentation import slic, quickshift

from xai_core import (
    add_common_cli, build_model, load_val_meta, ensure_dir, iter_images,
    run_core_outputs, tfm, device, _to_uint8_hwc, _predict_logits_tensor
)

def run_lime(model, img_path, target_idx, num_samples=1000, seg='slic',
             random_state=0, kernel_width=0.25, positive_only=False, num_features=25):
    img_rgb = np.array(Image.open(img_path).convert('RGB'))
    img_blur = cv2.GaussianBlur(img_rgb, ksize=(0, 0), sigmaX=2, sigmaY=2)

    def seg_slic(x):
        try:
            return slic(x, n_segments=120, compactness=10, sigma=0, start_label=0, channel_axis=-1)
        except TypeError:
            return slic(x, n_segments=120, compactness=10, sigma=0, start_label=0, multichannel=True)

    def seg_qs(x):
        try:
            return quickshift(x, kernel_size=6, max_dist=10, ratio=0.5, channel_axis=-1)
        except TypeError:
            return quickshift(x, kernel_size=6, max_dist=10, ratio=0.5, multichannel=True)

    seg_fn = seg_qs if seg == 'quickshift' else seg_slic

    def classifier_on_manifold(imgs_np_hw3):
        fixed = []
        for arr in imgs_np_hw3:
            arr8 = _to_uint8_hwc(arr)
            m = (arr8[..., 0] == 0) & (arr8[..., 1] == 0) & (arr8[..., 2] == 0)
            if m.any():
                arr8 = arr8.copy()
                Hb, Wb = img_blur.shape[:2]
                Ha, Wa = arr8.shape[:2]
                blur_resized = img_blur if (Hb, Wb) == (Ha, Wa) else cv2.resize(img_blur, (Wa, Ha), cv2.INTER_AREA)
                arr8[m] = blur_resized[m]
            fixed.append(arr8)

        with torch.no_grad():
            xs = torch.stack([tfm(Image.fromarray(x)) for x in fixed]).to(device)
            logits = _predict_logits_tensor(model, xs)
            probs = F.softmax(logits, dim=1).cpu().numpy()
        return probs

    explainer = lime_image.LimeImageExplainer(random_state=random_state, kernel_width=kernel_width)
    explanation = explainer.explain_instance(
        img_rgb,
        classifier_fn=classifier_on_manifold,
        top_labels=0,
        labels=[int(target_idx)],
        hide_color=(0, 0, 0),
        num_samples=int(num_samples),
        segmentation_fn=seg_fn,
        batch_size=64,
    )

    _img, mask = explanation.get_image_and_mask(
        label=int(target_idx),
        positive_only=positive_only,
        hide_rest=False,
        num_features=num_features,
        min_weight=0.0,
    )

    mask = mask.astype(np.float32)
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)

    raw_bgr = cv2.imread(img_path)
    heat_u8 = (cv2.resize(mask, (raw_bgr.shape[1], raw_bgr.shape[0]), interpolation=cv2.INTER_CUBIC) * 255).astype(np.uint8)
    heat_cm = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    out = cv2.addWeighted(raw_bgr, 0.6, heat_cm, 0.4, 0)
    return out

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run LIME only.")
    add_common_cli(parser)
    parser.add_argument('--lime_samples', type=int, default=1000)
    parser.add_argument('--seg', choices=['slic', 'quickshift'], default='slic')
    parser.add_argument('--kernel_width', type=float, default=0.25)
    args = parser.parse_args()

    model, _ = build_model()
    path2meta = load_val_meta(data_dir=args.data_dir)

    imgs = iter_images(args.data_dir)
    print(f"\nProcessing {len(imgs)} images...\n")
    t0 = datetime.datetime.now()

    ensure_dir(args.out_root)
    for p in imgs:
        if args.no_core:
            # minimal pred class selection
            x = tfm(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                pred_idx = int(_predict_logits_tensor(model, x).argmax(1).item())
            img_dir = ensure_dir(os.path.join(args.out_root, os.path.splitext(os.path.basename(p))[0] + "_unknown", "images"))
            stem = os.path.splitext(os.path.basename(p))[0]
        else:
            info = run_core_outputs(model, p, path2meta, args.out_root)
            pred_idx, img_dir, stem = info["pred_idx"], info["img_dir"], info["stem"]

        out_img = run_lime(model, p, target_idx=pred_idx, num_samples=args.lime_samples,
                           seg=args.seg, kernel_width=args.kernel_width)
        save_to = os.path.join(img_dir, f"{stem}_lime_new.jpg")
        cv2.imwrite(save_to, out_img)
        print("LIME saved to", save_to)

    print("Runtime:", datetime.datetime.now() - t0)

if __name__ == "__main__":
    main()
