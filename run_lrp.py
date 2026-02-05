

import os
import datetime
import copy
import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn

from xai_core import (
    add_common_cli, build_model, load_val_meta, ensure_dir, iter_images,
    run_core_outputs, tfm, device, _predict_logits_tensor
)

# Captum LRP (version-robust)
try:
    from captum.attr import LRP
    from captum.attr._utils.lrp_rules import EpsilonRule
    try:
        from captum.attr._utils.lrp_rules import GammaRule
    except Exception:
        GammaRule = None
    try:
        from captum.attr._utils.lrp_rules import IdentityRule
    except Exception:
        IdentityRule = None
except Exception:
    LRP = None
    EpsilonRule = None
    GammaRule = None
    IdentityRule = None


class _MultiCallModule(nn.Module):
    """
    Wrap a module so repeated calls within one forward use distinct copies.
    Needed for Captum LRP when a module instance is reused multiple times.
    """
    def __init__(self, module: nn.Module, max_calls: int = 256):
        super().__init__()
        self.mods = nn.ModuleList([copy.deepcopy(module) for _ in range(int(max_calls))])
        self._call_idx = 0

    def reset_calls(self):
        self._call_idx = 0

    def forward(self, *args, **kwargs):
        i = self._call_idx
        if i >= len(self.mods):
            raise RuntimeError(
                f"_MultiCallModule exceeded max_calls={len(self.mods)} for {self.mods[0].__class__.__name__}."
            )
        self._call_idx += 1
        return self.mods[i](*args, **kwargs)

def _reset_multicall_wrappers(root: nn.Module):
    for m in root.modules():
        if hasattr(m, "reset_calls") and callable(m.reset_calls):
            m.reset_calls()

def _wrap_by_dotted_name(root: nn.Module, dotted: str, max_calls: int = 256):
    parts = dotted.split(".")
    parent = root
    for p in parts[:-1]:
        if not hasattr(parent, p):
            raise AttributeError(f"Path not found while resolving '{dotted}': '{p}'")
        parent = getattr(parent, p)
    name = parts[-1]
    if not hasattr(parent, name):
        raise AttributeError(f"Path not found while resolving '{dotted}': '{name}'")
    child = getattr(parent, name)
    if not isinstance(child, nn.Module):
        raise TypeError(f"Resolved object is not nn.Module for '{dotted}'")
    setattr(parent, name, _MultiCallModule(child, max_calls=int(max_calls)))

def _attach_rules(net: nn.Module, rule_name: str, eps_: float, gamma_: float):
    if EpsilonRule is None:
        return net
    rule_name = (rule_name or "epsilon").lower()

    def linear_rule():
        if rule_name == "gamma" and (GammaRule is not None):
            return GammaRule(gamma=float(gamma_))
        return EpsilonRule(epsilon=float(eps_))

    def passthrough_rule():
        if IdentityRule is not None:
            return IdentityRule()
        return EpsilonRule(epsilon=float(eps_))

    passthrough_types = (
        nn.Identity, nn.Dropout, nn.Flatten, nn.LayerNorm,
        nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
        nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
        nn.ReLU, nn.GELU, nn.SiLU, nn.ELU, nn.LeakyReLU, nn.Tanh, nn.Sigmoid,
        _MultiCallModule,
    )

    for m in net.modules():
        if m.__class__.__name__ in ("DropPath", "StochasticDepth"):
            m.rule = passthrough_rule()
            continue
        if isinstance(m, passthrough_types):
            m.rule = passthrough_rule()
            continue
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            m.rule = linear_rule()
            continue
    return net

def _pool_to_patches(m2d: np.ndarray, patch: int | None) -> np.ndarray:
    if patch is None or patch <= 0:
        return m2d
    H, W = m2d.shape
    Hc, Wc = (H // patch) * patch, (W // patch) * patch
    if Hc <= 0 or Wc <= 0:
        return m2d
    m2d = m2d[:Hc, :Wc]
    blocks = m2d.reshape(Hc // patch, patch, Wc // patch, patch).transpose(0, 2, 1, 3)
    return blocks.mean(axis=(2, 3))

def _robust_norm01(m2d: np.ndarray, lo=80.0, hi=99.5, gamma=0.85, eps=1e-8) -> np.ndarray:
    lo_v = np.percentile(m2d, lo)
    hi_v = np.percentile(m2d, hi)
    if hi_v <= lo_v + eps:
        return np.zeros_like(m2d)
    m = np.clip(m2d, lo_v, hi_v)
    m = (m - lo_v) / (hi_v - lo_v + eps)
    if gamma != 1.0:
        m = np.power(m, gamma)
    return m

def run_lrp(
    model,
    img_path,
    target_idx=None,
    rule="epsilon",
    eps=1e-6,
    gamma=0.25,
    patch=16,
    p_low=80.0,
    p_high=99.5,
    gamma_corr=0.85,
    colormap=cv2.COLORMAP_VIRIDIS,
    use_amp=True,
    wrap_paths=None,
    wrap_calls=1024,
):
    if LRP is None or EpsilonRule is None:
        raise RuntimeError("Captum LRP is not available. Install/upgrade captum.")

    model.eval()
    dev = next(model.parameters()).device

    pil = Image.open(img_path).convert("RGB")
    x = tfm(pil).unsqueeze(0).to(dev)

    with torch.no_grad():
        logits = _predict_logits_tensor(model, x)
        pred_idx = int(logits.argmax(1).item())
    target_idx = pred_idx if target_idx is None else int(target_idx)

    class _LogitsOnly(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, inp):
            return _predict_logits_tensor(self.m, inp)

    wrapped = _LogitsOnly(model).to(dev).eval()
    wrapped_lrp = copy.deepcopy(wrapped).to(dev).eval()

    if wrap_paths:
        for pth in wrap_paths:
            _wrap_by_dotted_name(wrapped_lrp, pth, max_calls=int(wrap_calls))

    wrapped_lrp = _attach_rules(wrapped_lrp, rule, eps, gamma)

    _reset_multicall_wrappers(wrapped_lrp)
    explainer = LRP(wrapped_lrp)

    _reset_multicall_wrappers(wrapped_lrp)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    with torch.cuda.amp.autocast(enabled=bool(use_amp)):
        attr = explainer.attribute(x, target=int(target_idx))  # [1,3,H,W], signed

    rel = attr.sum(dim=1, keepdim=False)[0].detach().float().cpu().numpy()
    rel = np.maximum(rel, 0.0)
    rel = _pool_to_patches(rel, None if patch == 0 else patch)
    heat01 = _robust_norm01(rel, lo=p_low, hi=p_high, gamma=gamma_corr)

    raw = cv2.imread(img_path)
    heat_u8 = (cv2.resize(heat01, (raw.shape[1], raw.shape[0]), interpolation=cv2.INTER_NEAREST) * 255).astype(np.uint8)
    cm = cv2.applyColorMap(heat_u8, colormap)
    out = cv2.addWeighted(raw, 0.55, cm, 0.45, 0)
    return out, target_idx

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run LRP only (Captum).")
    add_common_cli(parser)

    parser.add_argument('--lrp_rule', choices=['epsilon', 'gamma'], default='epsilon')
    parser.add_argument('--lrp_eps', type=float, default=1e-6)
    parser.add_argument('--lrp_gamma', type=float, default=0.25)

    parser.add_argument('--patch', type=int, default=16, help="Patch pooling size; set 0 to disable.")
    parser.add_argument('--p_low', type=float, default=80.0)
    parser.add_argument('--p_high', type=float, default=99.5)
    parser.add_argument('--gamma', type=float, default=0.85)

    parser.add_argument('--no_amp', action='store_true', help="Disable autocast AMP for LRP.")
    parser.add_argument('--wrap_calls', type=int, default=1024)

    # Default matches your earlier usage; users can override.
    parser.add_argument('--wrap_paths', nargs='*', default=["head.relu", "head.sigmoid", "head.fc"],
                        help="Dotted module paths to wrap for repeated calls (LRP requirement).")
    args = parser.parse_args()

    model, _ = build_model()
    path2meta = load_val_meta(data_dir=args.data_dir)

    imgs = iter_images(args.data_dir)
    print(f"\nProcessing {len(imgs)} images...\n")
    t0 = datetime.datetime.now()

    ensure_dir(args.out_root)
    for p in imgs:
        if args.no_core:
            x = tfm(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                pred_idx = int(_predict_logits_tensor(model, x).argmax(1).item())
            img_dir = ensure_dir(os.path.join(args.out_root, os.path.splitext(os.path.basename(p))[0] + "_unknown", "images"))
            stem = os.path.splitext(os.path.basename(p))[0]
        else:
            info = run_core_outputs(model, p, path2meta, args.out_root)
            pred_idx, img_dir, stem = info["pred_idx"], info["img_dir"], info["stem"]

        out, _ = run_lrp(
            model, p, target_idx=pred_idx,
            rule=args.lrp_rule,
            eps=args.lrp_eps,
            gamma=args.lrp_gamma,
            patch=args.patch,
            p_low=args.p_low, p_high=args.p_high, gamma_corr=args.gamma,
            use_amp=(not args.no_amp),
            wrap_paths=args.wrap_paths,
            wrap_calls=args.wrap_calls,
        )
        save_to = os.path.join(img_dir, f"{stem}_lrp.jpg")
        cv2.imwrite(save_to, out)
        print("Saved:", save_to)

    print("Done. Runtime:", datetime.datetime.now() - t0)

if __name__ == "__main__":
    main()
