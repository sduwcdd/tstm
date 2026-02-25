'''
See Appendix Experiment: Pre-training using the Davis dataset, with the current file constructing an applicable dataset.
'''
import argparse
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _bbox_from_binary_mask(mask_hw: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask_hw > 0)
    if xs.size == 0 or ys.size == 0:
        return 0, 0, int(mask_hw.shape[1]), int(mask_hw.shape[0])
    x0 = int(xs.min())
    x1 = int(xs.max()) + 1
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1
    return x0, y0, x1, y1


def _crop_square_with_pad(img: np.ndarray, cx: float, cy: float, side: int, pad_value) -> np.ndarray:
    h, w = img.shape[:2]
    side = max(1, int(side))
    x0 = int(round(cx - side / 2.0))
    y0 = int(round(cy - side / 2.0))
    x1 = x0 + side
    y1 = y0 + side

    src_x0 = max(0, x0)
    src_y0 = max(0, y0)
    src_x1 = min(w, x1)
    src_y1 = min(h, y1)

    dst_x0 = src_x0 - x0
    dst_y0 = src_y0 - y0
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    dst_y1 = dst_y0 + (src_y1 - src_y0)

    if img.ndim == 2:
        out = np.full((side, side), pad_value, dtype=img.dtype)
        out[dst_y0:dst_y1, dst_x0:dst_x1] = img[src_y0:src_y1, src_x0:src_x1]
        return out

    out = np.full((side, side, img.shape[2]), pad_value, dtype=img.dtype)
    out[dst_y0:dst_y1, dst_x0:dst_x1, :] = img[src_y0:src_y1, src_x0:src_x1, :]
    return out


def _crop_square_clamped(img: np.ndarray, cx: float, cy: float, side: int) -> np.ndarray:
    h, w = img.shape[:2]
    side = max(1, int(side))
    side = min(side, int(h), int(w))
    x0 = int(round(cx - side / 2.0))
    y0 = int(round(cy - side / 2.0))

    x0 = max(0, min(int(w) - side, x0))
    y0 = max(0, min(int(h) - side, y0))
    x1 = x0 + side
    y1 = y0 + side
    return img[y0:y1, x0:x1].copy()


def _pad_to_square(img: np.ndarray, target_size: int, pad_value) -> np.ndarray:
    h, w = img.shape[:2]
    if h == target_size and w == target_size:
        return img
    top = (target_size - h) // 2
    bottom = target_size - h - top
    left = (target_size - w) // 2
    right = target_size - w - left
    if img.ndim == 2:
        return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=int(pad_value))
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_value)


def _load_rgb(path: Path) -> np.ndarray:
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Failed to load image: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def _load_ann(path: Path, shape_hw: Tuple[int, int]) -> np.ndarray:
    if not path.exists():
        return np.zeros(shape_hw, dtype=np.uint8)
    ann = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if ann is None:
        return np.zeros(shape_hw, dtype=np.uint8)
    if ann.shape[:2] != shape_hw:
        ann = cv2.resize(ann, (int(shape_hw[1]), int(shape_hw[0])), interpolation=cv2.INTER_NEAREST)
    return ann


def _copy_imagesets(src_root: Path, dst_root: Path, year: str):
    src = src_root / "ImageSets" / year
    dst = dst_root / "ImageSets" / year
    _ensure_dir(dst)
    for name in ["train.txt", "val.txt", "test.txt"]:
        p = src / name
        if p.exists():
            (dst / name).write_text(p.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_root", type=str, required=True)
    parser.add_argument("--dst_root", type=str, required=True)
    parser.add_argument("--year", type=str, default="2016", choices=["2016", "2017"])
    parser.add_argument("--in_resolution", type=str, default="480p")
    parser.add_argument("--out_resolution", type=str, default="84p")
    parser.add_argument("--target_size", type=int, default=84)
    parser.add_argument("--crop_margin", type=float, default=0.15)
    parser.add_argument("--bbox_mode", type=str, default="per_frame", choices=["union", "per_frame"])
    parser.add_argument("--small_behavior", type=str, default="resize", choices=["pad", "resize"])
    parser.add_argument("--save_ext", type=str, default="png", choices=["png", "jpg"])
    parser.add_argument("--jpg_quality", type=int, default=95)
    args = parser.parse_args()

    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)

    sets_dir = src_root / "ImageSets" / str(args.year)
    train_set = sets_dir / "train.txt"
    val_set = sets_dir / "val.txt"
    if not train_set.exists() or not val_set.exists():
        raise FileNotFoundError(f"Missing ImageSets files under: {sets_dir}")

    sequences = set()
    for p in [train_set, val_set]:
        for ln in p.read_text().splitlines():
            ln = ln.strip()
            if ln:
                sequences.add(ln)

    _copy_imagesets(src_root, dst_root, str(args.year))

    for seq in sorted(list(sequences)):
        frames_dir = src_root / "JPEGImages" / str(args.in_resolution) / seq
        ann_dir = src_root / "Annotations" / str(args.in_resolution) / seq
        frame_files = sorted(list(frames_dir.glob("*.jpg")))
        if not frame_files:
            frame_files = sorted(list(frames_dir.glob("*.png")))
        if not frame_files:
            continue

        ann_files = [ann_dir / f"{p.stem}.png" for p in frame_files]

        imgs = [_load_rgb(p) for p in frame_files]
        h0, w0 = imgs[0].shape[:2]
        anns = [_load_ann(p, (h0, w0)) for p in ann_files]

        union = np.zeros((h0, w0), dtype=np.uint8)
        for ann in anns:
            union |= (ann > 0).astype(np.uint8)

        x0, y0, x1, y1 = _bbox_from_binary_mask(union)
        bw = max(1, int(x1 - x0))
        bh = max(1, int(y1 - y0))
        side = int(max(bw, bh))
        margin_px = int(round(float(args.crop_margin) * float(side)))
        side = int(side + 2 * margin_px)
        cx = (float(x0) + float(x1)) * 0.5
        cy = (float(y0) + float(y1)) * 0.5

        out_frames_dir = dst_root / "JPEGImages" / str(args.out_resolution) / seq
        out_anns_dir = dst_root / "Annotations" / str(args.out_resolution) / seq
        _ensure_dir(out_frames_dir)
        _ensure_dir(out_anns_dir)

        for img_rgb, ann, fp in zip(imgs, anns, frame_files):
            if str(args.bbox_mode) == "per_frame":
                fx0, fy0, fx1, fy1 = _bbox_from_binary_mask((ann > 0).astype(np.uint8))
                if (fx1 - fx0) <= 1 or (fy1 - fy0) <= 1:
                    fx0, fy0, fx1, fy1 = x0, y0, x1, y1
                fbw = max(1, int(fx1 - fx0))
                fbh = max(1, int(fy1 - fy0))
                fside = int(max(fbw, fbh))
                fmargin_px = int(round(float(args.crop_margin) * float(fside)))
                fside = int(fside + 2 * fmargin_px)
                fcx = (float(fx0) + float(fx1)) * 0.5
                fcy = (float(fy0) + float(fy1)) * 0.5
            else:
                fside, fcx, fcy = side, cx, cy

            cropped_img = _crop_square_clamped(img_rgb, cx=fcx, cy=fcy, side=fside)
            cropped_ann = _crop_square_clamped(ann, cx=fcx, cy=fcy, side=fside)

            target = int(args.target_size)
            cur_side = int(cropped_img.shape[0])
            if cur_side != target:
                if cur_side < target and str(args.small_behavior) == "pad":
                    cropped_img = _pad_to_square(cropped_img, target, pad_value=(0, 0, 0))
                    cropped_ann = _pad_to_square(cropped_ann, target, pad_value=0)
                else:
                    interp_img = cv2.INTER_AREA if cur_side > target else cv2.INTER_CUBIC
                    cropped_img = cv2.resize(cropped_img, (target, target), interpolation=interp_img)
                    cropped_ann = cv2.resize(cropped_ann, (target, target), interpolation=cv2.INTER_NEAREST)

            out_img_path = out_frames_dir / f"{fp.stem}.{str(args.save_ext)}"
            out_ann_path = out_anns_dir / f"{fp.stem}.png"

            if str(args.save_ext) == "jpg":
                cv2.imwrite(
                    str(out_img_path),
                    cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR),
                    [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpg_quality)],
                )
            else:
                cv2.imwrite(str(out_img_path), cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR))
            ann_u8 = ((cropped_ann > 0).astype(np.uint8) * 255)
            cv2.imwrite(str(out_ann_path), ann_u8)

        print(f"Processed: {seq}")

    print(f"Done. Output dataset root: {dst_root}")


if __name__ == "__main__":
    main()
