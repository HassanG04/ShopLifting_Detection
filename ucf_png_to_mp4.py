"""
ucf_png_to_mp4.py  (v2 — handles flat UCF layout)
==================================================
Converts UCF Shoplifting PNG sequences into one MP4 per clip.

The UCF dataset comes in TWO possible layouts — this script handles both:

  Layout A — flat (most common download):
    Test/NormalVideos/Normal001_001.png
    Test/NormalVideos/Normal001_002.png
    Test/NormalVideos/Normal002_001.png   <- different clip, same folder
    Test/Shoplifting/Shoplifting001_001.png
    ...
    Groups by the clip-ID prefix (everything before the last _framenum).
    Writes: Test/NormalVideos/Normal001.mp4, Normal002.mp4, ...

  Layout B — nested (one subfolder per clip):
    Test/NormalVideos/Normal001/001.png
    Test/NormalVideos/Normal001/002.png
    Test/NormalVideos/Normal002/001.png
    ...
    Each leaf subfolder becomes one MP4 placed next to it.

Usage:
    python ucf_png_to_mp4.py
    python ucf_png_to_mp4.py --root different_data/UCF_Shoplifting --fps 10 --dry-run
    python ucf_png_to_mp4.py --root different_data/UCF_Shoplifting --overwrite

Options:
    --root      Root of the UCF dataset (default: different_data/UCF_Shoplifting)
    --fps       Output frame rate (default: 10)
    --size      Output WxH (default: 320x240)
    --overwrite Re-encode even if the .mp4 already exists
    --dry-run   Print what would be done without writing anything
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sort_key(p: Path) -> int:
    """Sort a PNG by the last integer in its filename stem (= frame number)."""
    digits = re.findall(r'\d+', p.stem)
    return int(digits[-1]) if digits else 0


def clip_id_from_flat_name(stem: str) -> str:
    """
    Extract the clip ID from a flat UCF filename.
    'Normal001_067'      -> 'Normal001'
    'Shoplifting003_012' -> 'Shoplifting003'
    'frame001'           -> 'frame001'   (no underscore -> whole stem)
    """
    m = re.match(r'^(.+?)_(\d+)$', stem)
    return m.group(1) if m else stem


def write_mp4(frames: list, out_path: Path,
              fps: float, size: tuple, dry_run: bool) -> bool:
    """Encode a sorted list of PNG paths to one MP4. Returns True on success."""
    if not frames:
        return False
    if dry_run:
        print(f"  [DRY]   {out_path.name}  ({len(frames)} frames)")
        return True

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, size)
    if not writer.isOpened():
        print(f"  [ERR]   Cannot open VideoWriter -> {out_path}")
        return False

    written = 0
    for png in sorted(frames, key=sort_key):
        img = cv2.imread(str(png))
        if img is None:
            continue
        writer.write(cv2.resize(img, size, interpolation=cv2.INTER_LINEAR))
        written += 1

    writer.release()
    if written == 0:
        out_path.unlink(missing_ok=True)
        print(f"  [ERR]   No readable frames -> {out_path.name}")
        return False

    mb = out_path.stat().st_size / 1024 ** 2
    print(f"  [OK]    {out_path.name}  ({written} frames, {mb:.1f} MB)")
    return True


# ---------------------------------------------------------------------------
# Layout detection
# ---------------------------------------------------------------------------

def find_jobs(root: Path, overwrite: bool, dry_run: bool) -> list:
    """
    Walk root, detect flat vs nested layout per PNG-containing folder,
    and return a list of (png_list, out_mp4_path) conversion jobs.
    """
    jobs = []

    for folder in sorted(root.rglob("*")):
        if not folder.is_dir():
            continue

        direct_pngs = list(folder.glob("*.png"))
        if not direct_pngs:
            continue

        # Group PNGs by clip-ID prefix
        groups = defaultdict(list)
        for png in direct_pngs:
            groups[clip_id_from_flat_name(png.stem)].append(png)

        if len(groups) == 1:
            # ── Layout B: folder itself is one clip ──────────────────────
            out_mp4 = folder.parent / f"{folder.name}.mp4"
            if out_mp4.exists() and not overwrite and not dry_run:
                print(f"  [SKIP]  {out_mp4.name}  (exists; use --overwrite)")
                continue
            jobs.append((direct_pngs, out_mp4))

        else:
            # ── Layout A: flat folder, each prefix group is one clip ─────
            for clip_id, pngs in sorted(groups.items()):
                out_mp4 = folder / f"{clip_id}.mp4"
                if out_mp4.exists() and not overwrite and not dry_run:
                    print(f"  [SKIP]  {out_mp4.name}  (exists; use --overwrite)")
                    continue
                jobs.append((pngs, out_mp4))

    return jobs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert UCF Shoplifting PNGs to MP4 (flat or nested layout).")
    parser.add_argument("--root",      default="different_data/UCF_Shoplifting",
                        help="Root of the UCF dataset tree.")
    parser.add_argument("--fps",       type=float, default=10.0,
                        help="Output FPS (default 10).")
    parser.add_argument("--size",      default="320x240",
                        help="Output resolution WxH (default 320x240).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-encode even if the MP4 already exists.")
    parser.add_argument("--dry-run",   action="store_true",
                        help="Show what would happen without writing files.")
    args = parser.parse_args()

    try:
        w, h = [int(x) for x in args.size.lower().split("x")]
        size = (w, h)
    except ValueError:
        print(f"ERROR: --size must be WxH, e.g. 320x240"); sys.exit(1)

    root = Path(args.root)
    if not root.exists():
        print(f"ERROR: {root.resolve()} does not exist.")
        print("Run this script from your project root directory.")
        sys.exit(1)

    print(f"\nScanning : {root.resolve()}")
    print(f"Settings : fps={args.fps}  size={size[0]}x{size[1]}  "
          f"overwrite={args.overwrite}  dry-run={args.dry_run}\n")

    # Diagnostic: show what folders contain PNGs and which layout
    png_folders = [d for d in sorted(root.rglob("*"))
                   if d.is_dir() and list(d.glob("*.png"))]

    if not png_folders:
        print("No PNG files found. Nothing to do.")
        sys.exit(0)

    print("PNG folders found:")
    for d in png_folders:
        pngs = list(d.glob("*.png"))
        groups = set(clip_id_from_flat_name(p.stem) for p in pngs)
        layout = "nested-leaf (1 clip)" if len(groups) <= 1 else f"flat ({len(groups)} clips)"
        # Show a sample filename so you can verify grouping looks right
        sample = sorted(pngs, key=sort_key)[:1]
        sample_str = sample[0].name if sample else ""
        print(f"  {d.relative_to(root)}  [{len(pngs)} PNGs, {layout}]  e.g. {sample_str}")
    print()

    jobs = find_jobs(root, args.overwrite, args.dry_run)

    if not jobs:
        print("No jobs to run. All MP4s already exist — use --overwrite to redo.")
        sys.exit(0)

    print(f"{'[DRY RUN] ' if args.dry_run else ''}Jobs to convert: {len(jobs)}\n")

    ok = fail = 0
    for pngs, out_mp4 in jobs:
        success = write_mp4(pngs, out_mp4, fps=args.fps, size=size,
                            dry_run=args.dry_run)
        ok   += success
        fail += not success

    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Done.")
    print(f"  Converted : {ok}")
    print(f"  Failed    : {fail}")
    print()
    print("Next: run the notebook. collect_from_source() finds MP4s via rglob('*.mp4').")


if __name__ == "__main__":
    main()
