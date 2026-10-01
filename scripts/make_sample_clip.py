"""Render a labelled sample clip from the dataset, as an animated GIF and a contact sheet.

    python scripts/make_sample_clip.py --auto
    python scripts/make_sample_clip.py --video "1CM1_1_R_#217" --start 0 --end 600

Draws the ground-truth label onto each frame. That is the point: watching the label change
at the moment the hand starts moving is a direct, visual check that the annotations line up
with the pixels -- which is the one thing the numeric verification in Stage 1 cannot show.
An off-by-one in the frame indexing would be visible here as a label that switches slightly
too early or too late.

Needs only Pillow, which already decodes the frames; no video codec, no ffmpeg. A GIF opens
on any machine and drops into slides, which matters more for a meeting than fidelity.

This also seeds the Stage 8 failure-mode montage, which renders the worst false positives as
actual frames -- the same job with a different frame selection.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import canonical as C  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

log = get_logger("sample-clip")

#: Colours for the label bar: grey while nothing is happening, so a gesture stands out.
NONE_COLOUR = (70, 74, 82)
GESTURE_COLOUR = (28, 92, 168)


def load_font(size: int):
    from PIL import ImageFont

    for name in ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def pick_segment(dataset: C.CanonicalDataset, video_id: str, max_frames: int):
    """Choose a window containing as many distinct gestures as possible.

    A window of pure `none` is useless for showing a supervisor, and so is one that starts
    mid-gesture: the interesting moment is the transition into a gesture.
    """
    rows = dataset.annotations
    rows = rows[rows["video_id"] == video_id].sort_values("start_frame")
    spans = list(zip(rows["start_frame"], rows["end_frame"], rows["class"]))

    best = (0, max_frames, 0)
    for index, (start, _, _) in enumerate(spans):
        # Begin a little before the span so the transition into it is visible.
        begin = max(0, int(start) - 20)
        end = begin + max_frames
        distinct = len({c for s, e, c in spans if int(s) < end and int(e) >= begin})
        if distinct > best[2]:
            best = (begin, end, distinct)
    return best[0], min(best[1], dataset.num_frames(video_id))


def render(
    dataset: C.CanonicalDataset,
    frames_root: Path,
    video_id: str,
    start: int,
    end: int,
    out_dir: Path,
    *,
    every: int,
    width: int,
    ms: int,
) -> tuple[Path, Path]:
    from PIL import Image, ImageDraw

    labels = dataset.frame_labels(video_id)
    classes = dataset.classes
    folder = frames_root / video_id
    font = load_font(max(13, width // 28))
    small = load_font(max(10, width // 40))

    indices = list(range(start, min(end, dataset.num_frames(video_id)), every))
    if not indices:
        raise SystemExit("empty frame range")

    rendered: list = []
    for index in indices:
        path = folder / f"{video_id}_{index + 1:06d}.jpg"
        if not path.is_file():
            raise SystemExit(f"missing frame file: {path}")
        with Image.open(path) as image:
            frame = image.convert("RGB")

        height = round(frame.height * width / frame.width)
        frame = frame.resize((width, height), Image.BILINEAR)

        bar = max(26, width // 14)
        canvas = Image.new("RGB", (width, height + bar), (18, 18, 18))
        canvas.paste(frame, (0, 0))

        name = classes[int(labels[index])]
        is_none = name == C.NONE_CLASS
        draw = ImageDraw.Draw(canvas)
        draw.rectangle(
            [0, height, width, height + bar],
            fill=NONE_COLOUR if is_none else GESTURE_COLOUR,
        )
        draw.text((7, height + bar // 2), name.replace("_", " "), font=font, anchor="lm")
        draw.text(
            (width - 7, height + bar // 2),
            f"frame {index}",
            font=small,
            anchor="rm",
        )

        # A timeline strip across the top: the whole window at a glance, so the viewer can
        # see where in the sequence this frame sits and how long each span lasts.
        strip = 5
        for x in range(width):
            frame_at = start + round(x * (indices[-1] - start) / max(width - 1, 1))
            label_at = classes[int(labels[min(frame_at, len(labels) - 1)])]
            colour = NONE_COLOUR if label_at == C.NONE_CLASS else GESTURE_COLOUR
            draw.line([(x, 0), (x, strip)], fill=colour)
        cursor = round((index - start) / max(indices[-1] - start, 1) * (width - 1))
        draw.line([(cursor, 0), (cursor, strip + 4)], fill=(255, 255, 255), width=2)

        rendered.append(canvas)

    out_dir.mkdir(parents=True, exist_ok=True)
    safe = video_id.replace("#", "")
    gif = out_dir / f"{safe}_{start}-{indices[-1]}.gif"
    rendered[0].save(
        gif, save_all=True, append_images=rendered[1:], duration=ms, loop=0, optimize=True
    )

    # A contact sheet for slides and for print, where a GIF cannot animate.
    columns = 5
    picks = rendered[:: max(1, len(rendered) // 10)][:10]
    rows = -(-len(picks) // columns)
    tile_w, tile_h = picks[0].size
    sheet = Image.new("RGB", (columns * tile_w, rows * tile_h), (18, 18, 18))
    for position, tile in enumerate(picks):
        sheet.paste(tile, ((position % columns) * tile_w, (position // columns) * tile_h))
    png = out_dir / f"{safe}_{start}-{indices[-1]}_sheet.png"
    sheet.save(png)

    return gif, png


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/ipn_hand"))
    parser.add_argument("--frames-root", type=Path, default=Path("raw/IPN_Hand/frames"))
    parser.add_argument("--out", type=Path, default=Path("reports/samples"))
    parser.add_argument("--video", default=None, help="video id; omit with --auto")
    parser.add_argument("--auto", action="store_true", help="pick an interesting segment")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=450, help="window length in frames")
    parser.add_argument("--every", type=int, default=3, help="keep every Nth frame")
    parser.add_argument("--width", type=int, default=420)
    parser.add_argument("--ms", type=int, default=100, help="milliseconds per GIF frame")
    args = parser.parse_args()

    if not args.frames_root.is_dir():
        log.error("no frames at %s -- unpack them first", args.frames_root)
        return 1

    dataset = C.CanonicalDataset(args.root)
    video_id = args.video or dataset.video_ids[0]
    if video_id not in dataset.video_ids:
        log.error("unknown video %r", video_id)
        return 1

    if args.auto or args.end is None:
        start, end = pick_segment(dataset, video_id, args.max_frames)
    else:
        start, end = args.start, args.end

    log.info("%s frames %d-%d", video_id, start, end)
    gif, png = render(
        dataset,
        args.frames_root,
        video_id,
        start,
        end,
        args.out,
        every=args.every,
        width=args.width,
        ms=args.ms,
    )

    labels = dataset.frame_labels(video_id)[start:end]
    shown = [dataset.classes[i] for i in sorted(set(labels.tolist()))]
    print()
    print(f"  gestures in this clip: {', '.join(s.replace('_', ' ') for s in shown)}")
    print(f"  GIF           {gif}  ({gif.stat().st_size / 1e6:.1f} MB)")
    print(f"  contact sheet {png}  ({png.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
