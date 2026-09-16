"""Independent spot-check of the IPN Hand conversion.

Deliberately does NOT import the adapter. It re-reads the raw IPN files and the canonical
output separately and compares them, so a bug in the adapter cannot hide by being used on
both sides of the comparison.
"""
import pandas as pd

RAW = "raw/IPN_Hand/annotations"
OUT = "data/ipn_hand"

raw = pd.read_csv(f"{RAW}/Annot_List.txt")
meta = pd.read_csv(f"{RAW}/metadata.csv")
out = pd.read_csv(f"{OUT}/annotations.csv")
classes = open(f"{OUT}/classes.txt").read().split()

print("=" * 70)
print("1. classes.txt -- index 0 must be 'none'")
print("=" * 70)
for i, c in enumerate(classes):
    print(f"  {i:>2}  {c}")

print()
print("=" * 70)
print("2. One video, raw vs converted. Look for the 1-frame shift.")
print("=" * 70)
vid = "1CM1_1_R_#217"
print(f"RAW ({vid}), IPN's 1-indexed t_start/t_end:")
print(raw[raw.video == vid].head(6).to_string(index=False))
print()
print("CONVERTED, 0-indexed. start_frame must be exactly t_start-1:")
print(out[out.video_id == vid].head(6).to_string(index=False))

print()
print("=" * 70)
print("3. Subjects. Expect 50, each owning 4 videos.")
print("=" * 70)
per = out.groupby("subject_id").video_id.nunique()
print(f"  subjects: {len(per)}   videos each: {sorted(per.unique())}")
print(f"  first 8:  {sorted(per.index)[:8]}")
print()
print("  Same subject TOKEN, different camera -> must be different subjects:")
token1 = out[out.video_id.str.split("_").str[1] == "1"]
for v in sorted(token1.video_id.unique())[::4]:
    s = token1[token1.video_id == v].subject_id.iloc[0]
    print(f"    {v:<18} -> {s}")
print(f"    ...{token1.subject_id.nunique()} distinct people share the token '1'")

print()
print("=" * 70)
print("4. Tiling. Every video covered [0, T-1] with no gap and no overlap.")
print("=" * 70)
bad = []
for v, r in out.groupby("video_id"):
    r = r.sort_values("start_frame")
    s, e = r.start_frame.values, r.end_frame.values
    if s[0] != 0 or (s[1:] - e[:-1] - 1).any():
        bad.append(v)
print(f"  videos checked: {out.video_id.nunique()}   badly tiled: {len(bad)}")

print()
print("=" * 70)
print("5. Frame count: annotations vs metadata.csv. 14 videos differ by 1.")
print("=" * 70)
conv = out.groupby("video_id").end_frame.max() + 1
cmp = conv.rename("converted").to_frame().join(
    meta.set_index("Video Name").Frames.rename("metadata_csv")
)
diff = cmp[cmp.converted != cmp.metadata_csv]
print(f"  total frames, converted:   {int(cmp.converted.sum())}")
print(f"  total frames, metadata:    {int(cmp.metadata_csv.sum())}")
print(f"  videos that disagree:      {len(diff)} (all by exactly "
      f"{sorted((diff.metadata_csv - diff.converted).unique())})")
print(diff.head(4).to_string())

print()
print("=" * 70)
print("6. Instance counts, recomputed from the RAW file, not from the adapter.")
print("=" * 70)
print(f"  gestures  (label != D0X):  {int((raw.label != 'D0X').sum())}   published 4218")
print(f"  none      (label == D0X):  {int((raw.label == 'D0X').sum())}   published 1431")
print(f"  videos:                    {raw.video.nunique()}   published 200")
print()
print("  Converted rows per class vs raw rows per code:")
codes = pd.read_csv(f"{RAW}/classIdx.txt").label.tolist()
name_of = dict(zip(codes, ["none"] + classes[1:]))
lhs = raw.label.map(name_of).value_counts()
rhs = out["class"].value_counts()
merged = lhs.rename("raw").to_frame().join(rhs.rename("converted"))
merged["match"] = merged.raw == merged.converted
print(merged.sort_index().to_string())
