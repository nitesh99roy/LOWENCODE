#!/usr/bin/env python3
"""
split_encode.py - split ONE video into pieces, encode every piece on its own GitHub
runner at the same time, then join the pieces and add the complete audio back.

Used by .github/workflows/split-encode.yml. Encoding settings (CRF, bitrate caps,
audio bitrates, preset, crop detection, file naming) come from CPUEncoder.py, so a
split encode gives the same result as encoding the whole file in one go.

How the pieces stay seamless:
  * Pieces are cut by stream copy (fast, no quality loss) with a few seconds of
    extra video on each side, keeping the original timestamps.
  * Each encoder then keeps exactly the frames between its two boundaries (trim on
    the original timestamps). Boundaries sit half a frame between two frames, so
    every frame belongs to exactly one piece: no gaps, no duplicates, and it also
    works for open-GOP sources (HEVC / Blu-ray).
  * The crop is detected ONCE on the full file, so every piece has the same size.
  * Audio is encoded once from the full source (never split), so it has no gaps.

Checks (a file is only uploaded when all of them pass):
  * prepare reads the time of every video frame in the source. Timestamps are only
    rebuilt when every frame sits on an even grid; otherwise the originals are kept.
  * every encoder counts the frames of its piece against the plan before uploading.
  * the join places each piece at its exact planned time, then reads the finished file:
    frame count, the time of every frame, a keyframe at every join, a real decode around
    every join, audio start / end / gaps against the source, audio-video write order.

Commands:
  download <url>                             fetch the source into $WORK/src/
  prepare  <chunks> <formats> <crop> [name]  plan, cut + upload pieces, encode audio
  encode   <index>                           encode one piece into every resolution
  audio                                      encode the full-length audio (own runner, in
                                             parallel with the video pieces)
  merge                                      join pieces + audio/subs/chapters -> $WORK/out/
  publish                                    put the results in a draft release
  gofile                                     upload the results to one Gofile folder
  cleanup                                    delete the temporary draft release
"""
import fnmatch
import hashlib
import importlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor

# Settings module: CPUEncoder (x264, 3 resolutions) or LowEncoder (x265, small size)
CE = importlib.import_module(os.environ.get("ENCODER_MODULE", "CPUEncoder"))

WORK = os.environ.get("WORK") or os.path.abspath("work")
TAG = os.environ.get("SCRATCH_TAG", "")
# For testing without GitHub: a local folder used instead of the scratch release
STORE = os.environ.get("LOCAL_STORE", "")

TS_OFFSET = 1000.0                      # added to piece timestamps so they are never negative
EDGE = 5.0                              # extra seconds cut on each side of a piece
MIN_PIECE_SEC = float(os.environ.get("MIN_PIECE_SEC", "30"))
MAX_PIECES = 20                         # GitHub Free runs at most 20 jobs at once
MAX_ASSET = int(os.environ.get("MAX_ASSET_MB", "1800")) * 1024 * 1024   # release assets < 2 GB

if hasattr(CE, "apply_env"):
    CE.apply_env()
if hasattr(CE, "PRESET_OVERRIDE"):
    # Workflow "preset" input: "per-format" (default) keeps each format's own preset
    chosen = os.environ.get("X264_PRESET", "")
    CE.PRESET_OVERRIDE = "" if chosen in ("", "per-format") else chosen
if hasattr(CE, "X264_TUNE"):
    if "X264_TUNE" in os.environ:
        CE.X264_TUNE = "" if os.environ["X264_TUNE"] in ("", "none") else os.environ["X264_TUNE"]
    if "X264_ADVANCED" in os.environ:
        CE.X264_ADVANCED = os.environ["X264_ADVANCED"].lower() == "true"


# Audio languages to keep: AUDIO_LANGS="eng" or "hin,eng" or "all" (default).
# Matches the track's language tag, or its title ("English", "Hindi", ...).
LANG_NAMES = {"eng": "english", "hin": "hindi", "tam": "tamil", "tel": "telugu", "mal": "malayalam",
              "kan": "kannada", "ben": "bengali", "mar": "marathi", "guj": "gujarati",
              "pan": "punjabi", "urd": "urdu", "jpn": "japanese", "kor": "korean",
              "spa": "spanish", "fre": "french", "ger": "german", "chi": "chinese"}
LANG_SHORT = {"en": "eng", "hi": "hin", "ta": "tam", "te": "tel", "ml": "mal", "kn": "kan",
              "bn": "ben", "mr": "mar", "gu": "guj", "pa": "pan", "ur": "urd", "ja": "jpn",
              "ko": "kor", "es": "spa", "fr": "fre", "de": "ger", "zh": "chi"}


def wanted_languages():
    raw = os.environ.get("AUDIO_LANGS", "all").strip().lower()
    if raw in ("", "all", "*"):
        return None
    codes = set()
    full = {v: k for k, v in LANG_NAMES.items()}
    for item in (x.strip() for x in raw.split(",") if x.strip()):
        codes.add(LANG_SHORT.get(item) or full.get(item) or item)
    return codes


def select_audio(tracks):
    """Audio tracks to keep, based on AUDIO_LANGS. Never returns an empty list
    when the source has audio: if nothing matches, every track is kept."""
    want = wanted_languages()
    if not want or not tracks:
        return tracks
    picked = []
    for a in tracks:
        lang = (a.get("language") or "").lower()
        title = (a.get("title") or "").lower()
        if lang in want or any(LANG_NAMES.get(w, w) in title for w in want):
            picked.append(a)
    if not picked:
        found = ", ".join(f"{a.get('language') or '?'} ({a.get('title') or 'no title'})" for a in tracks)
        print(f"::warning::No audio track matches '{','.join(sorted(want))}' - keeping all. Tracks: {found}")
        return tracks
    return picked


if hasattr(CE, "SIZE_MODE") and os.environ.get("SIZE_MODE"):
    CE.SIZE_MODE = os.environ["SIZE_MODE"]


def vlabel(tag):
    """Filter-graph label for a format (labels can't contain dots, e.g. 720p.10BIT)."""
    return "v_" + re.sub(r"\W", "_", tag)


def pre_filters(plan):
    """Crop + anamorphic fix, applied before splitting into the formats."""
    pre = []
    if plan["crop"]:
        pre.append(plan["crop"]["ffmpeg_crop"])
    if abs(plan["sar"] - 1.0) > 0.01:
        pre.append("scale='trunc(iw*sar/2)*2':ih")
    return pre


def scale_chain(plan, t):
    """Resize (+ pixel format) for one format."""
    eff_w = (plan["crop"]["w"] if plan["crop"] else plan["info"]["width"]) * plan["sar"]
    eff_h = plan["crop"]["h"] if plan["crop"] else plan["info"]["height"]
    if hasattr(CE, "scale_filter"):
        return CE.scale_filter(t, eff_w, eff_h)
    bw, bh = CE.FORMATS[t]["box"]
    return (f"scale={bw}:{bh}:force_original_aspect_ratio=decrease:force_divisible_by=2"
            f":flags=lanczos,setsar=1,format=yuv420p")


def describe_settings():
    if hasattr(CE, "describe"):
        return CE.describe()
    return (f"presets: {CE.presets_text()} | tune: {CE.X264_TUNE or 'none'} | "
            f"advanced: {'on' if CE.X264_ADVANCED else 'off'} | size: {getattr(CE, 'SIZE_MODE', 'crf')}")


def output_name(base, tag):
    if hasattr(CE, "output_name"):
        return CE.output_name(base, tag)
    return CE.generate_output_filename(base, tag) + ".mkv"


# -------------------------------------------------------------------------------
# Small helpers
# -------------------------------------------------------------------------------
def die(msg):
    print(f"::error::{msg}", flush=True)
    sys.exit(1)


def run(cmd, check=True):
    print("$ " + " ".join(cmd), flush=True)
    rc = subprocess.run(cmd).returncode
    if check and rc != 0:
        die(f"{os.path.basename(cmd[0])} failed with exit code {rc}")
    return rc


def ffmpeg(args, label):
    """Run ffmpeg with a short progress line every 20 s (readable in GitHub logs)."""
    cmd = ["ffmpeg", "-nostdin", "-y", "-hide_banner", "-v", "warning", "-nostats",
           "-progress", "pipe:1", "-stats_period", "20"] + args
    print("$ " + " ".join(cmd), flush=True)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
    stats = {}
    for line in p.stdout:
        key, _, value = line.strip().partition("=")
        stats[key] = value
        if key == "progress":
            print(f"   [{label}] time={stats.get('out_time', '?')[:8]} fps={stats.get('fps', '?')} "
                  f"speed={stats.get('speed', '?')}", flush=True)
    rc = p.wait()
    if rc != 0:
        die(f"ffmpeg failed ({label}) with exit code {rc}")


def set_output(key, value):
    path = os.environ.get("GITHUB_OUTPUT")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"{key}={value}\n")
    print(f"output {key}={value}")


def ratio(text):
    try:
        a, b = str(text).split("/")
        return float(a) / float(b) if float(b) else 0.0
    except ValueError:
        return 0.0


def mb(path):
    return f"{os.path.getsize(path) / (1024 * 1024):.1f} MB"


# -------------------------------------------------------------------------------
# Storage between jobs: a temporary draft release (no artifact storage quota)
# -------------------------------------------------------------------------------
def store_upload(path):
    if STORE:
        os.makedirs(STORE, exist_ok=True)
        shutil.copy2(path, STORE)
        return
    for attempt in range(1, 5):
        if run(["gh", "release", "upload", TAG, path, "--clobber"], check=False) == 0:
            return
        print(f"upload failed, retry {attempt}/4 ...", flush=True)
        time.sleep(15 * attempt)
    die(f"could not upload {os.path.basename(path)}")


def store_download(patterns, dest):
    os.makedirs(dest, exist_ok=True)
    if STORE:
        names = [f for f in os.listdir(STORE) if any(fnmatch.fnmatch(f, p) for p in patterns)]
        if not names:
            die(f"nothing matches {patterns} in {STORE}")
        for n in names:
            shutil.copy2(os.path.join(STORE, n), dest)
        return
    cmd = ["gh", "release", "download", TAG, "-D", dest, "--clobber"]
    for p in patterns:
        cmd += ["-p", p]
    for attempt in range(1, 5):
        if run(cmd, check=False) == 0:
            return
        print(f"download failed, retry {attempt}/4 ...", flush=True)
        time.sleep(15 * attempt)
    die(f"could not download {patterns}")


def store_upload_big(path):
    """Upload a file that may be bigger than the 2 GB asset limit: split into
    <name>.part00, .part01 ... (byte split, rejoined exactly by store_download_big)."""
    if os.path.getsize(path) <= MAX_ASSET:
        store_upload(path)
        return
    n = 0
    with open(path, "rb") as src:
        while True:
            part = f"{path}.part{n:02d}"
            written = 0
            with open(part, "wb") as dst:
                while written < MAX_ASSET:
                    buf = src.read(min(64 << 20, MAX_ASSET - written))
                    if not buf:
                        break
                    dst.write(buf)
                    written += len(buf)
            if written == 0:
                os.remove(part)
                break
            store_upload(part)
            os.remove(part)
            n += 1
    print(f"📦 {os.path.basename(path)} uploaded in {n} parts")


def store_download_big(name, dest):
    """Download a file uploaded with store_upload_big and rejoin its parts."""
    store_download([name + "*"], dest)
    parts = sorted(f for f in os.listdir(dest) if f.startswith(name + ".part"))
    if parts:
        with open(os.path.join(dest, name), "wb") as out:
            for p in parts:
                with open(os.path.join(dest, p), "rb") as f:
                    shutil.copyfileobj(f, out, 64 << 20)
                os.remove(os.path.join(dest, p))
    return os.path.join(dest, name)


# -------------------------------------------------------------------------------
# download
# -------------------------------------------------------------------------------
def cmd_download(url):
    dest = os.path.join(WORK, "src")
    os.makedirs(dest, exist_ok=True)
    start = time.time()
    if "drive.google.com" in url or "drive.usercontent.google.com" in url:
        run(["pipx", "run", "gdown", "--fuzzy", url, "-O", dest + "/"])
    elif shutil.which("aria2c"):
        run(["aria2c", "-x", "16", "-s", "16", "-k", "1M", "--file-allocation=none",
             "--console-log-level=warn", "--summary-interval=30", "--max-tries=5",
             "--auto-file-renaming=false", "-d", dest, url])
    else:
        run(["curl", "-fL", "--retry", "5", "-J", "-O", "--output-dir", dest, url])

    files = [f for f in os.listdir(dest) if not f.endswith(".aria2")]
    if len(files) != 1:
        die(f"expected one downloaded file, found: {files}")
    path = os.path.join(dest, files[0])
    print(f"✅ Downloaded {files[0]} ({mb(path)}) in {CE.format_time(time.time() - start)}")


def find_source():
    dest = os.path.join(WORK, "src")
    files = os.listdir(dest) if os.path.isdir(dest) else []
    if len(files) != 1:
        die(f"source not found in {dest}")
    return os.path.join(dest, files[0])


# -------------------------------------------------------------------------------
# prepare
# -------------------------------------------------------------------------------
def probe_timing(src, vindex):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                        "format=start_time:stream=index,start_time,avg_frame_rate,r_frame_rate",
                        "-of", "json", src], capture_output=True, text=True)
    data = json.loads(r.stdout or "{}")
    fmt_start = float(data.get("format", {}).get("start_time") or 0)
    s = next(s for s in data["streams"] if s["index"] == vindex)
    hints = [s.get("r_frame_rate") or "", s.get("avg_frame_rate") or ""]
    fps = ratio(s.get("avg_frame_rate")) or ratio(s.get("r_frame_rate")) or 24.0
    return fmt_start, fps, hints


def scan_packets(path):
    """Every packet of a file, in file order: (stream index, time, duration, is keyframe)."""
    p = subprocess.Popen(["ffprobe", "-v", "error", "-show_entries",
                          "packet=stream_index,pts_time,dts_time,duration_time,flags",
                          "-of", "csv=p=0", path], stdout=subprocess.PIPE, text=True)
    for line in p.stdout:
        a = line.strip().split(",")
        if len(a) < 3:
            continue
        ts = a[1] if a[1] not in ("", "N/A") else a[2]
        try:
            idx, t = int(a[0]), float(ts)
        except ValueError:
            continue
        try:
            d = float(a[3]) if len(a) > 3 else 0.0
        except ValueError:
            d = 0.0
        yield idx, t, d, len(a) > 4 and a[4].startswith("K")
    p.wait()


def scan_source(src, info):
    """One pass over the source: the time of every video frame (sorted) and where each
    audio track starts, ends and has gaps. This is the reference for all later checks."""
    vidx = info["vindex"]
    audio = {a["index"]: {"first": None, "end": None, "gaps": 0} for a in info["audio"]}
    times = []
    for idx, t, d, _ in scan_packets(src):
        if idx == vidx:
            times.append(t)
        elif idx in audio:
            a = audio[idx]
            if a["first"] is None:
                a["first"], a["end"] = t, t + d
            else:
                if t - a["end"] > 0.1:
                    a["gaps"] += 1
                a["first"], a["end"] = min(a["first"], t), max(a["end"], t + d)
    times.sort()
    return times, audio


STD_RATES = ["24000/1001", "24/1", "25/1", "30000/1001", "30/1", "48/1", "50/1", "60000/1001", "60/1"]


def detect_cfr(vs, hints):
    """(rate, grid start, glitch frames) if EVERY frame of the video sits on an even grid
    (start + n / fps), else None. On a grid, pieces get clean timestamps from the frame
    number, which also repairs small timestamp glitches in the source. If frames are missing
    or the timing drifts anywhere, the original timestamps are kept instead, so the video
    can never slide against the audio."""
    n = len(vs)
    if n < 10 or vs[-1] <= vs[0]:
        return None
    est = (n - 1) / (vs[-1] - vs[0])
    cands = sorted({r for r in list(hints) + STD_RATES if ratio(r) > 0}, key=lambda r: abs(ratio(r) - est))
    for rate in cands[:3]:
        fps = ratio(rate)
        if abs(fps - est) / est > 0.002:
            continue
        # grid start = median of (t - n/fps), so a glitch on the first frame can't shift it
        sample = sorted(vs[i] - i / fps for i in range(0, n, max(1, n // 2000)))
        start = sample[len(sample) // 2]
        tol = 0.5 / fps
        off = sum(1 for i, t in enumerate(vs) if abs(t - (start + i / fps)) > tol)
        tail_ok = all(abs(vs[i] - (start + i / fps)) <= tol for i in range(max(0, n - 200), n))
        if off <= max(3, n // 2000) and tail_ok:
            return rate, start, off
    return None


def plan_pieces(vs, n):
    """Cut by frame number: piece k = frames [cut k, cut k+1). Each boundary time sits halfway
    between the last frame of one piece and the first frame of the next, so every frame lands
    in exactly one piece - for constant AND variable frame rate sources. The plan also says
    how many frames each piece must contain; every encoder checks its piece against that."""
    total = len(vs)
    cuts = [0]
    for k in range(1, n):
        i = max(cuts[-1] + 1, round(total * k / n))
        while i < total - 1 and vs[i] <= vs[i - 1]:     # never cut between two frames with the same time
            i += 1
        cuts.append(min(i, total - 1))
    cuts.append(total)
    pieces = []
    for k in range(n):
        i0, i1 = cuts[k], cuts[k + 1]
        pieces.append({"index": k,
                       "start": None if k == 0 else (vs[i0 - 1] + vs[i0]) / 2,
                       "end": None if k == n - 1 else (vs[i1 - 1] + vs[i1]) / 2,
                       "first": i0, "frames": i1 - i0, "t0": vs[i0]})
    return pieces


def cut_piece(src, plan, p):
    """Stream-copy one piece (video only) with EDGE seconds extra, original timestamps + offset."""
    out = os.path.join(WORK, f"chunk_{p['index']:03d}.mkv")
    args = []
    if p["start"] is not None:
        args += ["-ss", f"{max(0.0, p['start'] - plan['fmt_start'] - EDGE):.3f}"]
    if p["end"] is not None:
        args += ["-to", f"{p['end'] - plan['fmt_start'] + EDGE:.3f}"]
    args += ["-copyts", "-i", src, "-map", f"0:{plan['vindex']}", "-c", "copy",
             "-output_ts_offset", str(TS_OFFSET), out]
    ffmpeg(args, f"cut {p['index']}")
    size = os.path.getsize(out)
    if size > 2000 * 1024 * 1024:
        die(f"piece {p['index']} is over 2 GB - use more chunks")
    print(f"✂️  piece {p['index']}: {mb(out)}", flush=True)
    store_upload(out)
    os.remove(out)
    return size


WEIGHT_MIN, WEIGHT_MAX = 0.6, 1.6     # a piece gets 0.6x - 1.6x of the format's bitrate


def piece_weights(plan, sizes):
    """VBR shares: each piece's share of the bitrate follows how many bits the SOURCE spent
    on it per second - a hard battle scene is much bigger in the source than calm talking.
    The time-weighted average share is exactly 1.0, so the whole movie lands on the
    format's bitrate. "vbr-flat" (or a single piece) = every piece gets 1.0."""
    n = len(plan["pieces"])
    if getattr(CE, "SIZE_MODE", "") != "vbr" or n < 2:
        return [1.0] * n
    first, last = plan["vstart"], plan["vstart"] + plan["duration"]
    durs, rates = [], []
    for p, size in zip(plan["pieces"], sizes):
        s = p["start"] if p["start"] is not None else first
        e = p["end"] if p["end"] is not None else last
        cut_len = (e - s) + (EDGE if p["start"] is not None else 0) + (EDGE if p["end"] is not None else 0)
        durs.append(max(e - s, 0.001))
        rates.append(size / max(cut_len, 0.001))
    mean = sum(r * d for r, d in zip(rates, durs)) / sum(durs)
    w = [r / mean for r in rates]
    for _ in range(6):        # keep inside the limits, then re-balance to average 1.0
        w = [min(WEIGHT_MAX, max(WEIGHT_MIN, x)) for x in w]
        avg = sum(x * d for x, d in zip(w, durs)) / sum(durs)
        w = [x / avg for x in w]
    return [round(x, 3) for x in w]


AUDIO_SOURCE = "audiosrc.mkv"


def extract_audio_source(src, plan):
    """For the separate audio job: COPY (no re-encoding, takes seconds) the kept audio tracks,
    subtitles, fonts, chapters and metadata into one file and upload it. Timestamps are written
    relative to the source's start exactly like before, so any audio delay is kept."""
    info = plan["info"]
    if not plan["has_extras"]:
        return
    out = os.path.join(WORK, AUDIO_SOURCE)
    args = ["-i", src]
    for a in plan.get("audio", info["audio"]):
        args += ["-map", f"0:{a['index']}"]
    for s in info["subs"]:
        args += ["-map", f"0:{s['index']}"]
    if info["has_attachments"]:
        args += ["-map", "0:t?"]
    args += ["-map_metadata", "0", "-map_chapters", "0", "-c", "copy"]
    for i, s in enumerate(info["subs"]):
        if s["codec"] in CE.SUBS_TO_SRT:          # MKV can't hold these as-is
            args += [f"-c:s:{i}", "srt"]
    args += [out]
    ffmpeg(args, "copy audio")
    print(f"🔊 Audio copied for the audio job: {mb(out)}", flush=True)
    store_upload_big(out)
    os.remove(out)


def make_extras(src, plan, copied=False):
    """Encode the complete audio once per format + copy subtitles/chapters/attachments.
    copied=True: src is the audio-only copy from extract_audio_source (audio job)."""
    info = plan["info"]
    if not plan["has_extras"]:
        print("ℹ️  No audio/subtitles in source - nothing to add later")
        return
    audio = plan.get("audio", info["audio"])
    filtered = len(audio) < len(info["audio"])
    # -copyts: the copy already has its timestamps relative to the source's start
    args = (["-copyts"] if copied else []) + ["-i", src]
    outs = []
    for tag in plan["tags"]:
        fmt = CE.FORMATS[tag]
        out = os.path.join(WORK, f"extras_{tag}.mkv")
        outs.append(out)
        if copied:                    # the copy holds exactly the kept tracks, in order
            args += ["-map", "0:a?", "-map", "0:s?", "-map", "0:t?"]
        else:
            if filtered:
                for a in audio:
                    args += ["-map", f"0:{a['index']}"]
            else:
                args += ["-map", "0:a?"]
            for s in info["subs"]:
                args += ["-map", f"0:{s['index']}"]
            if info["has_attachments"]:
                args += ["-map", "0:t?"]
        args += ["-map_metadata", "0", "-map_chapters", "0",
                 "-c:a", fmt.get("audio_codec", "aac"), "-b:a", fmt["audio_bitrate"]]
        if fmt["audio_channels"]:
            for i, a in enumerate(audio):
                if a["channels"] > fmt["audio_channels"]:
                    args += [f"-ac:a:{i}", str(fmt["audio_channels"])]
        if filtered and audio:
            args += ["-disposition:a:0", "default"]   # kept track plays by default
        for i, s in enumerate(info["subs"]):
            conv = s["codec"] in CE.SUBS_TO_SRT and not copied
            args += [f"-c:s:{i}", "srt" if conv else "copy"]
        if info["has_attachments"]:
            args += ["-c:t", "copy"]
        args += [out]
    ffmpeg(args, "audio")
    for out in outs:
        print(f"🔊 {os.path.basename(out)}: {mb(out)}", flush=True)
        store_upload(out)


def cmd_audio():
    """Audio job: runs on its own runner at the same time as the video pieces. Encodes the
    full-length audio (never split, so no joins in the sound) for every format."""
    d = os.path.join(WORK, "audio")
    store_download(["plan.json"], d)
    with open(os.path.join(d, "plan.json"), encoding="utf-8") as f:
        plan = json.load(f)
    if not plan["has_extras"]:
        print("ℹ️  No audio/subtitles in source - nothing to do")
        return
    start = time.time()
    src = store_download_big(AUDIO_SOURCE, d)
    print(f"🔊 Audio source: {mb(src)} | {len(plan.get('audio', []))} track(s) x {len(plan['tags'])} format(s)")
    os.makedirs(WORK, exist_ok=True)
    make_extras(src, plan, copied=True)
    print(f"✅ Audio done in {CE.format_time(time.time() - start)}")


def probe_crf(src, plan):
    """Target-size mode: encode a short sample of the movie (8 x 3 s from across it) with the
    real settings, measure the bitrate, and pick the CRF per format that lands on its target.
    x264/x265 rule: +6 CRF = half the bitrate. The CRF is kept inside a quality range, so a
    very hard movie goes a bit over the target instead of looking bad, and an easy movie
    comes in under it instead of wasting bits."""
    tags = plan["tags"]
    if getattr(CE, "SIZE_MODE", "crf") != "target" or \
            not all(CE.FORMATS[t].get("target_kbps") for t in tags):
        return {}
    dur = plan["duration"]
    n, seg = (8, 3.0) if dur >= 120 else (1, min(dur, 30.0))
    starts = [dur * (0.05 + 0.9 * i / (n - 1)) for i in range(n)] if n > 1 else [0.0]
    args = []
    for s in starts:
        args += ["-ss", f"{s:.2f}", "-t", f"{seg}", "-i", src]
    head = "".join(f"[{i}:{plan['vindex']}]" for i in range(n)) + f"concat=n={n}:v=1:a=0"
    pre = pre_filters(plan)
    if pre:
        head += "," + ",".join(pre)
    graph = [f"{head},split={len(tags)}" + "".join(f"[p{i}]" for i in range(len(tags)))]
    graph += [f"[p{i}]{scale_chain(plan, t)}[o{i}]" for i, t in enumerate(tags)]
    outs = {t: os.path.join(WORK, f"sizetest_{re.sub(r'[^A-Za-z0-9]', '_', t)}.mkv") for t in tags}
    args += ["-filter_complex", ";".join(graph)]
    for i, t in enumerate(tags):
        args += ["-map", f"[o{i}]"] + CE.video_args(t) + ["-f", "matroska", outs[t]]
    start = time.time()
    ffmpeg(args, "size test")

    crfs = {}
    for t in tags:
        fmt = CE.FORMATS[t]
        kbps = os.path.getsize(outs[t]) * 8 / (n * seg) / 1000
        os.remove(outs[t])
        lo, hi = (18, 30) if fmt.get("codec") == "x265" else (16, 26)
        want = fmt["crf"] + 6 * math.log2(max(kbps, 1) / fmt["target_kbps"])
        crf = round(min(hi, max(lo, want)), 1)
        note = ""
        if want > hi:
            note = f" (capped at CRF {hi} to protect quality - file will be above target)"
        elif want < lo:
            note = f" (easy video - will come in under target)"
        print(f"🎯 {t}: sample {kbps:.0f} kbps at CRF {fmt['crf']} -> CRF {crf} "
              f"for {fmt['target_kbps']} kbps target{note}", flush=True)
        crfs[t] = crf
    print(f"🎯 Size test done in {CE.format_time(time.time() - start)}")
    return crfs


def cmd_prepare(chunks_req, formats, crop_flag, name=""):
    src = find_source()
    info = CE.probe(src)
    if info is None:
        die("ffprobe could not read the source file")
    fmt_start, fps, rate_hints = probe_timing(src, info["vindex"])
    duration = info["duration"]
    if duration <= 0:
        die("could not read the video duration")
    # The time of every video frame + where each audio track starts and ends
    t_scan = time.time()
    vs, audio_scan = scan_source(src, info)
    if len(vs) < 2:
        die("could not read the video frames of the source")
    cfr = detect_cfr(vs, rate_hints)
    if cfr:
        cfr_rate, vstart, glitches = cfr
        fps = ratio(cfr_rate)
        print(f"⏱️  Constant {fps:.3f} fps ({len(vs)} frames, all on the grid"
              + (f" except {glitches} glitch frames" if glitches else "")
              + f") - timestamps will be rebuilt cleanly [scan {time.time() - t_scan:.0f}s]")
    else:
        cfr_rate, vstart = None, vs[0]
        print(f"⏱️  Variable frame rate or missing frames ({len(vs)} frames) - "
              f"keeping the original timestamps [scan {time.time() - t_scan:.0f}s]")
    print(f"📐 {os.path.basename(src)}: {info['width']}x{info['height']} @ {fps:.3f} fps, "
          f"{CE.format_time(duration)}, {len(info['audio'])} audio, {len(info['subs'])} subs")

    tags = [t.strip() for t in formats.split(",") if t.strip()]
    unknown = [t for t in tags if t not in CE.FORMATS]
    if unknown or not tags:
        die(f"unknown format(s) {unknown} - choose from {list(CE.FORMATS)}")

    crop = CE.smart_crop_detect(src, info) if str(crop_flag).lower() in ("true", "yes", "y", "1") else None

    eff_w = (crop["w"] if crop else info["width"]) * info["sar"]
    eff_h = crop["h"] if crop else info["height"]
    if CE.SKIP_UPSCALE:
        kept = []
        for t in tags:
            bw, bh = CE.FORMATS[t]["box"]
            if eff_w < bw * CE.UPSCALE_TOLERANCE and eff_h < bh * CE.UPSCALE_TOLERANCE:
                print(f"⏭️  {t}: source is smaller ({eff_w:.0f}x{eff_h}) - skipped (no upscaling)")
            else:
                kept.append(t)
        tags = kept
    if not tags:
        die("every selected resolution is bigger than the source")

    # Number of pieces: as asked, but each piece >= MIN_PIECE_SEC and < 2 GB.
    # One runner is kept for the audio job (e.g. 20 -> 19 video pieces + 1 audio).
    has_extras = bool(info["audio"] or info["subs"] or info["has_attachments"])
    max_video = MAX_PIECES - 1 if has_extras else MAX_PIECES
    n = max(1, min(int(chunks_req), max_video, int(duration // MIN_PIECE_SEC) or 1))
    if has_extras and int(chunks_req) > max_video:
        print(f"ℹ️  {n} video pieces + 1 audio job = {n + 1} runners at once")
    need = math.ceil(os.path.getsize(src) * 1.1 / MAX_ASSET)
    if need > n:
        if need > max_video:
            die("source is too large to split into release assets")
        print(f"ℹ️  Using {need} pieces so each stays under 2 GB")
        n = need

    # Boundaries sit between two frames -> every frame lands in exactly one piece
    step = duration / n
    pieces = plan_pieces(vs, n)

    base = name.strip() or os.path.splitext(os.path.basename(src))[0]
    if base.lower().endswith(CE.VIDEO_EXTENSIONS):
        base = os.path.splitext(base)[0]
    plan = {
        "source": os.path.basename(src), "base": base, "duration": duration,
        "fmt_start": fmt_start, "vstart": vstart, "fps": fps, "cfr_rate": cfr_rate,
        "frames": len(vs),
        "vindex": info["vindex"], "sar": info["sar"], "crop": crop, "tags": tags,
        "pieces": pieces, "info": info,
        "has_extras": has_extras,
        "outputs": {t: output_name(base, t) for t in tags},
    }
    # Audio tracks to keep (AUDIO_LANGS)
    plan["audio"] = select_audio(info["audio"])
    if info["audio"]:
        kept = ", ".join(f"{a.get('language') or '?'}" + (f" ({a['title']})" if a.get("title") else "")
                         for a in plan["audio"])
        print(f"🔊 Audio: keeping {len(plan['audio'])} of {len(info['audio'])} track(s): {kept}")
    # Where each kept audio track starts/ends in the source: the join is checked against this
    plan["audio_ref"] = [{"start": audio_scan[a["index"]]["first"] - fmt_start,
                          "end": audio_scan[a["index"]]["end"] - fmt_start,
                          "gaps": audio_scan[a["index"]]["gaps"]}
                         for a in plan["audio"] if audio_scan.get(a["index"], {}).get("first") is not None]
    if not cfr_rate:
        # Variable frame rate: the join is checked against the source's own frame times
        timeline_path = os.path.join(WORK, "timeline.json")
        with open(timeline_path, "w", encoding="utf-8") as f:
            json.dump([round(t - fmt_start, 6) for t in vs], f)
        store_upload(timeline_path)
        plan["timeline"] = "timeline.json"

    # Cover image(s): saved as files now, attached to the output when joining
    covers = CE.dump_covers(src, info, WORK) if hasattr(CE, "dump_covers") else []
    plan["covers"] = [{"file": os.path.basename(c["path"]), "filename": c["filename"],
                       "mimetype": c["mimetype"]} for c in covers]
    if covers:
        print(f"🖼️  Cover image: {', '.join(c['filename'] for c in covers)}")
    os.makedirs(WORK, exist_ok=True)
    plan_path = os.path.join(WORK, "plan.json")
    print(f"🧩 {n} pieces of ~{CE.format_time(step)} | formats: {' + '.join(tags)} | "
          f"crop: {crop['ffmpeg_crop'] if crop else 'none'} | {describe_settings()}")
    for t, o in plan["outputs"].items():
        print(f"   {t} -> {o}")
    for c in covers:
        store_upload(c["path"])

    # Cutting/uploading pieces, copying out the audio (both disk + network) and the size
    # test (CPU) run together. The audio is ENCODED later by its own job, in parallel with
    # the video pieces, so it no longer holds up the encoders.
    start = time.time()
    with ThreadPoolExecutor(2) as side:
        audio_copy = side.submit(extract_audio_source, src, plan)
        size_job = side.submit(probe_crf, src, plan)
        with ThreadPoolExecutor(3) as pool:
            sizes = list(pool.map(lambda p: cut_piece(src, plan, p), pieces))
        audio_copy.result()
        plan["crf"] = size_job.result()
    plan["size_mode"] = getattr(CE, "SIZE_MODE", "crf")
    if plan["size_mode"] in ("vbr", "vbr-flat"):
        plan["piece_weight"] = piece_weights(plan, sizes)
        w = plan["piece_weight"]
        hard = max(range(n), key=lambda i: w[i])
        print(f"🎚️  VBR shares per piece: {' '.join(f'{x:.2f}' for x in w)}")
        print(f"🎚️  Hardest piece {hard}: {w[hard]:.2f}x the bitrate | whole movie averages exactly 1.00x")
    # The plan goes up last: encode jobs start only after this job, and need the CRFs
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2)
    store_upload(plan_path)
    print(f"✅ Prepared in {CE.format_time(time.time() - start)}")

    set_output("matrix", json.dumps(list(range(n))))
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in base)
    set_output("name", safe)


# -------------------------------------------------------------------------------
# encode
# -------------------------------------------------------------------------------
def cmd_encode(index):
    k = int(index)
    d = os.path.join(WORK, f"piece_{k:03d}")
    store_download(["plan.json", f"chunk_{k:03d}.mkv"], d)
    with open(os.path.join(d, "plan.json"), encoding="utf-8") as f:
        plan = json.load(f)
    p = plan["pieces"][k]
    tags = plan["tags"]

    trim = []
    if p["start"] is not None:
        trim.append(f"start={p['start'] + TS_OFFSET:.6f}")
    if p["end"] is not None:
        trim.append(f"end={p['end'] + TS_OFFSET:.6f}")
    # Constant frame rate: clean timestamps from the frame number (repairs source glitches).
    # Otherwise keep the source timing, just starting the piece at 0.
    retime = f"setpts=N/({plan['cfr_rate']})/TB" if plan.get("cfr_rate") else "setpts=PTS-STARTPTS"
    chain = (["trim=" + ":".join(trim)] if trim else []) + [retime] + pre_filters(plan)

    # Size mode + the CRFs picked by the size test in the prepare job
    if hasattr(CE, "SIZE_MODE"):
        CE.SIZE_MODE = plan.get("size_mode", CE.SIZE_MODE)
    if hasattr(CE, "CRF_OVERRIDE"):
        CE.CRF_OVERRIDE = plan.get("crf") or {}
    if hasattr(CE, "PIECE_BITRATE") and plan.get("size_mode") in ("vbr", "vbr-flat"):
        w = (plan.get("piece_weight") or [1.0] * len(plan["pieces"]))[k]
        CE.PIECE_BITRATE = {t: CE.FORMATS[t]["target_kbps"] * w for t in tags}
        print(f"🎚️  Piece {k} VBR share {w:.2f}x: " +
              ", ".join(f"{t} {CE.PIECE_BITRATE[t]:.0f}k" for t in tags), flush=True)

    def scale(t):
        return f"{scale_chain(plan, t)}[{vlabel(t)}]"

    head = "[0:v:0]" + ",".join(chain)
    if len(tags) == 1:
        graph = f"{head},{scale(tags[0])}"
    else:
        labels = "".join(f"[s{i}]" for i in range(len(tags)))
        graph = ";".join([f"{head},split={len(tags)}{labels}"] +
                         [f"[s{i}]{scale(t)}" for i, t in enumerate(tags)])

    args = ["-copyts", "-i", os.path.join(d, f"chunk_{k:03d}.mkv"), "-filter_complex", graph]
    outs = []
    for t in tags:
        out = os.path.join(d, f"piece_{t}_{k:03d}.mkv")
        outs.append(out)
        vargs = CE.video_args(t) if hasattr(CE, "video_args") else CE.x264_args(t)
        args += ["-map", f"[{vlabel(t)}]"] + vargs + ["-fps_mode", "passthrough", out]

    start = time.time()
    ffmpeg(args, f"piece {k}")
    secs = time.time() - start
    print(f"✅ Piece {k} encoded in {CE.format_time(secs)}")

    # Check every output before it leaves this runner: its frame count must match the plan
    expected = p.get("frames")
    report = {"index": k, "expected": expected, "seconds": round(secs),
              "frames": {}, "bytes": {}, "header": {}, "md5": {}}
    wrong = []
    for t, out in zip(tags, outs):
        got = count_frames(out)
        report["frames"][t], report["bytes"][t], report["header"][t] = got, os.path.getsize(out), header_hash(out)
        report["md5"][t] = file_md5(out)        # the join verifies this: nothing may change on the way
        note = ""
        if expected is not None and got != expected:
            note = f"   <-- the plan says {expected}"
            if abs(got - expected) > FRAME_TOLERANCE:
                wrong.append(f"{t}: {got} frames, the plan says {expected}")
            else:
                print(f"::warning::piece {k} {t}: {got} frames, the plan says {expected} "
                      f"(damaged spot in the source?)")
        print(f"   {os.path.basename(out)}: {mb(out)}, {got} frames{note}")
    if wrong:
        die(f"piece {k} has the wrong number of frames - " + "; ".join(wrong))

    # One upload per piece: every format + the check report in one bundle
    report_path = os.path.join(d, f"piece_{k:03d}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f)
    bundle = os.path.join(d, f"piece_{k:03d}.tar")
    with tarfile.open(bundle, "w") as tar:
        for path in outs + [report_path]:
            tar.add(path, arcname=os.path.basename(path))
    store_upload(bundle)


FRAME_TOLERANCE = 2        # a piece may be off by this many frames (damaged spot in the source)


def file_md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def count_frames(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
                        "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", path],
                       capture_output=True, text=True)
    try:
        return int(r.stdout.strip().strip(",").split(",")[0])
    except ValueError:
        return -1


def header_hash(path):
    """Fingerprint of a piece's encoder header (resolution, profile, level, ...).
    For HEVC only the parameter sets (VPS/SPS/PPS) count: x265 also stores its settings text
    there, and that text is different in every piece (each piece has its own bitrate)."""
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_data",
                        "-show_entries", "stream=codec_name,extradata", "-of", "json", path],
                       capture_output=True, text=True)
    try:
        s = json.loads(r.stdout)["streams"][0]
    except (ValueError, KeyError, IndexError):
        return ""
    raw = bytearray()
    for line in s.get("extradata", "").splitlines():       # hex dump: "00000000: 0164 001e ...  text"
        if ":" not in line:
            continue
        try:
            raw += bytes.fromhex(line.split(":", 1)[1].strip().split("  ")[0].replace(" ", ""))
        except ValueError:
            return ""
    data = bytes(raw)
    if s.get("codec_name") == "hevc" and len(data) > 23 and data[0] == 1:
        keep, pos = bytearray(), 23
        try:
            for _ in range(data[22]):                       # arrays of NAL units
                nal_type = data[pos] & 0x3F
                count = int.from_bytes(data[pos + 1:pos + 3], "big")
                pos += 3
                for _ in range(count):
                    size = int.from_bytes(data[pos:pos + 2], "big")
                    pos += 2
                    if nal_type in (32, 33, 34):            # VPS, SPS, PPS
                        keep += data[pos:pos + size]
                    pos += size
            data = bytes(keep)
        except IndexError:
            pass
    return hashlib.md5(data).hexdigest() if data else ""


# -------------------------------------------------------------------------------
# merge
# -------------------------------------------------------------------------------
def media_duration(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "default=nw=1:nk=1", path], capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


def stream_kinds(path):
    """index -> 'main' (the movie's video), 'audio', 'cover', 'subtitle' ..."""
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                        "stream=index,codec_type:stream_disposition=attached_pic", "-of", "json", path],
                       capture_output=True, text=True)
    kinds, main_seen = {}, False
    for s in json.loads(r.stdout or "{}").get("streams", []):
        kind = s.get("codec_type")
        if kind == "video":
            if main_seen or s.get("disposition", {}).get("attached_pic"):
                kind = "cover"
            else:
                kind, main_seen = "main", True
        kinds[s["index"]] = kind
    return kinds


def check_output(path, plan, piece_frames, starts, timeline):
    """Read every packet of a finished file once and compare it with the plan: the frame
    count, the time of every single frame, a keyframe at every join, the audio's start / end /
    gaps against the source, and the audio-video write order.
    Returns (errors, warnings, facts, shift of the whole file in seconds)."""
    kinds = stream_kinds(path)
    video, audio = [], {}
    max_audio, lead = None, 0.0
    for idx, t, d, key in scan_packets(path):
        kind = kinds.get(idx)
        if kind == "main":
            video.append((t, key))
            if max_audio is not None:
                lead = max(lead, max_audio - t)
        elif kind == "audio":
            a = audio.get(idx)
            if a is None:
                audio[idx] = [t, t + d, 0]
            else:
                if t - a[1] > 0.1:
                    a[2] += 1
                a[0], a[1] = min(a[0], t), max(a[1], t + d)
            max_audio = t if max_audio is None else max(max_audio, t)
    video.sort()
    errors, warns, facts = [], [], {"frames": len(video), "lead": lead, "audio_start_ms": []}

    # 1. frame count
    total = sum(piece_frames)
    if len(video) != total:
        errors.append(f"the file has {len(video)} video frames but the pieces had {total}")
    if total != plan["frames"]:
        warns.append(f"{total} frames encoded, the source has {plan['frames']}")

    # 2. every frame at its exact time
    shift = 0.0
    if video:
        fps = ratio(plan["cfr_rate"]) if plan.get("cfr_rate") else 0.0
        shift = video[0][0] - starts[0]
        if not -0.002 <= shift <= 0.060:
            errors.append(f"the video starts {shift * 1000:+.0f} ms away from its place")
        expected = []
        for k, m in enumerate(piece_frames):
            p = plan["pieces"][k]
            if fps:
                expected += [starts[k] + j / fps for j in range(m)]
            elif timeline and m == p["frames"]:
                expected += timeline[p["first"]:p["first"] + m]
            else:
                expected += [None] * m
        worst, at, compared = 0.0, 0, 0
        for i, ((t, _), e) in enumerate(zip(video, expected)):
            if e is None:
                continue
            compared += 1
            err = abs(t - shift - e)
            if err > worst:
                worst, at = err, i
        facts["timing_ms"] = worst * 1000
        if worst > 0.0015:
            errors.append(f"frame {at} (at {video[at][0]:.3f}s) is {worst * 1000:.1f} ms away from its place")
        if compared < len(video):
            warns.append(f"{len(video) - compared} frames could not be compared with the source's timing")
        # 3. every piece must start with a keyframe
        pos = 0
        for k, m in enumerate(piece_frames):
            if pos < len(video) and not video[pos][1]:
                errors.append(f"piece {k} does not start with a keyframe (at {video[pos][0]:.3f}s)")
            pos += m

    # 4. audio against the source
    tracks = [audio[i] for i in sorted(audio)]
    want = len(plan.get("audio", [])) if plan["has_extras"] else 0
    if len(tracks) != want:
        errors.append(f"{len(tracks)} audio tracks in the file, expected {want}")
    for i, (a, ref) in enumerate(zip(tracks, plan.get("audio_ref", []))):
        d0 = (a[0] - shift) - ref["start"]      # normally minus the encoder's start-up (AAC ~21 ms)
        d1 = (a[1] - shift) - ref["end"]
        facts["audio_start_ms"].append(round(d0 * 1000))
        if not -0.060 <= d0 <= 0.010:
            errors.append(f"audio track {i + 1} starts {d0 * 1000:+.0f} ms away from the source's timing")
        if abs(d1) > 0.5:
            errors.append(f"audio track {i + 1} ends {d1:+.2f}s away from the source's audio")
        elif abs(d1) > 0.15:
            warns.append(f"audio track {i + 1} ends {d1 * 1000:+.0f} ms away from the source's audio")
        if a[2] > ref.get("gaps", 0):
            errors.append(f"audio track {i + 1} has {a[2]} gaps (the source has {ref.get('gaps', 0)})")
    if lead > 2:
        errors.append(f"audio is written up to {lead:.1f}s ahead of video - players may drop audio")
    return errors, warns, facts, shift


def decode_test(path, times):
    """Really decode 2.5 s of video around each given time (every join). Returns the problems."""
    def one(t):
        # Only the decoding is tested here (frame times are checked exactly above), so the
        # decoded frames get simple evenly spaced times: one every 0.1 s.
        r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-xerror",
                            "-ss", f"{max(0.0, t - 1.0):.3f}", "-t", "2.5", "-i", path,
                            "-map", "0:v:0", "-vf", "setpts=N/(10*TB)", "-fps_mode", "passthrough",
                            "-f", "null", "-"], capture_output=True, text=True)
        err = " ".join(r.stderr.split())
        return (t, err or f"decoder exit code {r.returncode}") if (r.returncode != 0 or err) else None
    with ThreadPoolExecutor(4) as pool:
        return [x for x in pool.map(one, times) if x]


def unpack_bundles(d):
    for name in sorted(os.listdir(d)):
        if name.startswith("piece_") and name.endswith(".tar"):
            path = os.path.join(d, name)
            with tarfile.open(path) as tar:
                try:
                    tar.extractall(d, filter="data")
                except TypeError:                     # Python older than 3.12
                    tar.extractall(d)
            os.remove(path)


def video_codec(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=codec_name", "-of", "csv=p=0", path], capture_output=True, text=True)
    return r.stdout.strip().strip(",")


def headers_inband(piece):
    """Rewrite one HEVC piece (stream copy) so its encoder header travels inside the stream."""
    tmp = piece + ".hdr.mkv"
    r = subprocess.run(["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", piece, "-map", "0:v:0",
                        "-c", "copy", "-bsf:v", "hevc_mp4toannexb", tmp], capture_output=True, text=True)
    if r.returncode != 0:
        die(f"could not rewrite {os.path.basename(piece)}: {r.stderr[:300]}")
    os.replace(tmp, piece)


def cmd_merge():
    d = os.path.join(WORK, "merge")
    store_download(["plan.json", "piece_*"], d)
    with open(os.path.join(d, "plan.json"), encoding="utf-8") as f:
        plan = json.load(f)
    unpack_bundles(d)
    if plan["has_extras"]:
        store_download(["extras_*"], d)
    covers = [dict(c, path=os.path.join(d, c["file"])) for c in plan.get("covers", [])]
    if covers:
        store_download([c["file"] for c in covers], d)
    timeline = None
    if plan.get("timeline"):
        store_download([plan["timeline"]], d)
        with open(os.path.join(d, plan["timeline"]), encoding="utf-8") as f:
            timeline = json.load(f)
    # Fonts (attachments) come in through the extras file, covers are attached after them
    fonts = plan["info"].get("n_attachments", 0) if plan["has_extras"] else 0

    out_dir = os.path.join(WORK, "out")
    os.makedirs(out_dir, exist_ok=True)
    ps = plan["pieces"]
    n = len(ps)
    # Audio/subs were written relative to the file start; video starts at vstart
    offset = plan["vstart"] - plan["fmt_start"]
    # Where every piece must start in the finished file, and how long the plan says it is.
    # The join puts each piece at exactly that time, so nothing can add up to a drift.
    fps = ratio(plan["cfr_rate"]) if plan.get("cfr_rate") else 0.0
    if fps:
        starts = [offset + p["first"] / fps for p in ps]
        planned = [p["frames"] / fps for p in ps]
    else:
        starts = [p["t0"] - plan["fmt_start"] for p in ps]
        planned = [ps[k + 1]["t0"] - ps[k]["t0"] for k in range(n - 1)] + [None]

    reports = []
    for k in range(n):
        path = os.path.join(d, f"piece_{k:03d}.json")
        if not os.path.exists(path):
            die(f"piece {k} is missing - no result was uploaded for it")
        with open(path, encoding="utf-8") as f:
            reports.append(json.load(f))
        if reports[k].get("index") != k:
            die(f"piece {k}: its bundle holds piece {reports[k].get('index')}")

    failed, rows = [], []
    for t in plan["tags"]:
        pieces = [os.path.join(d, f"piece_{t}_{k:03d}.mkv") for k in range(n)]
        missing = [os.path.basename(p) for p in pieces if not os.path.exists(p)]
        if missing:
            die(f"missing pieces for {t}: {missing}")
        piece_frames = [reports[k]["frames"].get(t, -1) for k in range(n)]
        notes = []
        # Every piece must arrive exactly as its encoder wrote it
        for k, p in enumerate(pieces):
            want = reports[k].get("md5", {}).get(t)
            if want and file_md5(p) != want:
                die(f"piece {k} ({t}) was damaged on the way (checksum mismatch) - re-run the failed job")

        # The encoder header must be the same in every piece
        if len({reports[k]["header"].get(t) for k in range(n)}) > 1:
            if video_codec(pieces[0]) == "hevc":
                for p in pieces:
                    headers_inband(p)
                notes.append("encoder headers differed between pieces - every piece now carries its own")
            else:
                notes.append("encoder headers differ between pieces (the join keeps each piece's own)")

        list_path = os.path.join(d, f"list_{t}.txt")
        with open(list_path, "w", encoding="utf-8") as f:
            for p, dur in zip(pieces, planned):
                f.write("file '" + p.replace("\\", "/").replace("'", "'\\''") + "'\n")
                if dur is not None:
                    f.write(f"duration {dur:.6f}\n")

        out = os.path.join(out_dir, plan["outputs"][t])
        extras = os.path.join(d, f"extras_{t}.mkv")
        args = ["-copyts", "-itsoffset", f"{offset:.6f}", "-f", "concat", "-safe", "0", "-i", list_path]
        if plan["has_extras"]:
            args += ["-i", extras, "-map", "0:v", "-map", "1",
                     "-map_metadata", "1", "-map_chapters", "1"]
        else:
            args += ["-map", "0:v"]
        args += ["-c", "copy"]
        if covers:
            args += CE.cover_attach_args(covers, fonts)
        if plan["info"]["subs"] and CE.DEFAULT_FIRST_SUBTITLE:
            args += ["-disposition:s:0", "default"]
        # Strict interleaving: never write audio ahead of video. Without this the muxer
        # gives up waiting after 10 s and writes audio up to minutes ahead, which makes
        # players drop the audio for a few seconds at those points.
        args += ["-max_interleave_delta", "0", "-max_muxing_queue_size", "4096", out]
        ffmpeg(args, f"join {t}")

        # ---- checks: nothing is uploaded unless all of them pass ----
        name = os.path.basename(out)
        got = media_duration(out)
        size_text = mb(out)
        errors, warns, facts, shift = check_output(out, plan, piece_frames, starts, timeline)
        if abs(got - plan["duration"]) > 1.5:
            warns.append(f"length is {got:.2f}s, the source is {plan['duration']:.2f}s")
        spots = [0.5] + [s + shift for s in starts[1:]] + [max(0.0, got - 2.5)]
        broken = decode_test(out, spots)
        errors += [f"decode error near {tm:.1f}s: {msg[:200]}" for tm, msg in broken[:5]]
        warns += notes

        print(f"{'❌' if errors else '✅'} {name}: {size_text}, {CE.format_time(got)}")
        video_kbps = sum(os.path.getsize(p) for p in pieces) * 8 / max(plan["duration"], 1) / 1000
        mode = plan.get("size_mode")
        target = CE.FORMATS[t].get("target_kbps") if mode in ("target", "vbr", "vbr-flat") else None
        how = f"CRF {(plan.get('crf') or {}).get(t)}" if mode == "target" else (mode or "").upper()
        print(f"🎯 {t} video: {video_kbps:.0f} kbps" +
              (f" (target {target}, {how}, {(video_kbps / target - 1) * 100:+.1f}%)" if target else ""))
        audio_text = ", ".join(f"{x:+d} ms" for x in facts["audio_start_ms"]) or "no audio"
        print(f"🔎 Checks: {facts['frames']} frames (source {plan['frames']}) | every frame within "
              f"{facts.get('timing_ms', 0):.1f} ms of its place | {len(spots)} spots decoded, "
              f"{len(broken)} bad | audio start {audio_text} | audio ahead of video {facts['lead']:.2f}s")
        for w in warns:
            print(f"::warning::{name}: {w}")
        for e in errors:
            print(f"::error::{name}: {e}")
        rows.append((name, size_text, f"{facts['frames']} / {plan['frames']}",
                     f"{facts.get('timing_ms', 0):.1f} ms", f"{len(spots) - len(broken)} / {len(spots)}",
                     audio_text, "FAILED" if errors else ("passed, see warnings" if warns else "passed")))
        if errors:
            os.replace(out, out + ".failed")        # keeps it out of every upload step
            failed.append(t)
        for p in pieces:
            os.remove(p)

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write("### Checks before upload\n\n| File | Size | Frames (file / source) | Frame timing | "
                    "Joins decoded OK | Audio start vs source | Result |\n|---|---|---|---|---|---|---|\n")
            for row in rows:
                f.write("| " + " | ".join(row) + " |\n")
            f.write("\n")
    if failed:
        die(f"checks failed for {', '.join(failed)} - those files are not uploaded. "
            f"The pieces are kept, so the join can be re-run.")


# -------------------------------------------------------------------------------
# publish / cleanup
# -------------------------------------------------------------------------------
def cmd_publish(release_tag, title):
    """Upload finished files (< 2 GB) to a new draft release; bigger ones go to out/big/."""
    out_dir = os.path.join(WORK, "out")
    big_dir = os.path.join(out_dir, "big")
    small = []
    for f in sorted(os.listdir(out_dir)):
        path = os.path.join(out_dir, f)
        if not os.path.isfile(path):
            continue
        if os.path.getsize(path) >= 2 * 1024 ** 3 - 1:
            os.makedirs(big_dir, exist_ok=True)
            shutil.move(path, os.path.join(big_dir, f))
            print(f"::warning::{f} is over 2 GB (release limit) - uploaded as an artifact instead")
        else:
            small.append(path)
    if small:
        run(["gh", "release", "create", release_tag, "--draft", "--title", title,
             "--notes", "Split encode output (CPU x264)."] + small)


GOFILE_UPLOAD = "https://upload.gofile.io/uploadfile"


def gofile_upload(path, token="", folder=""):
    """Upload one file to Gofile with curl (streams from disk). Returns the 'data' dict."""
    cmd = ["curl", "-sS", "--fail-with-body", "--retry", "3", "--retry-all-errors",
           "-F", f'file=@"{path}"']
    if folder:
        cmd += ["-F", f"folderId={folder}"]
    if token:
        cmd += ["-H", f"Authorization: Bearer {token}"]
    cmd.append(GOFILE_UPLOAD)
    print(f"⬆️  Uploading {os.path.basename(path)} ({mb(path)}) to Gofile ...", flush=True)
    start = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    try:
        reply = json.loads(r.stdout)
    except ValueError:
        reply = {}
    if r.returncode != 0 or reply.get("status") != "ok":
        die(f"Gofile upload failed for {os.path.basename(path)}: {r.stdout[:300]} {r.stderr[:300]}")
    secs = time.time() - start
    speed = os.path.getsize(path) / (1024 * 1024) / max(secs, 0.1)
    print(f"✅ {os.path.basename(path)} uploaded in {CE.format_time(secs)} ({speed:.0f} MB/s)", flush=True)
    return reply["data"]


def cmd_gofile():
    """Upload every finished file into one Gofile folder and print the link."""
    out_dir = os.path.join(WORK, "out")
    files = sorted(os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith(".mkv"))
    if not files:
        die("no finished files to upload")
    account = os.environ.get("GOFILE_TOKEN", "").strip()

    # First file creates the folder; the rest go into it at the same time
    first = gofile_upload(files[0], token=account)
    folder = first["parentFolder"]
    token = account or first.get("guestToken", "")
    with ThreadPoolExecutor(3) as pool:
        list(pool.map(lambda f: gofile_upload(f, token=token, folder=folder), files[1:]))

    link = first["downloadPage"]
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if account:
        # Repo logs are public: keep the link out of them, it's in the Gofile account
        print("🔗 Uploaded to your Gofile account (link not printed because logs are public).")
        if summary:
            with open(summary, "a", encoding="utf-8") as f:
                f.write("### ✅ Uploaded to your Gofile account\n")
        return
    print(f"🔗 Gofile link: {link}")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write(f"### ✅ Gofile download\n\n{link}\n\n")
            for p in files:
                f.write(f"- {os.path.basename(p)} ({mb(p)})\n")


def cmd_cleanup():
    run(["gh", "release", "delete", TAG, "--yes"], check=False)


def main():
    if len(sys.argv) < 2:
        die(__doc__)
    cmd, args = sys.argv[1], sys.argv[2:]
    commands = {"download": cmd_download, "prepare": cmd_prepare, "encode": cmd_encode,
                "audio": cmd_audio,
                "merge": cmd_merge, "publish": cmd_publish, "gofile": cmd_gofile,
                "cleanup": cmd_cleanup}
    if cmd not in commands:
        die(f"unknown command {cmd}")
    commands[cmd](*args)


if __name__ == "__main__":
    main()
