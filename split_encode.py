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

Commands:
  download <url>                             fetch the source into $WORK/src/
  prepare  <chunks> <formats> <crop> [name]  plan, cut + upload pieces, encode audio
  encode   <index>                           encode one piece into every resolution
  merge                                      join pieces + audio/subs/chapters -> $WORK/out/
  publish                                    put the results in a draft release
  gofile                                     upload the results to one Gofile folder
  cleanup                                    delete the temporary draft release
"""
import fnmatch
import importlib
import json
import math
import os
import shutil
import subprocess
import sys
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
MAX_ASSET = 1800 * 1024 * 1024          # release assets must stay under 2 GB

if hasattr(CE, "apply_env"):
    CE.apply_env()
if hasattr(CE, "X264_PRESET"):
    CE.X264_PRESET = os.environ.get("X264_PRESET") or CE.X264_PRESET
    if "X264_TUNE" in os.environ:
        CE.X264_TUNE = "" if os.environ["X264_TUNE"] in ("", "none") else os.environ["X264_TUNE"]
    if "X264_ADVANCED" in os.environ:
        CE.X264_ADVANCED = os.environ["X264_ADVANCED"].lower() == "true"


def describe_settings():
    if hasattr(CE, "describe"):
        return CE.describe()
    return (f"preset: {CE.X264_PRESET} | tune: {CE.X264_TUNE or 'none'} | "
            f"advanced: {'on' if CE.X264_ADVANCED else 'off'}")


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
    vstart = float(s.get("start_time") or fmt_start)
    fps = ratio(s.get("avg_frame_rate")) or ratio(s.get("r_frame_rate")) or 24.0
    return fmt_start, vstart, fps


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
    if os.path.getsize(out) > 2000 * 1024 * 1024:
        die(f"piece {p['index']} is over 2 GB - use more chunks")
    print(f"✂️  piece {p['index']}: {mb(out)}", flush=True)
    store_upload(out)
    os.remove(out)


def make_extras(src, plan):
    """Encode the complete audio once per resolution + copy subtitles/chapters/attachments."""
    info = plan["info"]
    if not plan["has_extras"]:
        print("ℹ️  No audio/subtitles in source - nothing to add later")
        return
    args = ["-i", src]
    outs = []
    for tag in plan["tags"]:
        fmt = CE.FORMATS[tag]
        out = os.path.join(WORK, f"extras_{tag}.mkv")
        outs.append(out)
        args += ["-map", "0:a?"]
        for s in info["subs"]:
            args += ["-map", f"0:{s['index']}"]
        if info["has_attachments"]:
            args += ["-map", "0:t?"]
        args += ["-map_metadata", "0", "-map_chapters", "0",
                 "-c:a", "aac", "-b:a", fmt["audio_bitrate"]]
        if fmt["audio_channels"]:
            for i, a in enumerate(info["audio"]):
                if a["channels"] > fmt["audio_channels"]:
                    args += [f"-ac:a:{i}", str(fmt["audio_channels"])]
        for i, s in enumerate(info["subs"]):
            args += [f"-c:s:{i}", "srt" if s["codec"] in CE.SUBS_TO_SRT else "copy"]
        if info["has_attachments"]:
            args += ["-c:t", "copy"]
        args += [out]
    ffmpeg(args, "audio")
    for out in outs:
        print(f"🔊 {os.path.basename(out)}: {mb(out)}", flush=True)
        store_upload(out)


def cmd_prepare(chunks_req, formats, crop_flag, name=""):
    src = find_source()
    info = CE.probe(src)
    if info is None:
        die("ffprobe could not read the source file")
    fmt_start, vstart, fps = probe_timing(src, info["vindex"])
    duration = info["duration"]
    if duration <= 0:
        die("could not read the video duration")
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

    # Number of pieces: as asked, but each piece >= MIN_PIECE_SEC and < 2 GB
    n = max(1, min(int(chunks_req), MAX_PIECES, int(duration // MIN_PIECE_SEC) or 1))
    need = math.ceil(os.path.getsize(src) * 1.1 / MAX_ASSET)
    if need > n:
        if need > MAX_PIECES:
            die("source is too large to split into release assets")
        print(f"ℹ️  Using {need} pieces so each stays under 2 GB")
        n = need

    # Boundaries half a frame between frames -> every frame lands in exactly one piece
    step = duration / n
    bounds = [vstart + (round(k * step * fps) - 0.5) / fps for k in range(1, n)]
    pieces = [{"index": k,
               "start": bounds[k - 1] if k > 0 else None,
               "end": bounds[k] if k < n - 1 else None} for k in range(n)]

    base = name.strip() or os.path.splitext(os.path.basename(src))[0]
    if base.lower().endswith(CE.VIDEO_EXTENSIONS):
        base = os.path.splitext(base)[0]
    plan = {
        "source": os.path.basename(src), "base": base, "duration": duration,
        "fmt_start": fmt_start, "vstart": vstart, "fps": fps,
        "vindex": info["vindex"], "sar": info["sar"], "crop": crop, "tags": tags,
        "pieces": pieces, "info": info,
        "has_extras": bool(info["audio"] or info["subs"] or info["has_attachments"]),
        "outputs": {t: output_name(base, t) for t in tags},
    }
    os.makedirs(WORK, exist_ok=True)
    plan_path = os.path.join(WORK, "plan.json")
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2)
    print(f"🧩 {n} pieces of ~{CE.format_time(step)} | formats: {' + '.join(tags)} | "
          f"crop: {crop['ffmpeg_crop'] if crop else 'none'} | {describe_settings()}")
    for t, o in plan["outputs"].items():
        print(f"   {t} -> {o}")
    store_upload(plan_path)

    start = time.time()
    with ThreadPoolExecutor(1) as audio_pool:
        audio_job = audio_pool.submit(make_extras, src, plan)
        with ThreadPoolExecutor(3) as pool:
            list(pool.map(lambda p: cut_piece(src, plan, p), pieces))
        audio_job.result()
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
    chain = (["trim=" + ":".join(trim)] if trim else []) + ["setpts=PTS-STARTPTS"]
    if plan["crop"]:
        chain.append(plan["crop"]["ffmpeg_crop"])
    if abs(plan["sar"] - 1.0) > 0.01:
        chain.append("scale='trunc(iw*sar/2)*2':ih")

    eff_w = (plan["crop"]["w"] if plan["crop"] else plan["info"]["width"]) * plan["sar"]
    eff_h = plan["crop"]["h"] if plan["crop"] else plan["info"]["height"]

    def scale(t):
        if hasattr(CE, "scale_filter"):
            return CE.scale_filter(t, eff_w, eff_h) + f"[v_{t}]"
        bw, bh = CE.FORMATS[t]["box"]
        return (f"scale={bw}:{bh}:force_original_aspect_ratio=decrease:force_divisible_by=2"
                f":flags=lanczos,setsar=1,format=yuv420p[v_{t}]")

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
        fmt = CE.FORMATS[t]
        out = os.path.join(d, f"piece_{t}_{k:03d}.mkv")
        outs.append(out)
        vargs = CE.video_args(t) if hasattr(CE, "video_args") else CE.x264_args(t)
        args += ["-map", f"[v_{t}]"] + vargs + ["-fps_mode", "passthrough", out]

    start = time.time()
    ffmpeg(args, f"piece {k}")
    print(f"✅ Piece {k} encoded in {CE.format_time(time.time() - start)}")
    for out in outs:
        print(f"   {os.path.basename(out)}: {mb(out)}")
        store_upload(out)


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


def interleave_lead(path):
    """How many seconds audio is written ahead of video, worst case, in file order.
    A few tenths is normal; many seconds makes players drop the audio."""
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=index,codec_type",
                        "-of", "csv=p=0", path], capture_output=True, text=True)
    kinds = dict(line.split(",")[:2] for line in r.stdout.split() if "," in line)
    p = subprocess.Popen(["ffprobe", "-v", "error", "-show_entries", "packet=stream_index,pts_time",
                          "-of", "csv=p=0", path], stdout=subprocess.PIPE, text=True)
    max_audio, worst = 0.0, 0.0
    for line in p.stdout:
        parts = line.strip().split(",")
        if len(parts) < 2 or parts[1] in ("", "N/A"):
            continue
        kind, t = kinds.get(parts[0]), float(parts[1])
        if kind == "audio":
            max_audio = max(max_audio, t)
        elif kind == "video":
            worst = max(worst, max_audio - t)
    p.wait()
    return worst


def cmd_merge():
    d = os.path.join(WORK, "merge")
    store_download(["plan.json", "piece_*"], d)
    with open(os.path.join(d, "plan.json"), encoding="utf-8") as f:
        plan = json.load(f)
    if plan["has_extras"]:
        store_download(["extras_*"], d)

    out_dir = os.path.join(WORK, "out")
    os.makedirs(out_dir, exist_ok=True)
    n = len(plan["pieces"])
    # Audio/subs were written relative to the file start; video starts at vstart
    offset = plan["vstart"] - plan["fmt_start"]

    for t in plan["tags"]:
        pieces = [os.path.join(d, f"piece_{t}_{k:03d}.mkv") for k in range(n)]
        missing = [os.path.basename(p) for p in pieces if not os.path.exists(p)]
        if missing:
            die(f"missing pieces for {t}: {missing}")
        list_path = os.path.join(d, f"list_{t}.txt")
        with open(list_path, "w", encoding="utf-8") as f:
            for p in pieces:
                f.write("file '" + p.replace("\\", "/").replace("'", "'\\''") + "'\n")

        out = os.path.join(out_dir, plan["outputs"][t])
        extras = os.path.join(d, f"extras_{t}.mkv")
        args = ["-copyts", "-itsoffset", f"{offset:.6f}", "-f", "concat", "-safe", "0", "-i", list_path]
        if plan["has_extras"]:
            args += ["-i", extras, "-map", "0:v", "-map", "1",
                     "-map_metadata", "1", "-map_chapters", "1"]
        else:
            args += ["-map", "0:v"]
        args += ["-c", "copy"]
        if plan["info"]["subs"] and CE.DEFAULT_FIRST_SUBTITLE:
            args += ["-disposition:s:0", "default"]
        # Strict interleaving: never write audio ahead of video. Without this the muxer
        # gives up waiting after 10 s and writes audio up to minutes ahead, which makes
        # players drop the audio for a few seconds at those points.
        args += ["-max_interleave_delta", "0", "-max_muxing_queue_size", "4096", out]
        ffmpeg(args, f"join {t}")

        got = media_duration(out)
        if abs(got - plan["duration"]) > 1.5:
            print(f"::warning::{os.path.basename(out)} is {got:.2f}s but the source is "
                  f"{plan['duration']:.2f}s - please check it")
        print(f"✅ {os.path.basename(out)}: {mb(out)}, {CE.format_time(got)}")
        lead = interleave_lead(out)
        if lead > 2:
            print(f"::warning::{os.path.basename(out)}: audio is written up to {lead:.1f}s ahead of "
                  f"video - players may drop audio")
        else:
            print(f"🔊 Audio/video order OK (audio at most {lead:.2f}s ahead of video)")
        for p in pieces:
            os.remove(p)


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
                "merge": cmd_merge, "publish": cmd_publish, "gofile": cmd_gofile,
                "cleanup": cmd_cleanup}
    if cmd not in commands:
        die(f"unknown command {cmd}")
    commands[cmd](*args)


if __name__ == "__main__":
    main()
