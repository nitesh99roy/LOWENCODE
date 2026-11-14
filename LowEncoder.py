#!/usr/bin/env python3
"""
LowEncoder.py - CPU (libx265 / HEVC) small-size encoder. One quality, no watermark.

CPU version of MOVIES.py (HEVC QSV) with the same encode structure:
  * HEVC, profile main, GOP 240 closed (IDR), 7 B-frames, 4 reference frames
  * fixed output height (480 / 720 / 1080), width follows the aspect ratio
  * every audio track -> AAC 64k stereo, subtitles copied (first one default)
  * smart black-bar crop
Rate control: CRF 27 (quality-based) capped at the old QSV max bitrate for
each quality, so busy scenes can never blow up the file size.

Local use: put videos in Encode_Input/, run `py LowEncoder.py`, pick a quality.
GitHub use: split_encode.py imports this file for its settings (ENCODER_MODULE=LowEncoder).
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter

# -------------------------------------------------------------------------------
# Settings
# -------------------------------------------------------------------------------
INPUT_FOLDER = "Encode_Input"
OUTPUT_FOLDER = "Encode_Output"

CRF = 27                  # higher = smaller file. 27 = small but clean for HEVC
X265_PRESET = "slow"      # medium / slow / slower / veryslow (MOVIES.py used veryslow on QSV)
TEN_BIT = False           # True = HEVC Main10: ~3-5% smaller + no banding, needs a 10-bit capable player

# x265 settings matching MOVIES.py + extras for better quality per MB
X265_PARAMS = {
    "keyint": 240,         # GOP 240 (MOVIES.py: -g 240)
    "min-keyint": 24,
    "open-gop": 0,         # closed GOP, every I-frame is IDR (MOVIES.py: forced_idr / idr_interval 1)
    "bframes": 7,          # MOVIES.py: -bf 7
    "ref": 4,              # MOVIES.py: -refs 4
    "b-adapt": 2,          # best B-frame placement (MOVIES.py: adaptive_b)
    "weightb": 1,
    "rc-lookahead": 60,    # look ahead when spending bits (MOVIES.py: look_ahead)
    "aq-mode": 3,          # adaptive quantisation + dark-scene bias (MOVIES.py: mbbrc)
    "psy-rd": 2.0,         # keeps texture/detail instead of smoothing it away
    "psy-rdoq": 1.0,
    "rdoq-level": 2,       # rate-distortion optimised quantisation (MOVIES.py: rdo)
    "log-level": "error",
}

# One entry per quality. maxrate/bufsize = MOVIES.py's max bitrate window (kbps).
QUALITIES = {
    "480p": {"height": 480, "maxrate": "500k", "bufsize": "1000k"},
    "720p": {"height": 720, "maxrate": "900k", "bufsize": "1800k"},
    "1080p": {"height": 1080, "maxrate": "1750k", "bufsize": "3500k"},
}
AUDIO_BITRATE = "64k"     # per audio track (MOVIES.py: -b:a 64k -ac 2)
AUDIO_CHANNELS = 2

# Shared-interface values used by split_encode.py
FORMATS = {tag: {"box": (q["height"] * 16 // 9, q["height"]),
                 "audio_bitrate": AUDIO_BITRATE, "audio_channels": AUDIO_CHANNELS}
           for tag, q in QUALITIES.items()}
SKIP_UPSCALE = True
UPSCALE_TOLERANCE = 0.9
DEFAULT_FIRST_SUBTITLE = True
SUBS_TO_SRT = {"mov_text"}
FFMPEG_STALL_TIMEOUT = 300
VIDEO_EXTENSIONS = ('.mp4', '.mkv', '.avi', '.mov', '.flv', '.wmv',
                    '.mpeg', '.mpg', '.m4v', '.3gp', '.webm', '.ts', '.m2ts')


def apply_env():
    """Settings from the GitHub workflow inputs (environment variables)."""
    global CRF, X265_PRESET, TEN_BIT
    if os.environ.get("CRF"):
        CRF = float(os.environ["CRF"])
        if CRF == int(CRF):
            CRF = int(CRF)
    X265_PRESET = os.environ.get("X265_PRESET") or X265_PRESET
    if os.environ.get("TEN_BIT"):
        TEN_BIT = os.environ["TEN_BIT"].lower() == "true"


def describe():
    return (f"x265 {X265_PRESET} | CRF {CRF} | {'10-bit Main10' if TEN_BIT else '8-bit Main'} | "
            f"audio AAC {AUDIO_BITRATE} stereo")


def video_args(tag):
    q = QUALITIES[tag]
    args = ["-c:v", "libx265", "-preset", X265_PRESET, "-crf", str(CRF),
            "-profile:v", "main10" if TEN_BIT else "main",
            "-maxrate", q["maxrate"], "-bufsize", q["bufsize"]]
    args += ["-x265-params", ":".join(f"{k}={v}" for k, v in X265_PARAMS.items())]
    return args


def target_size(tag, src_w, src_h):
    """Fixed height, width from the aspect ratio, rounded down to even (same as MOVIES.py)."""
    h = QUALITIES[tag]["height"]
    w = max(2, int(src_w * h // src_h))
    return w - (w % 2), h


def scale_filter(tag, src_w, src_h):
    w, h = target_size(tag, src_w, src_h)
    pix = "yuv420p10le" if TEN_BIT else "yuv420p"
    return f"scale={w}:{h}:flags=lanczos,setsar=1,format={pix}"


def output_name(base, tag):
    return generate_output_filename(base, tag) + ".mkv"


# -------------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------------
def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}h {m}m {s:.2f}s"


def is_video_file(filename):
    return filename.lower().endswith(VIDEO_EXTENSIONS)


def log_line(output_dir, text):
    with open(os.path.join(output_dir, "conversiontime.txt"), "a", encoding="utf-8") as f:
        f.write(text + "\n")


def generate_output_filename(base_name, target_quality):
    """Replace the quality tag in the source name (or insert it after the year)."""
    if base_name.endswith('_Source'):
        base_name = base_name[:-7]
    pattern = r'\b(480p|576p|720p|1080p|2160p|4k)\b'
    if re.search(pattern, base_name, re.IGNORECASE):
        return re.sub(pattern, target_quality, base_name, count=1, flags=re.IGNORECASE)
    year_pattern = r'(\b(?:19|20)\d{2}\b)'
    if re.search(year_pattern, base_name):
        return re.sub(year_pattern, rf"\1.{target_quality}", base_name, count=1)
    return f"{base_name}.{target_quality}"


def parse_ratio(text):
    try:
        num, den = (int(x) for x in str(text).split(":"))
        return num / den if num > 0 and den > 0 else 1.0
    except (ValueError, ZeroDivisionError):
        return 1.0


def probe(path):
    """One ffprobe call: main video stream, duration, audio channels, subtitles, attachments."""
    cmd = ["ffprobe", "-v", "error", "-show_entries",
           "format=duration:stream=index,codec_type,codec_name,width,height,channels,"
           "sample_aspect_ratio:stream_disposition=attached_pic:stream_tags=filename,mimetype",
           "-of", "json", path]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                           errors="ignore", timeout=60)
        data = json.loads(r.stdout or "{}")
    except Exception:
        return None
    if r.returncode != 0:
        return None
    streams = data.get("streams", [])
    videos = [s for s in streams if s.get("codec_type") == "video"
              and not s.get("disposition", {}).get("attached_pic")]
    if not videos or not videos[0].get("width"):
        return None
    v = videos[0]
    try:
        duration = float(data.get("format", {}).get("duration") or 0)
    except ValueError:
        duration = 0.0
    return {
        "vindex": v["index"], "width": int(v["width"]), "height": int(v["height"]),
        "sar": parse_ratio(v.get("sample_aspect_ratio", "1:1")), "duration": duration,
        "audio": [{"channels": int(s.get("channels") or 0)}
                  for s in streams if s.get("codec_type") == "audio"],
        "subs": [{"index": s["index"], "codec": s.get("codec_name", "")}
                 for s in streams if s.get("codec_type") == "subtitle"],
        "has_attachments": any(s.get("codec_type") == "attachment" for s in streams),
        "n_attachments": sum(1 for s in streams if s.get("codec_type") == "attachment"),
        # Cover image(s): MKV image attachments, which ffmpeg shows as picture streams
        "pics": [cover_info(s) for s in streams if s.get("codec_type") == "video"
                 and s.get("disposition", {}).get("attached_pic")],
    }


def cover_info(s):
    tags = {k.lower(): v for k, v in (s.get("tags") or {}).items()}
    ext = {"mjpeg": "jpg", "bmp": "bmp", "webp": "webp", "gif": "gif"}.get(s.get("codec_name"), "png")
    return {"index": s["index"], "ext": ext,
            "filename": tags.get("filename") or f"cover.{ext}",
            "mimetype": tags.get("mimetype") or ("image/jpeg" if ext == "jpg" else f"image/{ext}")}


def dump_covers(source, info, folder):
    """Save the source's cover image(s) to files so they can be attached to the output.
    (Copying the picture stream instead would turn it into a second video track.)"""
    covers = []
    for i, p in enumerate(info.get("pics", [])):
        path = os.path.join(folder, f"cover_{i}.{p['ext']}")
        r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", source,
                            "-map", f"0:{p['index']}", "-c", "copy", "-frames:v", "1",
                            "-f", "image2", path])
        if r.returncode == 0 and os.path.exists(path):
            covers.append({"path": path, "filename": p["filename"], "mimetype": p["mimetype"]})
    return covers


def cover_attach_args(covers, first_attachment):
    """-attach args; first_attachment = attachment streams already in the output (fonts)."""
    args = []
    for i, c in enumerate(covers):
        k = first_attachment + i
        args += ["-attach", c["path"], f"-metadata:s:t:{k}", f"mimetype={c['mimetype']}",
                 f"-metadata:s:t:{k}", f"filename={c['filename']}"]
    return args


def smart_crop_detect(input_path, info):
    """Sample 9 points (10%..90%), vote on the crop. Returns a crop dict or None."""
    duration, orig_w, orig_h = info["duration"], info["width"], info["height"]
    if duration <= 0:
        print("⚠️  Unknown duration - skipping crop detection")
        return None
    print(f"🔍 Detecting black bars (9 sample points across {format_time(duration)})...")
    all_crops = []
    for pct in (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90):
        cmd = ['ffmpeg', '-nostdin', '-hide_banner', '-ss', f'{duration * pct:.2f}',
               '-i', input_path, '-map', f'0:{info["vindex"]}',
               '-vf', 'cropdetect=limit=0.094:round=2:reset=0', '-f', 'null', '-t', '3', '-']
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8',
                               errors='ignore', timeout=60)
        except subprocess.TimeoutExpired:
            continue
        found = re.findall(r'crop=(\d+):(\d+):(\d+):(\d+)', r.stderr)
        if found:
            all_crops.append(tuple(map(int, found[-1])))
    if not all_crops:
        print("❌ No crop data collected - encoding without crop")
        return None

    best, count = Counter(all_crops).most_common(1)[0]
    confidence = count / len(all_crops) * 100
    if confidence < 50 and len(all_crops) >= 3:
        groups = {}
        for c in all_crops:
            for key in groups:
                if all(abs(c[i] - key[i]) <= 4 for i in range(4)):
                    groups[key].append(c)
                    break
            else:
                groups[c] = [c]
        largest = max(groups.values(), key=len)
        confidence = len(largest) / len(all_crops) * 100
        if confidence < 40:
            print(f"⚠️  Crop samples disagree ({confidence:.0f}%) - encoding without crop")
            return None
        best = max(largest, key=lambda c: c[0] * c[1])
        count = len(largest)

    width, height, x, y = best
    left, top = x, y
    right, bottom = orig_w - width - x, orig_h - height - y
    if left + right + top + bottom <= 10:
        print("✨ No significant black bars")
        return None
    width, height, x, y = width - width % 2, height - height % 2, x - x % 2, y - y % 2
    print(f"✅ Crop: {orig_w}x{orig_h} → {width}x{height} "
          f"(L={left} R={right} T={top} B={bottom}, {confidence:.0f}% agree)")
    return {"ffmpeg_crop": f"crop={width}:{height}:{x}:{y}", "w": width, "h": height,
            "original_size": f"{orig_w}x{orig_h}", "cropped_size": f"{width}x{height}"}


def run_ffmpeg(cmd):
    """Run ffmpeg with one updating progress line. Returns (return_code, log_lines)."""
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding="utf-8", errors="replace")
    lines, last_output, stalled, done = [], [time.time()], [False], threading.Event()

    def watchdog():
        while not done.wait(5):
            if time.time() - last_output[0] > FFMPEG_STALL_TIMEOUT:
                stalled[0] = True
                try:
                    process.kill()
                except Exception:
                    pass
                return

    threading.Thread(target=watchdog, daemon=True).start()
    shown = False
    try:
        for line in process.stdout:
            last_output[0] = time.time()
            text = line.rstrip()
            if text.startswith("frame=") or text.startswith("size="):
                print("\r" + text[:150].ljust(150), end="", flush=True)
                shown = True
                continue
            if shown:
                print()
                shown = False
            print(text)
            lines.append(line)
        rc = process.wait()
    except KeyboardInterrupt:
        done.set()
        process.terminate()
        process.wait()
        raise
    finally:
        done.set()
    if shown:
        print()
    if stalled[0]:
        lines.append(f"\nKilled: no ffmpeg output for {FFMPEG_STALL_TIMEOUT}s\n")
        rc = rc or 1
    return rc, lines


# -------------------------------------------------------------------------------
# Local encode (one file, one quality)
# -------------------------------------------------------------------------------
def encode_file(video, input_dir, output_dir, tag, enable_crop):
    print(f"\n🎬 ========== {video} ==========")
    start = time.time()
    source = os.path.join(input_dir, video)
    base = os.path.splitext(video)[0]
    info = probe(source)
    if info is None:
        print("❌ Could not read this file - skipping")
        return False

    final = os.path.join(output_dir, output_name(base, tag))
    if os.path.exists(final):
        print(f"⏭️  Exists, skipping: {os.path.basename(final)}")
        return True
    root, ext = os.path.splitext(final)
    part = f"{root}.part{ext}"

    crop = smart_crop_detect(source, info) if enable_crop else None
    eff_w = (crop["w"] if crop else info["width"]) * info["sar"]
    eff_h = crop["h"] if crop else info["height"]
    if SKIP_UPSCALE and eff_h < QUALITIES[tag]["height"] * UPSCALE_TOLERANCE:
        print(f"⏭️  Source ({eff_w:.0f}x{eff_h}) is smaller than {tag} - skipped (no upscaling)")
        return True

    chain = []
    if crop:
        chain.append(crop["ffmpeg_crop"])
    if abs(info["sar"] - 1.0) > 0.01:
        chain.append("scale='trunc(iw*sar/2)*2':ih")
    chain.append(scale_filter(tag, eff_w, eff_h))
    w, h = target_size(tag, eff_w, eff_h)

    cmd = ["ffmpeg", "-nostdin", "-y", "-hide_banner", "-i", source,
           "-filter_complex", f"[0:{info['vindex']}]" + ",".join(chain) + "[v]",
           "-map", "[v]", "-map", "0:a?"]
    for s in info["subs"]:
        cmd += ["-map", f"0:{s['index']}"]
    if info["has_attachments"]:
        cmd += ["-map", "0:t?", "-c:t", "copy"]
    cmd += ["-map_metadata", "0", "-map_chapters", "0"] + video_args(tag)
    cmd += ["-c:a", "aac", "-b:a", AUDIO_BITRATE, "-ac", str(AUDIO_CHANNELS)]
    for i, s in enumerate(info["subs"]):
        cmd += [f"-c:s:{i}", "srt" if s["codec"] in SUBS_TO_SRT else "copy"]
    if info["subs"] and DEFAULT_FIRST_SUBTITLE:
        cmd += ["-disposition:s:0", "default"]
    cover_dir = tempfile.mkdtemp(prefix="covers_")
    cmd += cover_attach_args(dump_covers(source, info, cover_dir), info.get("n_attachments", 0))
    cmd += ["-max_muxing_queue_size", "4096", part]

    print(f"🚀 {tag} {w}x{h} | {describe()}")
    print(" ".join(f'"{c}"' if " " in c else c for c in cmd))
    try:
        rc, lines = run_ffmpeg(cmd)
        shutil.rmtree(cover_dir, ignore_errors=True)
        if rc != 0 or not os.path.exists(part) or os.path.getsize(part) < 1024:
            log = os.path.join(output_dir, f"error_log_{re.sub(r'[^A-Za-z0-9]', '_', base)}.txt")
            with open(log, "w", encoding="utf-8") as f:
                f.write("FFmpeg command: " + " ".join(cmd) + "\n\nOutput:\n")
                f.writelines(lines)
            raise RuntimeError(f"ffmpeg exited with code {rc} (log: {log})")
        os.replace(part, final)
        elapsed = time.time() - start
        print(f"✅ {os.path.basename(final)} - {os.path.getsize(final) / 1048576:.1f} MB "
              f"in {format_time(elapsed)}")
        log_line(output_dir, f"{video} - {tag} {describe()}: {format_time(elapsed)}")
        return True
    except KeyboardInterrupt:
        if os.path.exists(part):
            os.remove(part)
        raise
    except Exception as e:
        if os.path.exists(part):
            os.remove(part)
        print(f"❌ Failed '{video}': {e}")
        log_line(output_dir, f"{video} - {tag}: FAILED - {e}")
        return False


def main():
    print("🎬 LowEncoder - CPU HEVC (libx265), one quality, no watermark")
    for tool in ("ffmpeg", "ffprobe"):
        try:
            subprocess.run([tool, "-version"], check=True, capture_output=True, timeout=15)
        except Exception:
            print(f"❌ {tool} not found")
            sys.exit(1)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, INPUT_FOLDER)
    output_dir = os.path.join(script_dir, OUTPUT_FOLDER)
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    videos = sorted(f for f in os.listdir(input_dir) if is_video_file(f))
    if not videos:
        print(f"❌ No video files in {input_dir}")
        sys.exit(0)
    print(f"📹 Found {len(videos)} video file(s)")

    try:
        enable_crop = input("Remove black bars automatically? (y/n, default y): ").strip().lower() != "n"
        print("\n1) 480p  [default]\n2) 720p\n3) 1080p")
        tag = {"1": "480p", "2": "720p", "3": "1080p"}.get(input("Choose quality (1-3): ").strip(), "480p")
    except EOFError:
        enable_crop, tag = True, "480p"
    print(f"✅ {tag} | {describe()} | crop: {'on' if enable_crop else 'off'}")

    start, ok, failed = time.time(), 0, 0
    try:
        for video in videos:
            if encode_file(video, input_dir, output_dir, tag, enable_crop):
                ok += 1
            else:
                failed += 1
    except KeyboardInterrupt:
        print("\n⏸️  Stopped by user")
        sys.exit(0)
    print(f"\n🎉 Done in {format_time(time.time() - start)} - ✅ {ok}  ❌ {failed}")


if __name__ == "__main__":
    main()
