#!/usr/bin/env python3
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# -------------------------------------------------------------------------------
# Helper: format seconds into H:M:S
# -------------------------------------------------------------------------------
def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}h {m}m {s:.2f}s"

# -------------------------------------------------------------------------------
# Filename generation with quality replacement
# -------------------------------------------------------------------------------
def generate_output_filename(base_name, target_quality):
    if base_name.endswith(('.mkv', '.mp4', '.avi')):
        base_name = os.path.splitext(base_name)[0]
    if base_name.endswith('_Source'):
        base_name = base_name[:-7]

    quality_patterns = [
        r'\b(480p|720p|1080p|4k|2160p)\b',
        r'\b(480P|720P|1080P|4K|2160P)\b'
    ]

    new_name = base_name
    for pattern in quality_patterns:
        if re.search(pattern, new_name, re.IGNORECASE):
            return re.sub(pattern, target_quality, new_name, count=1, flags=re.IGNORECASE)

    year_pattern = r'(\b\d{4}\b)'
    if re.search(year_pattern, new_name):
        return re.sub(year_pattern, f"\\1.{target_quality}", new_name, count=1)
    return f"{target_quality}.{new_name}"

# -------------------------------------------------------------------------------
# Crop detection (HandBrake primary, FFmpeg fallback)
# -------------------------------------------------------------------------------
def detect_crop_handbrake(input_path, video_name):
    print(f"🔍 HandBrake crop detection for '{video_name}'...")
    handbrake_path = r"C:\HandBrakeCLI\HandBrakeCLI.exe"
    cmd = [handbrake_path, '--input', str(input_path), '--scan', '--preview', '30']
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                encoding='utf-8', errors='ignore', timeout=180)
        output = result.stderr if result.stderr else result.stdout
        if not output:
            print("⚠️  No output from HandBrake, trying fallback scan...")
            simple_cmd = [handbrake_path, '-i', str(input_path), '--scan']
            simple_result = subprocess.run(simple_cmd, capture_output=True, text=True,
                                           encoding='utf-8', errors='ignore', timeout=120)
            output = simple_result.stderr if simple_result.stderr else simple_result.stdout
            if not output:
                print("❌ HandBrake scan failed - skipping crop detection")
                return None

        crop_patterns = [
            r'autocrop:\s*(\d+)/(\d+)/(\d+)/(\d+)',
            r'Crop:\s*(\d+):(\d+):(\d+):(\d+)',
            r'crop\s*=\s*(\d+)/(\d+)/(\d+)/(\d+)'
        ]
        crop_match = None
        for pattern in crop_patterns:
            crop_match = re.search(pattern, output, re.IGNORECASE)
            if crop_match:
                break

        if not crop_match:
            print("📺 No black bars detected by HandBrake")
            return {'no_crop_needed': True}

        top, bottom, left, right = map(int, crop_match.groups())
        dim_matches = re.findall(r'(\d+)x(\d+)', output)
        if not dim_matches:
            print("⚠️  Could not determine video dimensions")
            return None

        orig_w, orig_h = map(int, dim_matches[0])
        crop_w = orig_w - left - right
        crop_h = orig_h - top - bottom
        total_cropped = top + bottom + left + right
        if total_cropped <= 10:
            print(f"✨ No significant black bars ({total_cropped}px total)")
            return {'no_crop_needed': True}

        ffmpeg_crop = f"crop={crop_w}:{crop_h}:{left}:{top}"
        print(f"✅ HandBrake crop: {orig_w}x{orig_h} → {crop_w}x{crop_h}")
        return {
            'handbrake_crop': f"{top}/{bottom}/{left}/{right}",
            'ffmpeg_crop': ffmpeg_crop,
            'original_size': f"{orig_w}x{orig_h}",
            'cropped_size': f"{crop_w}x{crop_h}",
            'removed_pixels': {'left': left, 'right': right, 'top': top, 'bottom': bottom}
        }
    except subprocess.TimeoutExpired:
        print("⏰ HandBrake scan timed out - skipping crop detection")
        return None
    except FileNotFoundError:
        print(f"❌ HandBrakeCLI not found at: {handbrake_path}")
        print("   Please install HandBrake CLI and update the path")
        return None
    except Exception as e:
        print(f"❌ HandBrake error: {str(e)}")
        return None

def detect_crop_ffmpeg_fallback(input_path, video_name):
    print(f"🔄 Fallback: FFmpeg crop detection for '{video_name}'...")
    cmd = ['ffmpeg', '-hide_banner', '-i', input_path,
           '-vf', 'cropdetect=limit=24:round=16:reset=0', '-f', 'null', '-t', '30', '-']
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        crop_lines = [line.strip() for line in result.stderr.split('\n') if 'crop=' in line]
        if not crop_lines:
            print("📺 FFmpeg fallback: No crop needed")
            return None
        last_crop = crop_lines[-1]
        m = re.search(r'crop=(\d+):(\d+):(\d+):(\d+)', last_crop)
        if not m:
            print("📺 FFmpeg fallback: No crop needed")
            return None

        width, height, x, y = map(int, m.groups())
        dim_match = re.search(r'Stream.*Video.*?(\d+)x(\d+)', result.stderr)
        if not dim_match:
            return None
        orig_w, orig_h = map(int, dim_match.groups())
        left_crop = x
        right_crop = orig_w - width - left_crop
        top_crop = y
        bottom_crop = orig_h - height - top_crop
        total_cropped = left_crop + right_crop + top_crop + bottom_crop
        if total_cropped <= 10:
            print("📺 FFmpeg fallback: No crop needed")
            return None

        print(f"✅ FFmpeg fallback crop: {orig_w}x{orig_h} → {width}x{height}")
        return {
            'ffmpeg_crop': f"crop={width}:{height}:{x}:{y}",
            'original_size': f"{orig_w}x{orig_h}",
            'cropped_size': f"{width}x{height}",
            'removed_pixels': {'left': left_crop, 'right': right_crop, 'top': top_crop, 'bottom': bottom_crop}
        }
    except Exception as e:
        print(f"❌ FFmpeg fallback failed: {str(e)}")
        return None

def detect_crop_params(input_path, video_name):
    crop_info = detect_crop_handbrake(input_path, video_name)
    if crop_info is None:
        print("🔄 HandBrake failed, trying FFmpeg fallback...")
        crop_info = detect_crop_ffmpeg_fallback(input_path, video_name)
    elif isinstance(crop_info, dict) and crop_info.get('no_crop_needed'):
        return None
    return crop_info

# -------------------------------------------------------------------------------
# Probe dimensions (for exact scale_qsv target width)
# -------------------------------------------------------------------------------
def get_video_dimensions(input_path):
    try:
        cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height',
            '-of', 'csv=p=0:s=x', input_path
        ]
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True, timeout=10)
        w_str, h_str = r.stdout.strip().split('x')
        return int(w_str), int(h_str)
    except Exception:
        return None, None

# -------------------------------------------------------------------------------
# QSV options probing & availability
# -------------------------------------------------------------------------------
def hevc_qsv_supports(option_flag_name: str) -> bool:
    try:
        r = subprocess.run(['ffmpeg', '-hide_banner', '-h', 'encoder=hevc_qsv'],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, timeout=10)
        pattern = rf'^\s*-{re.escape(option_flag_name.lower())}\b'
        return re.search(pattern, r.stdout.lower(), re.MULTILINE) is not None
    except Exception:
        return False

def check_qsv_support():
    try:
        r = subprocess.run(['ffmpeg', '-hide_banner', '-hwaccels'],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10)
        if 'qsv' in r.stdout:
            print("✅ Intel QSV hardware acceleration: Available")
            return True
        print("❌ Intel QSV hardware acceleration: Not available")
        return False
    except Exception:
        print("❌ Could not check QSV support")
        return False

def vpp_qsv_has_crop() -> bool:
    """Check whether current FFmpeg build exposes vpp_qsv crop_* options."""
    try:
        r = subprocess.run(
            ['ffmpeg', '-hide_banner', '-h', 'filter=vpp_qsv'],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=10
        )
        s = r.stdout.lower()
        return all(k in s for k in ('crop_w','crop_h','crop_x','crop_y'))
    except Exception:
        return False

# -------------------------------------------------------------------------------
# Encode config (QUALITY-FIRST; kbps window)
# -------------------------------------------------------------------------------
MODE_480P_QSV = {
    "codec": "hevc_qsv",
    "height": 480,
    "preset": "veryslow",
    "profile": "main",
    "look_ahead": 1,
    "look_ahead_depth": 0,
    "rdo": 1,
    "transform_skip": 1,
    "adaptive_i": 1,
    "adaptive_b": 1,
    "extbrc": 1,
    "mbbrc": 1,
    "refs": 4,
    "g": 240,
    "bf": 7,
    "forced_idr": 1,
    "idr_interval": 1,
    "low_power": 0,
    "force_keyint_seconds": 0,
    "async_depth": 6,
    "extra_hw_frames": 64,
    # --- kbps, not KBps ---
    "min_kbps": 300,
    "avg_kbps": 350,
    "max_kbps": 500,
}

MINRATE = f"{MODE_480P_QSV['min_kbps']}k"
AVGRATE = f"{MODE_480P_QSV['avg_kbps']}k"
MAXRATE = f"{MODE_480P_QSV['max_kbps']}k"
BUFSIZE = f"{MODE_480P_QSV['max_kbps'] * 2}k"

MODE_720P_QSV = dict(MODE_480P_QSV)
MODE_720P_QSV.update({
    "height": 720,
    "min_kbps": 550,
    "avg_kbps": 700,
    "max_kbps": 900,
})

MODE_1080P_QSV = dict(MODE_480P_QSV)
MODE_1080P_QSV.update({
    "height": 1080,
    "min_kbps": 1100,
    "avg_kbps": 1450,
    "max_kbps": 1750,
})

ENCODE_OPTIONS = {
    "1": {"label": "480p only", "qualities": ["480p"], "parallel": False},
    "2": {"label": "720p only", "qualities": ["720p"], "parallel": False},
    "3": {"label": "1080p only", "qualities": ["1080p"], "parallel": False},
    "4": {"label": "720p and 1080p", "qualities": ["720p", "1080p"], "parallel": False},
    "5": {"label": "All 3 qualities one by one", "qualities": ["480p", "720p", "1080p"], "parallel": False},
    "6": {"label": "All 3 qualities parallel", "qualities": ["480p", "720p", "1080p"], "parallel": True},
}

DEFAULT_WATERMARK_NAME = "watermark.png"

# -------------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------------
def is_video_file(filename):
    VIDEO_EXTENSIONS = ['.mp4', '.mkv', '.avi', '.mov', '.flv', '.wmv', '.mpeg', '.mpg', '.m4v', '.3gp', '.webm']
    return any(filename.lower().endswith(ext) for ext in VIDEO_EXTENSIONS)

def verify_input_file(input_path):
    try:
        subprocess.run(['ffprobe', '-v', 'error', '-show_format', '-show_streams', input_path],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=30)
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False

def has_stream(input_path, kind):  # kind: 'a' for audio, 's' for subs
    try:
        r = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', kind, '-show_entries', 'stream=index',
                            '-of', 'csv=p=0', input_path],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True, timeout=10)
        return r.stdout.strip() != ""
    except Exception:
        return False

# -------------------------------------------------------------------------------
# Parse 'crop=W:H:X:Y' into ints for vpp_qsv
# -------------------------------------------------------------------------------
def parse_ffmpeg_crop(crop_str):
    m = re.search(r'crop=(\d+):(\d+):(\d+):(\d+)', crop_str or '')
    if not m:
        return None
    w, h, x, y = map(int, m.groups())
    return {"w": w, "h": h, "x": x, "y": y}

# -------------------------------------------------------------------------------
# QSV 480p HEVC encode (GPU crop+scale when supported)
# -------------------------------------------------------------------------------
def convert_480p_qsv(input_path, output_path, video_name, output_dir, enable_crop=True):
    start_time = time.time()
    print(f"\n🎬 Starting 480p H.265/QSV for '{video_name}' ...")
    print(f"🎯 BRC window: min={MINRATE}, avg={AVGRATE}, max={MAXRATE}, buf={BUFSIZE}")
    print(f"🧰 Preset: {MODE_480P_QSV['preset']} | Profile: {MODE_480P_QSV['profile']}")
    print(f"⚡ QSV: full-feature path (low_power={MODE_480P_QSV['low_power']})")
    print(f"🧵 Pipeline depth: async_depth={MODE_480P_QSV['async_depth']} extra_hw_frames={MODE_480P_QSV['extra_hw_frames']}")

    # Crop detection
    crop_info = detect_crop_params(input_path, video_name) if enable_crop else None
    crop_box = parse_ffmpeg_crop(crop_info['ffmpeg_crop']) if (crop_info and 'ffmpeg_crop' in crop_info) else None

    # Choose dimension basis (cropped vs original) for AR-correct width
    in_w, in_h = get_video_dimensions(input_path)
    if crop_box:
        src_w, src_h = crop_box['w'], crop_box['h']
    else:
        src_w, src_h = (in_w or 854), (in_h or 480)

    if src_h > 0:
        target_w = max(2, int((src_w * 480) // src_h))
        if target_w % 2:
            target_w -= 1
    else:
        target_w = 854
    print(f"🖼️  Target scale: {target_w}x480 (AR from {'cropped' if crop_box else 'original'} size)")

    # Build filter graph with smart fallback
    qsv_can_crop = vpp_qsv_has_crop()

    if crop_box and qsv_can_crop:
        filter_complex = (
            f"[0:v]format=nv12,hwupload=extra_hw_frames={MODE_480P_QSV['extra_hw_frames']},"
            f"vpp_qsv=crop_w={crop_box['w']}:crop_h={crop_box['h']}:"
            f"crop_x={crop_box['x']}:crop_y={crop_box['y']},"
            f"scale_qsv=w={target_w}:h=480,setsar=1[outv]"
        )
        print("🔧 Using GPU crop (vpp_qsv) + GPU scale (scale_qsv)")
    elif crop_box and not qsv_can_crop:
        print("⚠️  vpp_qsv crop_* not available in this FFmpeg build → CPU crop → hwupload → GPU scale")
        filter_complex = (
            f"[0:v]crop={crop_box['w']}:{crop_box['h']}:{crop_box['x']}:{crop_box['y']},"
            f"format=nv12,hwupload=extra_hw_frames={MODE_480P_QSV['extra_hw_frames']},"
            f"scale_qsv=w={target_w}:h=480,setsar=1[outv]"
        )
    else:
        filter_complex = (
            f"[0:v]format=nv12,hwupload=extra_hw_frames={MODE_480P_QSV['extra_hw_frames']},"
            f"scale_qsv=w={target_w}:h=480,setsar=1[outv]"
        )
        print("🔧 No crop needed → GPU scale only (scale_qsv)")

    # Assemble command
    cmd = ["ffmpeg", "-y",
           "-extra_hw_frames", str(MODE_480P_QSV["extra_hw_frames"]),
           "-init_hw_device", "qsv=hw", "-filter_hw_device", "hw",
           "-i", input_path,
           "-map_metadata", "0", "-map_chapters", "0",
           "-filter_complex", filter_complex,
           "-map", "[outv]", "-map", "0:a?", "-map", "0:s?"]

    # Video encode settings (QSV bitrate window in kbps)
    cmd.extend([
        "-c:v", MODE_480P_QSV["codec"],
        "-profile:v", MODE_480P_QSV["profile"],
        "-preset", MODE_480P_QSV["preset"],
        "-b:v", AVGRATE,
        "-maxrate", MAXRATE,
        "-minrate", MINRATE,
        "-bufsize", BUFSIZE,
        "-look_ahead", str(MODE_480P_QSV["look_ahead"]),
        "-extbrc", str(MODE_480P_QSV["extbrc"]),
        "-rdo", str(MODE_480P_QSV["rdo"]),
        "-transform_skip", str(MODE_480P_QSV["transform_skip"]),
        "-adaptive_i", str(MODE_480P_QSV["adaptive_i"]),
        "-adaptive_b", str(MODE_480P_QSV["adaptive_b"]),
        "-mbbrc", str(MODE_480P_QSV["mbbrc"]),
        "-refs", str(MODE_480P_QSV["refs"]),
        "-g", str(MODE_480P_QSV["g"]),
        "-bf", str(MODE_480P_QSV["bf"]),
        "-forced_idr", str(MODE_480P_QSV["forced_idr"]),
        "-idr_interval", str(MODE_480P_QSV["idr_interval"]),
        "-low_power", str(MODE_480P_QSV["low_power"]),
        "-async_depth", str(MODE_480P_QSV["async_depth"]),
    ])

    if MODE_480P_QSV["look_ahead_depth"] > 0 and hevc_qsv_supports("look_ahead_depth"):
        cmd.extend(["-look_ahead_depth", str(MODE_480P_QSV["look_ahead_depth"])])

    # Optional GPB
    if hevc_qsv_supports("gpb"):
        cmd.extend(["-gpb", "1"])
        print("✅ GPB frames: enabled (-gpb 1)")

    print(f"🎬 B-frames: {MODE_480P_QSV['bf']} | GOP: {MODE_480P_QSV['g']} | Refs: {MODE_480P_QSV['refs']}")

    # Audio/subs
    if has_stream(input_path, 'a'):
        cmd.extend(["-c:a", "aac", "-b:a", "64k", "-ac", "2"])
    cmd.extend(["-c:s", "copy"])
    if has_stream(input_path, 's'):
        cmd.extend(["-disposition:s:0", "default"])

    cmd.append(output_path)

    print("Executing 480p H.265 QSV command:\n", " ".join(cmd))
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   universal_newlines=True, encoding="utf-8", errors="replace")
        output_lines = []
        error_patterns = [
            'Error while opening encoder', 'Conversion failed', 'Invalid data found',
            'No such file or directory', 'Permission denied', 'Unrecognized option',
            'Cannot load Intel QSV', 'No QSV device found'
        ]
        for line in process.stdout:
            print(line, end=""); output_lines.append(line)
            for pat in error_patterns:
                if pat.lower() in line.lower():
                    print(f"❌ Critical error detected: {pat}")
                    process.terminate()
                    raise RuntimeError(f"FFmpeg error: {pat}")

        rc = process.wait()
        if rc != 0:
            print(f"❌ FFmpeg process failed with return code: {rc}")
            errlog = os.path.join(output_dir, f"error_log_{video_name}_480p_qsv.txt")
            with open(errlog, "w", encoding="utf-8") as f:
                f.write(f"FFmpeg command: {' '.join(cmd)}\n\nOutput:\n")
                f.writelines(output_lines)
            print(f"📝 Error log saved to: {errlog}")
            raise RuntimeError(f"FFmpeg encoding failed ({rc})")

        if not os.path.exists(output_path):
            raise RuntimeError(f"Output file was not created: {output_path}")

        output_size = os.path.getsize(output_path)
        if output_size < 1024:
            raise RuntimeError(f"Output file too small ({output_size} bytes)")

        elapsed = time.time() - start_time
        print(f"✅ 480p HEVC QSV completed for '{video_name}' in {format_time(elapsed)}.")
        print(f"📁 Output size: {output_size / (1024*1024):.2f} MB")

        with open(os.path.join(output_dir, "conversiontime.txt"), "a", encoding="utf-8") as f:
            crop_note = ""
            if crop_info and 'original_size' in crop_info and 'cropped_size' in crop_info:
                crop_note = f" (cropped {crop_info['original_size']} → {crop_info['cropped_size']})"
            f.write(f"{video_name} - 480p HEVC QSV{crop_note}: {format_time(elapsed)} "
                    f"- window {MINRATE}/{AVGRATE}/{MAXRATE}\n")
        return True

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ 480p H.265 QSV failed for '{video_name}' after {format_time(elapsed)}: {str(e)}")
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
                print(f"🗑️  Removed partial output: {output_path}")
            except:
                pass
        with open(os.path.join(output_dir, "conversiontime.txt"), "a", encoding="utf-8") as f:
            f.write(f"{video_name} - 480p HEVC QSV: FAILED after {format_time(elapsed)} - Error: {str(e)}\n")
        return False

# -------------------------------------------------------------------------------
# Shared multi-quality helpers
# -------------------------------------------------------------------------------
def get_default_width(target_height):
    default_widths = {
        480: 854,
        720: 1280,
        1080: 1920,
    }
    return default_widths.get(target_height, max(2, (target_height * 16) // 9))

def get_rate_window(mode_config):
    minrate = f"{mode_config['min_kbps']}k"
    avgrate = f"{mode_config['avg_kbps']}k"
    maxrate = f"{mode_config['max_kbps']}k"
    bufsize = f"{mode_config['max_kbps'] * 2}k"
    return minrate, avgrate, maxrate, bufsize

def get_auto_watermark_path(script_dir):
    watermark_path = os.path.join(script_dir, DEFAULT_WATERMARK_NAME)
    if os.path.isfile(watermark_path):
        print(f"✅ Watermark: ENABLED ({watermark_path})")
        return watermark_path

    print(f"✅ Watermark: DISABLED ({DEFAULT_WATERMARK_NAME} not found in script folder)")
    return None


def build_output_path(output_dir, video_name, quality_name):
    base, _ = os.path.splitext(video_name)
    out_name = generate_output_filename(base, quality_name) + "_qsv.mkv"
    return os.path.join(output_dir, out_name)

def describe_quality_windows(quality_names):
    quality_map = {
        "480p": MODE_480P_QSV,
        "720p": MODE_720P_QSV,
        "1080p": MODE_1080P_QSV,
    }
    return ", ".join(
        f"{quality_name}: {quality_map[quality_name]['min_kbps']}/{quality_map[quality_name]['avg_kbps']}/{quality_map[quality_name]['max_kbps']}k"
        for quality_name in quality_names
    )

def convert_quality_qsv(input_path, output_path, video_name, output_dir, quality_name, mode_config,
                        enable_crop=True, watermark_path=None, crop_info_cache=None):
    minrate, avgrate, maxrate, bufsize = get_rate_window(mode_config)
    target_height = mode_config["height"]
    start_time = time.time()

    print(f"\nStarting {quality_name} H.265/QSV for '{video_name}' ...")
    print(f"BRC window: min={minrate}, avg={avgrate}, max={maxrate}, buf={bufsize}")
    print(f"Preset: {mode_config['preset']} | Profile: {mode_config['profile']}")
    print(f"QSV full-feature path (low_power={mode_config['low_power']})")
    print(f"Pipeline depth: async_depth={mode_config['async_depth']} extra_hw_frames={mode_config['extra_hw_frames']}")

    # Use pre-computed crop info if provided (avoids re-running slow crop detection for every quality)
    if crop_info_cache is not None:
        crop_info = crop_info_cache
    elif enable_crop:
        crop_info = detect_crop_params(input_path, video_name)
    else:
        crop_info = None
    crop_box = parse_ffmpeg_crop(crop_info['ffmpeg_crop']) if (crop_info and 'ffmpeg_crop' in crop_info) else None

    in_w, in_h = get_video_dimensions(input_path)
    default_width = get_default_width(target_height)
    if crop_box:
        src_w, src_h = crop_box['w'], crop_box['h']
    else:
        src_w, src_h = (in_w or default_width), (in_h or target_height)

    if src_h > 0:
        target_w = max(2, int((src_w * target_height) // src_h))
        if target_w % 2:
            target_w -= 1
    else:
        target_w = default_width
    print(f"Target scale: {target_w}x{target_height} (AR from {'cropped' if crop_box else 'original'} size)")

    # ── Filter chain strategy ───────────────────────────────────────────────────
    # NO watermark: pure CPU filter chain → format=nv12 → hevc_qsv uploads
    #               internally.  No -init_hw_device needed.
    # WITH watermark: CPU crop/scale/overlay → format=nv12 → explicit hwupload
    #               → scale_qsv passthrough → hevc_qsv.  Requires
    #               -init_hw_device qsv=hw / -filter_hw_device hw so the
    #               hwupload and scale_qsv nodes can reach the QSV device.
    #               Omitting hwupload when a second input exists causes
    #               'Conversion failed' on Intel QSV drivers.
    # ───────────────────────────────────────────────────────────────────────────
    wm_h = 20  # fixed 20px tall watermark (matches waterMovies.py)
    if watermark_path:
        crop_prefix = ""
        if crop_box:
            crop_prefix = f"crop={crop_box['w']}:{crop_box['h']}:{crop_box['x']}:{crop_box['y']},"
        # CPU: crop → scale → overlay → nv12 → hwupload → scale_qsv (passthrough)
        # Explicit hwupload is required — hevc_qsv cannot auto-upload when a
        # watermark (second input) is present and -filter_hw_device is active.
        filter_complex = (
            f"[0:v]{crop_prefix}scale={target_w}:{target_height}:flags=lanczos,setsar=1[base];"
            f"[1:v]scale=-1:{wm_h}:flags=lanczos[wm];"
            f"[base][wm]overlay=W-w-2:H-h-2,"
            f"format=nv12,hwupload=extra_hw_frames={mode_config['extra_hw_frames']},"
            f"scale_qsv=w={target_w}:h={target_height}[outv]"
        )
        print(f"Watermark: {wm_h}px tall, bottom-right corner (2px margin) → hwupload → scale_qsv → hevc_qsv")
    elif crop_box:
        # CPU crop + CPU scale → nv12, encoder uploads to GPU internally.
        filter_complex = (
            f"[0:v]crop={crop_box['w']}:{crop_box['h']}:{crop_box['x']}:{crop_box['y']},"
            f"scale={target_w}:{target_height}:flags=lanczos,setsar=1,format=nv12[outv]"
        )
        print("Crop + scale (CPU) → nv12 → hevc_qsv (GPU upload internal)")
    else:
        # CPU scale → nv12, encoder uploads to GPU internally.
        filter_complex = (
            f"[0:v]scale={target_w}:{target_height}:flags=lanczos,setsar=1,format=nv12[outv]"
        )
        print("Scale (CPU) → nv12 → hevc_qsv (GPU upload internal)")

    # When a watermark is active we need explicit QSV device context so that
    # hwupload and scale_qsv work correctly.  For no-watermark paths the
    # encoder handles GPU upload internally without the global device flags.
    if watermark_path:
        cmd = ["ffmpeg", "-y",
               "-init_hw_device", "qsv=hw", "-filter_hw_device", "hw",
               "-i", input_path, "-i", watermark_path]
    else:
        cmd = ["ffmpeg", "-y", "-i", input_path]
    cmd.extend([
           "-map_metadata", "0", "-map_chapters", "0",
           "-filter_complex", filter_complex,
           "-map", "[outv]", "-map", "0:a?", "-map", "0:s?"
    ])

    cmd.extend([
        "-c:v", mode_config["codec"],
        "-profile:v", mode_config["profile"],
        "-preset", mode_config["preset"],
        "-b:v", avgrate,
        "-maxrate", maxrate,
        "-minrate", minrate,
        "-bufsize", bufsize,
        "-look_ahead", str(mode_config["look_ahead"]),
        "-extbrc", str(mode_config["extbrc"]),
        "-rdo", str(mode_config["rdo"]),
        "-transform_skip", str(mode_config["transform_skip"]),
        "-adaptive_i", str(mode_config["adaptive_i"]),
        "-adaptive_b", str(mode_config["adaptive_b"]),
        "-mbbrc", str(mode_config["mbbrc"]),
        "-refs", str(mode_config["refs"]),
        "-g", str(mode_config["g"]),
        "-bf", str(mode_config["bf"]),
        "-forced_idr", str(mode_config["forced_idr"]),
        "-idr_interval", str(mode_config["idr_interval"]),
        "-low_power", str(mode_config["low_power"]),
        "-async_depth", str(mode_config["async_depth"]),
    ])

    if mode_config["look_ahead_depth"] > 0 and hevc_qsv_supports("look_ahead_depth"):
        cmd.extend(["-look_ahead_depth", str(mode_config["look_ahead_depth"])])

    if hevc_qsv_supports("gpb"):
        cmd.extend(["-gpb", "1"])
        print("GPB frames enabled (-gpb 1)")

    print(f"B-frames: {mode_config['bf']} | GOP: {mode_config['g']} | Refs: {mode_config['refs']}")

    if has_stream(input_path, 'a'):
        cmd.extend(["-c:a", "aac", "-b:a", "64k", "-ac", "2"])
    cmd.extend(["-c:s", "copy"])
    if has_stream(input_path, 's'):
        cmd.extend(["-disposition:s:0", "default"])

    cmd.append(output_path)

    print(f"Executing {quality_name} H.265 QSV command:\n", " ".join(cmd))
    # Track whether FFmpeg itself completed the encode so we never delete a
    # successfully-produced output file just because a post-encode step throws.
    encode_succeeded = False
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   universal_newlines=True, encoding="utf-8", errors="replace")
        output_lines = []
        error_patterns = [
            'Error while opening encoder', 'Conversion failed', 'Invalid data found',
            'No such file or directory', 'Permission denied', 'Unrecognized option',
            'Cannot load Intel QSV', 'No QSV device found'
        ]
        for line in process.stdout:
            print(line, end="")
            output_lines.append(line)
            for pat in error_patterns:
                if pat.lower() in line.lower():
                    print(f"Critical error detected: {pat}")
                    process.terminate()
                    raise RuntimeError(f"FFmpeg error: {pat}")

        rc = process.wait()
        if rc != 0:
            print(f"FFmpeg process failed with return code: {rc}")
            errlog = os.path.join(output_dir, f"error_log_{video_name}_{quality_name}_qsv.txt")
            with open(errlog, "w", encoding="utf-8") as f:
                f.write(f"FFmpeg command: {' '.join(cmd)}\n\nOutput:\n")
                f.writelines(output_lines)
            print(f"Error log saved to: {errlog}")
            raise RuntimeError(f"FFmpeg encoding failed ({rc})")

        if not os.path.exists(output_path):
            raise RuntimeError(f"Output file was not created: {output_path}")

        output_size = os.path.getsize(output_path)
        if output_size < 1024:
            raise RuntimeError(f"Output file too small ({output_size} bytes)")

        # ── FFmpeg encode is complete and verified. Mark as succeeded. ──
        # Any exception beyond this point must NOT delete the output file.
        encode_succeeded = True

        elapsed = time.time() - start_time
        print(f"✅ {quality_name} HEVC QSV completed for '{video_name}' in {format_time(elapsed)}.")
        print(f"📁 Output: {output_path}")
        print(f"📦 Output size: {output_size / (1024 * 1024):.2f} MB")

        try:
            with open(os.path.join(output_dir, "conversiontime.txt"), "a", encoding="utf-8") as f:
                crop_note = ""
                if crop_info and 'original_size' in crop_info and 'cropped_size' in crop_info:
                    crop_note = f" (cropped {crop_info['original_size']} -> {crop_info['cropped_size']})"
                wm_note = " + watermark" if watermark_path else ""
                f.write(f"{video_name} - {quality_name} HEVC QSV{crop_note}{wm_note}: {format_time(elapsed)} "
                        f"- window {minrate}/{avgrate}/{maxrate}\n")
        except Exception as log_err:
            print(f"⚠️  Could not write conversiontime.txt (encode still succeeded): {log_err}")

        return True

    except Exception as e:
        elapsed = time.time() - start_time
        if encode_succeeded:
            # FFmpeg finished cleanly — a post-encode step threw (e.g. logging).
            # Do NOT touch the output file; it is complete and valid.
            print(f"⚠️  Post-encode step failed for '{video_name}' [{quality_name}] (output file is SAFE): {e}")
            print(f"📁 Output preserved at: {output_path}")
            try:
                with open(os.path.join(output_dir, "conversiontime.txt"), "a", encoding="utf-8") as f:
                    wm_note = " + watermark" if watermark_path else ""
                    f.write(f"{video_name} - {quality_name} HEVC QSV{wm_note}: completed in {format_time(elapsed)} "
                            f"[post-encode warning: {e}]\n")
            except Exception:
                pass
            return True   # encode DID succeed; do not block the next quality
        else:
            # Genuine encode failure — clean up any partial output.
            print(f"❌ {quality_name} H.265 QSV FAILED for '{video_name}' after {format_time(elapsed)}: {e}")
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                    print(f"🗑️  Removed partial output: {output_path}")
                except Exception:
                    pass
            try:
                with open(os.path.join(output_dir, "conversiontime.txt"), "a", encoding="utf-8") as f:
                    wm_note = " + watermark" if watermark_path else ""
                    f.write(f"{video_name} - {quality_name} HEVC QSV{wm_note}: FAILED after {format_time(elapsed)} - Error: {e}\n")
            except Exception:
                pass
            return False

def convert_720p_qsv(input_path, output_path, video_name, output_dir, enable_crop=True, watermark_path=None):
    return convert_quality_qsv(input_path, output_path, video_name, output_dir, "720p", MODE_720P_QSV,
                               enable_crop, watermark_path)

def convert_1080p_qsv(input_path, output_path, video_name, output_dir, enable_crop=True, watermark_path=None):
    return convert_quality_qsv(input_path, output_path, video_name, output_dir, "1080p", MODE_1080P_QSV,
                               enable_crop, watermark_path)

def run_encode_job(job, output_dir, enable_crop, watermark_path=None):
    quality_map = {
        "480p": MODE_480P_QSV,
        "720p": MODE_720P_QSV,
        "1080p": MODE_1080P_QSV,
    }
    quality_name = job["quality_name"]
    if quality_name not in quality_map:
        raise ValueError(f"Unsupported quality: {quality_name}")
    # Pass pre-computed crop_info if available in the job dict (avoids re-running crop per quality)
    crop_info_cache = job.get("crop_info_cache", None)
    return convert_quality_qsv(job["input_path"], job["output_path"], job["video_name"], output_dir,
                               quality_name, quality_map[quality_name], enable_crop, watermark_path,
                               crop_info_cache=crop_info_cache)

# -------------------------------------------------------------------------------
# System requirements
# -------------------------------------------------------------------------------
def check_system_requirements():
    print("🔧 Checking system requirements...")
    try:
        subprocess.run(['ffmpeg', '-version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
        print("✅ FFmpeg: Available")
    except Exception:
        print("❌ FFmpeg: Not found"); return False

    try:
        subprocess.run(['ffprobe', '-version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
        print("✅ FFprobe: Available")
    except Exception:
        print("❌ FFprobe: Not found"); return False

    if not check_qsv_support():
        print("❌ Intel QSV required but not available")
        print("   Ensure: 13th-gen i5 iGPU enabled, latest Intel driver, FFmpeg with QSV")
        return False

    handbrake_path = r"C:\HandBrakeCLI\HandBrakeCLI.exe"
    try:
        subprocess.run([handbrake_path, '--version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
        print("✅ HandBrakeCLI: Available (primary crop detector)")
    except Exception:
        print("⚠️  HandBrakeCLI not found (will use FFmpeg fallback):", handbrake_path)
        print("   Download: https://handbrake.fr/downloads.php")
    print()
    return True

# -------------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------------
def main():
    print("🎬 Dedicated HEVC QSV multi-quality encode")
    print("📐 GPU crop + GPU scale (when supported) or CPU crop -> hwupload fallback")
    print(f"🎯 Available presets: {describe_quality_windows(['480p', '720p', '1080p'])}")
    if not check_system_requirements():
        sys.exit(1)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, "Encode_Input")
    output_dir = os.path.join(script_dir, "Encode_Output")

    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    print(f"📁 Using input folder: {input_dir}")
    print(f"📁 Using output folder: {output_dir}\n")

    video_files = [f for f in os.listdir(input_dir) if is_video_file(f)]
    if not video_files:
        print("❌ No video files found in Encode_Input.")
        sys.exit(0)
    print(f"📹 Found {len(video_files)} file(s).\n")

    print("🎯 Hybrid Crop Detection: HandBrake -> FFmpeg fallback")
    try:
        crop_choice = input("Enable automatic black bar removal? (y/n, default: y): ").strip().lower()
        enable_crop = crop_choice != 'n'
    except:
        enable_crop = True
    print("✅ Auto-crop:", "ENABLED" if enable_crop else "DISABLED")

    watermark_path = get_auto_watermark_path(script_dir)

    print("\nSelect encode mode:")
    print("1) 480p only")
    print("2) 720p only")
    print("3) 1080p only")
    print("4) 720p and 1080p")
    print("5) All 3 qualities one by one")
    print("6) All 3 qualities parallel")
    try:
        encode_choice = input("Choose mode (1-6, default: 1): ").strip()
    except:
        encode_choice = "1"
    if encode_choice not in ENCODE_OPTIONS:
        print("⚠️  Invalid selection. Defaulting to 1) 480p only.")
        encode_choice = "1"

    selected_option = ENCODE_OPTIONS[encode_choice]
    selected_qualities = selected_option["qualities"]
    use_parallel = selected_option["parallel"]
    print(f"✅ Selected mode: {selected_option['label']}")
    print(f"🎯 Selected windows: {describe_quality_windows(selected_qualities)}")

    total_ok, total_fail = 0, 0
    jobs = []
    for v in video_files:
        in_path = os.path.join(input_dir, v)
        if not verify_input_file(in_path):
            print(f"❌ Skipping '{v}': input verify failed.")
            total_fail += len(selected_qualities)
            continue

        # ── Detect crop ONCE per input file, then reuse across all qualities ──
        if enable_crop:
            print(f"\n🔍 Crop detection for '{v}' (runs once for all qualities)...")
            crop_info_cache = detect_crop_params(in_path, v)
            if crop_info_cache and 'ffmpeg_crop' in crop_info_cache:
                print(f"✅ Crop cached: {crop_info_cache['ffmpeg_crop']}")
            else:
                print("📺 No crop needed (or detection failed). Proceeding without crop.")
                crop_info_cache = None
        else:
            crop_info_cache = None

        # ── Check for output filename collisions ──
        seen_out_paths = {}
        for quality_name in selected_qualities:
            out_path = build_output_path(output_dir, v, quality_name)
            if out_path in seen_out_paths:
                print(f"⚠️  WARNING: '{quality_name}' and '{seen_out_paths[out_path]}' would produce the SAME output filename!")
                print(f"   → {out_path}")
                print(f"   The input filename may not contain a quality tag (e.g. 480p/720p/1080p).")
                print(f"   Rename the input file to include quality info, e.g. Movie.2024.1080p.mkv")
            seen_out_paths[out_path] = quality_name

        for quality_name in selected_qualities:
            out_path = build_output_path(output_dir, v, quality_name)
            if os.path.exists(out_path):
                print(f"⏭️  Output exists. Skipping: {out_path}")
                total_ok += 1
                continue
            jobs.append({
                "video_name": v,
                "quality_name": quality_name,
                "input_path": in_path,
                "output_path": out_path,
                "crop_info_cache": crop_info_cache,  # pre-computed, no re-detection per quality
            })

    overall_start = time.time()

    try:
        if not jobs:
            print("\n⏭️  No new encode jobs queued.")
        elif use_parallel:
            max_workers = min(len(jobs), len(selected_qualities))
            print(f"\n🚀 Parallel encoding for {len(jobs)} job(s) (max {max_workers} workers)...")
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = {}
                for job in jobs:
                    futures[ex.submit(run_encode_job, job, output_dir, enable_crop, watermark_path)] = job
                for fut in as_completed(futures):
                    job = futures[fut]
                    try:
                        if fut.result():
                            total_ok += 1
                        else:
                            total_fail += 1
                    except Exception as e:
                        print(f"❌ Exception while processing '{job['video_name']}' [{job['quality_name']}]: {e}")
                        total_fail += 1
        else:
            print(f"\n🎬 Sequential encoding for {len(jobs)} job(s)...")
            for job in jobs:
                ok = run_encode_job(job, output_dir, enable_crop, watermark_path)
                total_ok += 1 if ok else 0
                total_fail += 0 if ok else 1

        elapsed = time.time() - overall_start
        print("\n🎉 ========== BATCH COMPLETE ==========")
        print(f"⏱️  Total time: {format_time(elapsed)}")
        print(f"✅ Success: {total_ok}  ❌ Failed: {total_fail}")
        print(f"⚡ Mode: {selected_option['label']} | QSV full-feature; veryslow; bitrate window")
        with open(os.path.join(output_dir, "conversiontime.txt"), "a", encoding="utf-8") as f:
            f.write(f"\nBATCH [{selected_option['label']}] {format_time(elapsed)} - "
                    f"Success: {total_ok}, Fail: {total_fail} - "
                    f"Windows: {describe_quality_windows(selected_qualities)}\n")
    except KeyboardInterrupt:
        print("\n⏸️  Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
