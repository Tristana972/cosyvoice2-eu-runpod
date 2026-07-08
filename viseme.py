"""
Procedural viseme (mouth-swap) animator for stylized avatars (Zuzu, Titu)
that neural lip-sync models (MuseTalk, etc.) can't handle because they lack
human facial topology.

Ported from the standalone viseme_animate.py script, adapted to run inside
the cosyvoice2-eu-runpod worker with the character mouth images bundled in
/content/assets/.

Also supports a "duo" mode: two characters visible together in the same
frame over an AI-generated background, taking turns speaking (video-call
style bubbles), for the same-frame Zuzu/Titu interaction feature.
"""
import wave
import os
import subprocess
import numpy as np
from PIL import Image, ImageDraw

FPS = 25
CLOSED_THRESH = 0.12
OPEN_THRESH = 0.45

ASSETS_DIR = "/content/assets"

CHARACTER_IMAGES = {
    "zuzu": {
        "closed": os.path.join(ASSETS_DIR, "zuzu_mouth_closed.png"),
        "mid": os.path.join(ASSETS_DIR, "zuzu_mouth_mid.png"),
        "open": os.path.join(ASSETS_DIR, "zuzu_mouth_open.png"),
    },
    "titu": {
        "closed": os.path.join(ASSETS_DIR, "titu_mouth_closed.png"),
        "mid": os.path.join(ASSETS_DIR, "titu_mouth_mid.png"),
        "open": os.path.join(ASSETS_DIR, "titu_mouth_open.png"),
    },
}

# Layout for the duo (same-frame) mode: each character sits in a round
# bubble over the generated background, side by side like a video call.
DUO_CANVAS = (720, 1280)
DUO_BUBBLE_RADIUS = 150
DUO_POSITIONS = {
    "zuzu": (210, 620),
    "titu": (510, 620),
}
DUO_RING_COLOR = (34, 211, 238)  # cyan, matches the app's accent color
DUO_RING_WIDTH = 10


def _read_wav_rms_states(audio_path):
    """Read a mono/stereo 16-bit WAV file and return one mouth state
    ('closed'/'mid'/'open') per video frame at FPS, RMS-based with
    temporal smoothing. Also returns the audio duration in seconds."""
    wf = wave.open(audio_path, 'rb')
    n_channels = wf.getnchannels()
    framerate = wf.getframerate()
    n_frames = wf.getnframes()
    raw = wf.readframes(n_frames)
    wf.close()

    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)

    duration = n_frames / framerate
    total_video_frames = max(1, int(duration * FPS))
    samples_per_frame = max(1, int(framerate / FPS))

    rms_values = []
    for i in range(total_video_frames):
        start = i * samples_per_frame
        end = start + samples_per_frame
        chunk = audio[start:end]
        if len(chunk) == 0:
            rms_values.append(0.0)
            continue
        rms = np.sqrt(np.mean(chunk.astype(np.float64) ** 2))
        rms_values.append(rms)

    rms_values = np.array(rms_values)
    max_rms = rms_values.max() if rms_values.max() > 0 else 1.0
    norm = rms_values / max_rms

    states = []
    for v in norm:
        if v < CLOSED_THRESH:
            states.append("closed")
        elif v < OPEN_THRESH:
            states.append("mid")
        else:
            states.append("open")

    smoothed = list(states)
    window = 2
    for i in range(len(states)):
        lo = max(0, i - window)
        hi = min(len(states), i + window + 1)
        window_states = states[lo:hi]
        smoothed[i] = max(set(window_states), key=window_states.count)

    return smoothed, duration


def animate(audio_path, character, out_video_path, frames_dir):
    if character not in CHARACTER_IMAGES:
        raise ValueError(
            "Unknown character '%s', expected one of %s" % (character, list(CHARACTER_IMAGES))
        )

    imgs = CHARACTER_IMAGES[character]
    os.makedirs(frames_dir, exist_ok=True)

    smoothed, _duration = _read_wav_rms_states(audio_path)

    closed_img = Image.open(imgs["closed"]).convert("RGB")
    mid_img = Image.open(imgs["mid"]).convert("RGB")
    open_img = Image.open(imgs["open"]).convert("RGB")
    state_to_img = {"closed": closed_img, "mid": mid_img, "open": open_img}

    for i, st in enumerate(smoothed):
        state_to_img[st].save(os.path.join(frames_dir, "frame_%05d.png" % i))

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(FPS),
        "-i", os.path.join(frames_dir, "frame_%05d.png"),
        "-i", audio_path,
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-shortest",
        out_video_path
    ]
    subprocess.run(cmd, check=True)
    return out_video_path


def _make_bubble(char_img_path, radius):
    """Center-crop a character image to a square and mask it into a circle,
    so it can be composited onto a different background without needing a
    true transparent cutout of the original photo."""
    img = Image.open(char_img_path).convert("RGBA")
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    img = img.resize((radius * 2, radius * 2), Image.LANCZOS)

    mask = Image.new("L", (radius * 2, radius * 2), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, radius * 2, radius * 2), fill=255)
    img.putalpha(mask)
    return img


def animate_duo(turns, background_path, out_video_path, frames_dir):
    """
    turns: list of dicts, each { "character": "zuzu"|"titu", "audio_path": "..." },
        in the order they speak.
    background_path: path to a generated background image (any size, will be
        cover-cropped to fill the canvas).

    Produces one video where both characters are visible for the whole
    duration, side by side over the background (video-call style bubbles),
    with a highlight ring around whichever one is currently speaking and
    only that character's mouth animating (the other stays closed).
    """
    os.makedirs(frames_dir, exist_ok=True)
    canvas_w, canvas_h = DUO_CANVAS

    # Background: cover-fit crop to the canvas size.
    bg = Image.open(background_path).convert("RGB")
    bg_ratio = bg.width / bg.height
    canvas_ratio = canvas_w / canvas_h
    if bg_ratio > canvas_ratio:
        new_h = canvas_h
        new_w = int(bg_ratio * new_h)
    else:
        new_w = canvas_w
        new_h = int(new_w / bg_ratio)
    bg = bg.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - canvas_w) // 2
    top = (new_h - canvas_h) // 2
    bg = bg.crop((left, top, left + canvas_w, top + canvas_h))

    # Per-turn mouth states.
    turn_states = []
    for turn in turns:
        states, _duration = _read_wav_rms_states(turn["audio_path"])
        turn_states.append(states)

    # Pre-render each character's 3 mouth-state bubbles once.
    bubble_cache = {}
    for character in DUO_POSITIONS:
        imgs = CHARACTER_IMAGES[character]
        bubble_cache[character] = {
            state: _make_bubble(imgs[state], DUO_BUBBLE_RADIUS)
            for state in ("closed", "mid", "open")
        }

    frame_idx = 0
    for turn_i, turn in enumerate(turns):
        speaker = turn["character"]
        states = turn_states[turn_i]
        for state in states:
            frame = bg.copy()
            for character, (cx, cy) in DUO_POSITIONS.items():
                st = state if character == speaker else "closed"
                bubble = bubble_cache[character][st]
                px = cx - DUO_BUBBLE_RADIUS
                py = cy - DUO_BUBBLE_RADIUS
                frame.paste(bubble, (px, py), bubble)
                if character == speaker:
                    draw = ImageDraw.Draw(frame)
                    r = DUO_BUBBLE_RADIUS + DUO_RING_WIDTH // 2
                    draw.ellipse(
                        (cx - r, cy - r, cx + r, cy + r),
                        outline=DUO_RING_COLOR,
                        width=DUO_RING_WIDTH,
                    )
            frame.save(os.path.join(frames_dir, "frame_%05d.png" % frame_idx))
            frame_idx += 1

    # Concatenate all turn audios into one continuous track.
    concat_list_path = os.path.join(frames_dir, "concat_list.txt")
    with open(concat_list_path, "w") as f:
        for turn in turns:
            f.write("file '%s'\n" % turn["audio_path"])
    concat_audio_path = os.path.join(frames_dir, "concat_audio.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_list_path, "-c", "copy", concat_audio_path],
        check=True,
    )

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(FPS),
        "-i", os.path.join(frames_dir, "frame_%05d.png"),
        "-i", concat_audio_path,
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-shortest",
        out_video_path
    ]
    subprocess.run(cmd, check=True)
    return out_video_path
