"""
Procedural viseme (mouth-swap) animator for stylized avatars (Zuzu, Titu)
that neural lip-sync models (MuseTalk, etc.) can't handle because they lack
human facial topology.

Ported from the standalone viseme_animate.py script, adapted to run inside
the cosyvoice2-eu-runpod worker with the character mouth images bundled in
/content/assets/.
"""
import wave
import os
import subprocess
import numpy as np
from PIL import Image

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


def animate(audio_path, character, out_video_path, frames_dir):
    if character not in CHARACTER_IMAGES:
        raise ValueError(
            "Unknown character '%s', expected one of %s" % (character, list(CHARACTER_IMAGES))
        )

    imgs = CHARACTER_IMAGES[character]
    os.makedirs(frames_dir, exist_ok=True)

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
    total_video_frames = int(duration * FPS)
    samples_per_frame = int(framerate / FPS)

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
