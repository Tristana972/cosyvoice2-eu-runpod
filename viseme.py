"""
Procedural viseme (mouth-swap) animator for stylized avatars (Zuzu, Titu)
that neural lip-sync models (MuseTalk, etc.) can't handle because they lack
human facial topology.

Ported from the standalone viseme_animate.py script, adapted to run inside
the cosyvoice2-eu-runpod worker with the character mouth images bundled in
/content/assets/.

Also supports a "duo" mode: two characters visible together in the same
frame over an AI-generated background, taking turns speaking (side by side,
full body), for the same-frame Zuzu/Titu interaction feature.

Changes in this revision:
- Idle motion is now a mesh warp, not a rigid whole-image rotate/translate.
  A single flat image can only be rotated/translated as ONE stiff block --
  that's why the first version of this file "floated" instead of feeling
  alive (arms/tail can't move independently from the torso if there's
  nothing separating them). A hand-rigged, per-limb-layer puppet would fix
  that but only for characters someone manually re-cuts into layers -- it
  can't generalize to a photo a user uploads in the app. A mesh warp
  instead displaces a grid of points across the *same* flat cutout, with
  displacement amplitude growing away from the feet (like a reed rooted at
  the ground, swaying more at the head/antenna/arm tips than at the base).
  That gives visually independent-looking limb/extremity motion without
  any manual layer separation, so it works automatically on ANY full-body
  cutout -- Zuzu, Titu, or a future avatar generated from a user's photo.
- Duo mode no longer crops characters into a circular "video call" bubble
  (which only showed the bust). It now composites full-body cutouts
  (transparent PNGs, background removed from the original art) standing
  side by side on the generated background, with a soft contact shadow and
  a non-circular "speaking" highlight (soft colour glow + slight scale-up)
  instead of a ring.
"""
import wave
import os
import math
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

FPS = 25
CLOSED_THRESH = 0.12
OPEN_THRESH = 0.45

ASSETS_DIR = "/content/assets"

# Original full-frame mouth-swap images (own baked-in background), used as-is
# for solo (single character) videos.
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

# Background-removed (transparent) full-body cutouts of the same 6 images,
# used for duo mode so characters can be composited onto a different
# generated background without a circular mask.
CUTOUT_IMAGES = {
    "zuzu": {
        "closed": os.path.join(ASSETS_DIR, "zuzu_mouth_closed_cutout.png"),
        "mid": os.path.join(ASSETS_DIR, "zuzu_mouth_mid_cutout.png"),
        "open": os.path.join(ASSETS_DIR, "zuzu_mouth_open_cutout.png"),
    },
    "titu": {
        "closed": os.path.join(ASSETS_DIR, "titu_mouth_closed_cutout.png"),
        "mid": os.path.join(ASSETS_DIR, "titu_mouth_mid_cutout.png"),
        "open": os.path.join(ASSETS_DIR, "titu_mouth_open_cutout.png"),
    },
}

# --- Idle body motion (applied every frame, solo + duo) ---------------------
# Mesh-warp "reed" motion: displacement grows away from the feet (anchor),
# so the head/antenna/arm tips sway noticeably more than the torso base --
# reads as limbs moving independently even though it's a single flat cutout.
IDLE_ZOOM = 1.28          # solo mode: render margin so the warp never samples outside the image
IDLE_PAD_FRAC = 0.25      # duo mode: transparent margin fraction, same purpose
MESH_COLS = 6
MESH_ROWS = 10
IDLE_WOBBLE_AMP_X = 0.05   # horizontal sway at the tip, fraction of width
IDLE_WOBBLE_AMP_Y = 0.032  # vertical sway at the tip, fraction of height
IDLE_WOBBLE_T1 = 1.7
IDLE_WOBBLE_T2 = 2.3
IDLE_WOBBLE_T3 = 2.9

# --- Duo layout ---------------------------------------------------------
DUO_CANVAS = (720, 1280)
DUO_CHAR_HEIGHT_FRAC = 0.55   # each character's rendered height, fraction of canvas height
DUO_GROUND_Y_FRAC = 0.84      # baseline ("feet") y position, fraction of canvas height
DUO_SLOT_X_FRAC = {
    "zuzu": 0.27,
    "titu": 0.73,
}
DUO_SEED_OFFSET = {
    "zuzu": 0.0,
    "titu": 3.1,
}
DUO_GLOW_COLOR = (34, 211, 238)  # cyan, matches the app's accent color
DUO_GLOW_BLUR = 16
DUO_SPEAKER_SCALE = 1.08
DUO_NONSPEAKER_BRIGHTNESS = 0.82
DUO_SHADOW_OPACITY = 90


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


def _mesh_wobble(img, frame_idx, fps, seed=0.0, cols=MESH_COLS, rows=MESH_ROWS,
                  amp_x_frac=IDLE_WOBBLE_AMP_X, amp_y_frac=IDLE_WOBBLE_AMP_Y):
    """Warp `img` (any PIL mode, incl. RGBA) with a smooth per-vertex
    displacement field over a cols x rows grid. Amplitude grows away from
    the bottom edge (feet), like a reed rooted at the ground -- so the
    head/antennae/raised arm sway visibly more than the torso base. This is
    what makes the body read as alive instead of one rigid floating block,
    and unlike a hand-cut limb rig it works on ANY full-body cutout with no
    per-character prep, so it will apply just as well to a future avatar
    generated from a user's own photo as it does to Zuzu/Titu.

    The image must already have enough margin (via zoom or transparent
    padding -- see callers) that displaced sample points never fall outside
    its bounds, or the warp will pull in blank/edge pixels.
    """
    w, h = img.size
    t = frame_idx / float(fps)

    xs = [round(i * w / cols) for i in range(cols + 1)]
    ys = [round(j * h / rows) for j in range(rows + 1)]

    def vertex_disp(u, v):
        vw = (1 - v) ** 1.5  # 0 at the feet (v=1), max at the top (v=0)
        phase = seed + u * 2.4 + v * 1.1
        dx = amp_x_frac * w * vw * (
            0.65 * math.sin(2 * math.pi * t / IDLE_WOBBLE_T1 + phase) +
            0.35 * math.sin(2 * math.pi * t / IDLE_WOBBLE_T2 + phase * 1.7 + 0.6)
        )
        dy = amp_y_frac * h * vw * 0.6 * math.sin(2 * math.pi * t / IDLE_WOBBLE_T3 + phase + 1.0)
        return dx, dy

    verts = {}
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            dx, dy = vertex_disp(x / w, y / h)
            verts[(i, j)] = (x + dx, y + dy)

    mesh = []
    for j in range(rows):
        for i in range(cols):
            box = (xs[i], ys[j], xs[i + 1], ys[j + 1])
            p00 = verts[(i, j)]
            p01 = verts[(i, j + 1)]
            p11 = verts[(i + 1, j + 1)]
            p10 = verts[(i + 1, j)]
            quad = (*p00, *p01, *p11, *p10)
            mesh.append((box, quad))

    return img.transform((w, h), Image.MESH, mesh, resample=Image.BICUBIC)


def _apply_idle_solo(zoomed_img, orig_w, orig_h, frame_idx, fps, seed=0.0):
    """zoomed_img is the state image pre-scaled by IDLE_ZOOM, giving margin
    for the mesh warp to sample from. Warps it, then center-crops back down
    to the original frame size."""
    zw, zh = zoomed_img.size
    warped = _mesh_wobble(zoomed_img, frame_idx, fps, seed=seed)
    left = (zw - orig_w) // 2
    top = (zh - orig_h) // 2
    return warped.crop((left, top, left + orig_w, top + orig_h))


def _apply_idle_rgba(img, frame_idx, fps, seed=0.0):
    """Same idea as _apply_idle_solo but for a transparent RGBA cutout: pad
    with transparent margin instead of zooming (no baked-in background to
    stretch), so the warp never reveals a hard edge."""
    w, h = img.size
    pad = int(max(w, h) * IDLE_PAD_FRAC) + 4
    canvas = Image.new("RGBA", (w + pad * 2, h + pad * 2), (0, 0, 0, 0))
    canvas.paste(img, (pad, pad), img)
    warped = _mesh_wobble(canvas, frame_idx, fps, seed=seed)
    return warped.crop((pad, pad, pad + w, pad + h))


def animate(audio_path, character, out_video_path, frames_dir):
    if character not in CHARACTER_IMAGES:
        raise ValueError(
            "Unknown character '%s', expected one of %s" % (character, list(CHARACTER_IMAGES))
        )

    imgs = CHARACTER_IMAGES[character]
    os.makedirs(frames_dir, exist_ok=True)

    smoothed, _duration = _read_wav_rms_states(audio_path)

    orig_size = None
    zoomed_by_state = {}
    for state, path in imgs.items():
        im = Image.open(path).convert("RGB")
        if orig_size is None:
            orig_size = im.size
        zw, zh = int(im.size[0] * IDLE_ZOOM), int(im.size[1] * IDLE_ZOOM)
        zoomed_by_state[state] = im.resize((zw, zh), Image.LANCZOS)

    orig_w, orig_h = orig_size
    seed = 0.0

    for i, st in enumerate(smoothed):
        frame = _apply_idle_solo(zoomed_by_state[st], orig_w, orig_h, i, FPS, seed)
        frame.save(os.path.join(frames_dir, "frame_%05d.png" % i))

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


def _prepare_full_body(cutout_path, canvas_h):
    """Load a transparent cutout and scale it to a fixed on-screen height."""
    img = Image.open(cutout_path).convert("RGBA")
    target_h = max(1, int(canvas_h * DUO_CHAR_HEIGHT_FRAC))
    w, h = img.size
    scale = target_h / float(h)
    return img.resize((max(1, int(w * scale)), target_h), Image.LANCZOS)


def _tinted_glow(rgba_img, color, blur):
    """Blurred, colour-tinted silhouette used as a soft 'speaking' highlight
    behind a character, instead of the old circular ring."""
    alpha = rgba_img.split()[-1]
    solid = Image.new("RGBA", rgba_img.size, color + (255,))
    solid.putalpha(alpha)
    pad = blur * 2
    canvas = Image.new("RGBA", (rgba_img.size[0] + pad * 2, rgba_img.size[1] + pad * 2), (0, 0, 0, 0))
    canvas.paste(solid, (pad, pad), solid)
    return canvas.filter(ImageFilter.GaussianBlur(blur)), pad


def _dim(rgba_img, factor):
    r, g, b, a = rgba_img.split()
    rgb = ImageEnhance.Brightness(Image.merge("RGB", (r, g, b))).enhance(factor)
    r2, g2, b2 = rgb.split()
    return Image.merge("RGBA", (r2, g2, b2, a))


def _paste_character(frame_rgba, char_img, anchor_x, ground_y, speaking):
    """Composite one full-body character onto the frame: optional speaking
    glow + slight scale-up, contact shadow, then the character itself."""
    img = char_img
    if speaking:
        w, h = img.size
        img = img.resize((int(w * DUO_SPEAKER_SCALE), int(h * DUO_SPEAKER_SCALE)), Image.LANCZOS)
        glow, gpad = _tinted_glow(img, DUO_GLOW_COLOR, DUO_GLOW_BLUR)
        gx = anchor_x - glow.size[0] // 2
        gy = ground_y - img.size[1] - gpad + int(img.size[1] * 0.06)
        frame_rgba.alpha_composite(glow, (gx, gy))
    else:
        img = _dim(img, DUO_NONSPEAKER_BRIGHTNESS)

    w, h = img.size

    shadow_h = max(4, int(h * 0.10))
    shadow_w = int(w * 0.9)
    sblur = max(2, int(shadow_h * 0.3))
    spad = sblur * 2
    shadow = Image.new("RGBA", (shadow_w + spad * 2, shadow_h + spad * 2), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).ellipse(
        (spad, spad, spad + shadow_w, spad + shadow_h), fill=(0, 0, 0, DUO_SHADOW_OPACITY)
    )
    shadow = shadow.filter(ImageFilter.GaussianBlur(sblur))
    sx = anchor_x - shadow.size[0] // 2
    sy = ground_y - shadow_h // 2 - spad
    frame_rgba.alpha_composite(shadow, (sx, sy))

    px = anchor_x - w // 2
    py = ground_y - h
    frame_rgba.alpha_composite(img, (px, py))


def animate_duo(turns, background_path, out_video_path, frames_dir):
    """
    turns: list of dicts, each { "character": "zuzu"|"titu", "audio_path": "..." },
        in the order they speak.
    background_path: path to a generated background image (any size, will be
        cover-cropped to fill the canvas).

    Produces one video where both characters are visible for the whole
    duration, full body, standing side by side over the background, with a
    soft colour glow + slight scale-up (no circular crop/ring) around
    whichever one is currently speaking, and only that character's mouth
    animating (the other stays closed / dimmed). Both characters have a
    light idle body motion (bob/sway/micro-rotation) at all times.
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
    bg = bg.crop((left, top, left + canvas_w, top + canvas_h)).convert("RGBA")

    # Per-turn mouth states.
    turn_states = []
    for turn in turns:
        states, _duration = _read_wav_rms_states(turn["audio_path"])
        turn_states.append(states)

    # Pre-render each character's 3 mouth-state full-body cutouts once.
    char_state_imgs = {}
    for character in DUO_SLOT_X_FRAC:
        imgs = CUTOUT_IMAGES[character]
        char_state_imgs[character] = {
            state: _prepare_full_body(imgs[state], canvas_h)
            for state in ("closed", "mid", "open")
        }

    ground_y = int(canvas_h * DUO_GROUND_Y_FRAC)
    anchors_x = {c: int(canvas_w * frac) for c, frac in DUO_SLOT_X_FRAC.items()}

    frame_idx = 0
    for turn_i, turn in enumerate(turns):
        speaker = turn["character"]
        states = turn_states[turn_i]
        for state in states:
            frame = bg.copy()
            for character in DUO_SLOT_X_FRAC:
                is_speaker = character == speaker
                st = state if is_speaker else "closed"
                base_img = char_state_imgs[character][st]
                moved = _apply_idle_rgba(base_img, frame_idx, FPS, DUO_SEED_OFFSET[character])
                _paste_character(frame, moved, anchors_x[character], ground_y, speaking=is_speaker)
            frame.convert("RGB").save(os.path.join(frames_dir, "frame_%05d.png" % frame_idx))
            frame_idx += 1

    # Concatenate all turn audios into one continuous track.
    concat_list_path = os.path.join(frames_dir, "concat_list.txt")
    with open(concat_list_path, "w") as f:
        for turn in turns:
            f.write("file '%s'\n" % os.path.abspath(turn["audio_path"]))
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
