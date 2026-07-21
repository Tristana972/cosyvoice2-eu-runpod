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

Round 1:
- Duo mode composites full-body cutouts (transparent PNGs, background
  removed from the original art) standing side by side on the generated
  background, with a soft contact shadow and a non-circular "speaking"
  highlight (soft colour glow + slight scale-up) instead of the old
  circular "video call" bubble crop.

Round 2 (per feedback "ca flotte, les bras ne bougent pas, la queue ne
frétille pas" on a rigid whole-image rotate/translate):
- Tried a mesh warp (grid of points displaced with amplitude growing away
  from the feet). Amplified further on a second pass. Still read as the
  whole image undulating/melting rather than a character actually moving
  -- because it fundamentally is a single flat image being deformed, not
  distinct parts moving.

Round 3 (per feedback "ca module toujours, c'est pas realiste, c'est pas
anime, le but c'est de l'animer comme s'il etait vivant" -- the mesh warp
approach was abandoned):
- Replaced the whole-image warp with an actual 2-part rig: the torso stays
  essentially still (only a small idle weight-shift bob, a plain
  translate, no deformation), and the one limb each character visibly has
  (Zuzu's raised arm, Titu's tail) is cut out as its own layer and
  *rotated* around its real attachment point (shoulder / tail base) with
  a slow wave plus a faster small "flick" on top -- so the arm genuinely
  waves and the tail genuinely wags, instead of the whole silhouette
  rippling. The cut is a plain rectangular crop around the limb (not a
  hand-traced silhouette): the same rectangle is erased (to transparent)
  from the torso and re-drawn each frame rotated about a pivot that sits
  near one corner of that rectangle, so at rest (angle 0) it's pixel
  -identical to the original artwork, and at speed the near-pivot pixels
  barely move while the far tip sweeps a visible arc. This works on any
  full-body cutout given a manually-picked box+pivot per character --
  it's not fully automatic like the old mesh warp was, but it's what
  actually reads as "alive" instead of "floating/melting", which is what
  matters here.
- Solo mode used to reuse the flat baked-background art (CHARACTER_IMAGES)
  and warp the whole frame including that background. Since the rig now
  needs to erase-and-redraw just the limb, solo mode switched to the same
  transparent-cutout-over-a-background-plate compositing duo mode already
  used, with a simple synthetic background plate (a 2-stop vertical
  gradient + a flat grass band, colour-sampled from the original baked
  art) generated once per character so there's something clean to reveal
  behind the erased limb.

Round 4 (per feedback, see build_duo_composites docstring): whole-image
rotate/warp rig abandoned for duo in favour of a single static composite
per possible speaker, fed to WaveSpeedAI (Wan2.2-S2V) for the actual
animation.

Round 5 (20 juillet, retour Tristana "les 2 perso sont toujours au premier
plan" -- observé APRES que le fond IA ait été enrichi côté worker.js pour
avoir une vraie composition en plans, premier plan/plan intermédiaire/
arrière-plan, voir buildSceneBackgroundPrompt) :
- Un fond mieux composé ne suffit pas a lui seul : les personnages sont
  collés en DERNIER, par-dessus l'image de fond entière, donc rien ne les
  recouvre jamais, même quand le fond contient un élément de premier plan
  dessiné "near the bottom edges" comme demandé -- il reste visuellement
  DERRIERE eux dans notre empilement de calques, donc illisible comme
  premier plan. Corrigé sans appel IA supplémentaire (le fond en contient
  déjà un, gratuit) : on redécoupe une bande du bas de CE MEME fond généré
  et on la replaque par-dessus les personnages une fois posés -- ce qui
  masque leurs pieds/bas de jambes derrière ce qui a été dessiné là
  (plante, rambarde, bord de table...), donnant une vraie occlusion de
  premier plan au lieu d'un aplat.
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

# Original full-frame mouth-swap images (own baked-in background). Only
# used now as the colour source for the synthetic solo background plate
# (see _solo_background) -- actual solo rendering composites the
# transparent CUTOUT_IMAGES over that plate, same as duo mode.
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

# Background-removed (transparent) full-body cutouts of the same 6 images.
# This is the actual source used to render both solo and duo now, since
# the limb rig needs real alpha to erase/redraw the arm or tail.
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
# Whole-body idle motion: a small, plain translate (weight-shift bob), NOT a
# deformation -- the torso should read as solid/still, so the limb rig below
# is what carries the "alive" feeling instead of the body wobbling too.
BOB_AMP_Y_FRAC = 0.010   # vertical bob amplitude, fraction of character height
BOB_AMP_X_FRAC = 0.005   # horizontal sway amplitude, fraction of character width
BOB_PERIOD = 2.2         # seconds per bob cycle

# Limb rig: per character, a list of {box, pivot, amp_deg, period, ...}.
# `box` = (left, top, right, bottom) in the ORIGINAL cutout's own pixel
# coordinates (240x425 for zuzu, 240x356 for titu), picked by hand around
# the one limb each character has, with generous padding so the tip never
# gets clipped when it swings. `pivot` = the rotation point (shoulder / tail
# base), also in original pixel coordinates, and must fall inside `box`.
# The box is erased from the torso and redrawn each frame rotated about the
# pivot -- at rest (angle 0) that reproduces the original artwork exactly.
LIMB_DEFS = {
    "zuzu": [
        {  # the raised/waving arm (hand + fingers), bottom-left of the
            # character. `mask_polygon` traces the actual paw shape so only
            # the arm erases/rotates, not the chunk of shoulder/body that
            # also sits inside `box` (same fix as Titu's tail -- see the
            # docstring on `mask_polygon` in LIMB_DEFS["titu"]).
            "box": (-15, 205, 108, 300),
            "pivot": (63, 233),
            "mask_polygon": [
                (63, 233), (58, 227), (48, 224), (40, 226),
                (36, 232), (33, 238),
                (28, 230), (20, 225), (14, 229),
                (8, 236), (5, 246), (5, 255),
                (7, 265), (12, 275), (20, 282),
                (30, 287), (40, 286), (50, 280),
                (58, 270), (62, 255), (63, 240),
            ],
            "rest_deg": 4,
            "amp_deg": 16,
            "period": 1.25,
            "flick_deg": 5,
            "flick_period": 0.42,
        },
    ],
    "titu": [
        {  # the tail, bottom-right of the character. `box` just bounds the
            # crop/rotation; `mask_polygon` (hand-traced around the tail's
            # actual silhouette, in the SAME original-image coordinates as
            # `box`/`pivot`) is what actually separates the tail from the
            # chunk of rear/body that also sits inside this box -- without
            # it the whole box's opaque area (tail + torso) got erased and
            # rotated together, which read as a rigid rectangle swinging
            # instead of just the tail.
            "box": (145, 225, 240, 330),
            "pivot": (178, 248),
            "mask_polygon": [
                (180, 245), (190, 246), (200, 248), (210, 252), (218, 258),
                (225, 265), (227, 275), (223, 285), (215, 293),
                (207, 297), (200, 292), (196, 285),
                (190, 288), (183, 295), (178, 305), (173, 313),
                (168, 318), (162, 315), (160, 305),
                (158, 295), (158, 285), (160, 275), (163, 265),
                (168, 255), (175, 248),
            ],
            "rest_deg": 0,
            "amp_deg": 18,
            "period": 0.85,
            "flick_deg": 7,
            "flick_period": 0.35,
        },
    ],
}

# --- Duo layout ---------------------------------------------------------
# DUO_CHAR_HEIGHT_FRAC was 0.55 with slots at 0.27/0.73 (332px apart on a
# 720px-wide canvas): at that height each full-body cutout renders wide
# enough that the two characters' bounding boxes actually overlap in the
# middle (confirmed visually -- Titu drawn over Zuzu's arm/body instead of
# standing beside it, "superposés" per feedback). Shrunk + spread out so
# there's real clearance between them.
DUO_CANVAS = (720, 1280)
DUO_CHAR_HEIGHT_FRAC = 0.40   # each character's rendered height, fraction of canvas height
DUO_GROUND_Y_FRAC = 0.84      # baseline ("feet") y position, fraction of canvas height
DUO_SLOT_X_FRAC = {
    "zuzu": 0.27,
    "titu": 0.73,
}
# Titu is a dog (on all fours, low to the ground) standing next to Zuzu, not
# the same height -- per feedback, Titu should only come up to about Zuzu's
# neck. Both characters used to be scaled to the exact same target_h, which
# erased that size difference (and made them read as equal-height, floating
# at the wrong relative scale). This is a per-character multiplier applied
# on top of DUO_CHAR_HEIGHT_FRAC's base height.
DUO_CHAR_HEIGHT_SCALE = {
    "zuzu": 1.0,
    "titu": 0.72,
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

# Round 5 (see module docstring): fraction of canvas height where the
# foreground-occlusion strip starts. Originally 0.80 (just above
# DUO_GROUND_Y_FRAC, 0.84) so the strip only overlapped the characters'
# ankles -- worked for a standing pose, but barely touches a SEATED pose
# (bench, chair...): the character's body occupies most of the canvas
# ABOVE the ground line (DUO_CHAR_HEIGHT_FRAC=0.40 means the top of a
# character is around y=0.44), so a strip starting at 0.80 only ever
# touched the very bottom ~10% of their height, hence Tristana's continued
# "les 2 perso sont toujours au premier plan" after testing a seated scene.
# Raising this a lot further (e.g. to cover half the body) was considered
# and rejected: that same lower-body band is exactly where the earlier
# "jambes coupées" regression came from (a background object overlapping
# the characters' legs in the source composite confused WaveSpeedAI's
# animation into rendering them as cut off, see worker.js
# buildSceneBackgroundPrompt/DUO comments) -- reintroducing a hard object
# edge directly across their legs here risks the same bug. Nudged up more
# modestly instead (0.80 -> 0.75) for a bit more coverage without eating
# deep into the leg silhouette; buildSceneBackgroundPrompt in worker.js now
# also asks for that foreground detail to sit toward the LEFT/RIGHT edges
# of the frame rather than directly in front of where the characters
# stand, so the strip mostly frames them from the sides instead of
# potentially cutting across their body with a hard edge.
DUO_FOREGROUND_Y_FRAC = 0.75


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


def _solo_background(character):
    """Build a simple synthetic background plate for solo mode: a 2-stop
    vertical gradient (sky) + a flat grass band, colour-sampled from the
    original baked-background art so it stays visually consistent, at the
    exact same canvas size as the cutout (so no rescaling is needed when
    compositing). This exists purely so there's something clean behind the
    erased limb -- solo mode used to warp the baked art directly, but the
    rig needs to erase/redraw a rectangle, which would tear a visible hole
    in a baked-in background."""
    baked = Image.open(CHARACTER_IMAGES[character]["closed"]).convert("RGB")
    w, h = baked.size
    top_color = baked.getpixel((w - 3, 2))
    grass_top_y = int(h * 0.90)
    mid_color = baked.getpixel((w - 3, max(0, grass_top_y - 2)))
    grass_color = baked.getpixel((2, h - 3))

    bg = Image.new("RGB", (w, h))
    px = bg.load()
    for y in range(grass_top_y):
        frac = y / float(max(1, grass_top_y - 1))
        r = int(top_color[0] + (mid_color[0] - top_color[0]) * frac)
        g = int(top_color[1] + (mid_color[1] - top_color[1]) * frac)
        b = int(top_color[2] + (mid_color[2] - top_color[2]) * frac)
        for x in range(w):
            px[x, y] = (r, g, b)
    for y in range(grass_top_y, h):
        for x in range(w):
            px[x, y] = grass_color
    return bg.convert("RGBA")


LIMB_ALPHA_THRESH = 10   # alpha value above which a pixel counts as "limb", not background
LIMB_HOLE_GROW = 1       # px the hole is grown beyond the traced silhouette (covers AA rim)
LIMB_HOLE_FEATHER = 4    # px soft fade at the hole's edge (blends into surrounding torso)


def _erase_and_extract_limb(cutout_img, limb_def):
    """Crop `limb_def['box']` out of cutout_img as a standalone RGBA patch.

    The box's own alpha (opaque = character) is NOT enough on its own to
    know which pixels are "the limb": the box necessarily also covers a
    chunk of the TORSO right next to the limb (e.g. Titu's tail-erase box
    also overlaps the round rear of the body), and that torso area is just
    as opaque as the tail. Using "opaque inside the box" as the erase mask
    therefore erased a whole rectangular slab of real body, not just the
    tail -- which read as a moving cut-out frame rather than a wagging
    tail. `mask_polygon` (hand-traced, in the box's own local coordinates)
    is what actually separates "this pixel is the limb" from "this pixel
    is the torso that happens to sit inside the box": only pixels that are
    BOTH inside that polygon AND opaque are erased/extracted, so a slightly
    generous polygon still can't eat into the body outside it, and the box
    itself only exists to bound the crop/rotation, not to define the mask."""
    box = limb_def["box"]
    patch = cutout_img.crop(box)
    a_arr = np.asarray(patch.split()[-1], dtype=np.float32)
    opaque = a_arr > LIMB_ALPHA_THRESH

    polygon = limb_def.get("mask_polygon")
    if polygon:
        local_poly = [(px - box[0], py - box[1]) for px, py in polygon]
        poly_mask = Image.new("L", patch.size, 0)
        ImageDraw.Draw(poly_mask).polygon(local_poly, fill=255)
        poly_arr = np.asarray(poly_mask, dtype=np.float32) > 10
        solid_bool = opaque & poly_arr
    else:
        solid_bool = opaque

    # The background-removal pass that produced the transparent cutout PNGs
    # left small erroneous transparent gaps INSIDE the character's true
    # silhouette in a couple of spots (e.g. between Titu's tail and its
    # rump) -- confirmed by comparing against the original baked-background
    # art, which shows solid body there with no gap at all. Composited over
    # any background, that stray hole reveals a sliver of it right next to
    # the limb, reading as a coloured line that doesn't belong. These holes
    # are "enclosed" (fully surrounded by opaque pixels, unlike the real
    # background which touches the crop's own border), so they can be
    # found and treated as "also erase/fill" for the torso -- not as part
    # of the limb itself, just healing a bad matte -- without needing to
    # touch the true outer edge of the silhouette at all.
    import cv2 as _cv2
    transparent = (~opaque).astype(np.uint8)
    num_t, labels_t = _cv2.connectedComponents(transparent, connectivity=4)
    border_labels = set(labels_t[0, :]) | set(labels_t[-1, :]) | set(labels_t[:, 0]) | set(labels_t[:, -1])
    enclosed = np.isin(labels_t, list(border_labels), invert=True) & (labels_t > 0)
    solid_bool_torso = solid_bool | enclosed

    # Two different masks are derived from the same traced polygon, grown by
    # two different amounts, because they serve two different jobs that
    # were previously (wrongly) sharing one mask:
    #  - the PATCH's own alpha should hug the true limb silhouette tightly
    #    (only a couple px of softening for anti-aliasing) so that at rest
    #    (angle 0) it reproduces the original limb edge pixel-for-pixel.
    #  - the TORSO's erase area needs to be grown more generously, since the
    #    hand-traced polygon is never pixel-perfect and any true limb pixel
    #    left un-erased shows up as a frozen "ghost" fragment.
    # Using the SAME (generously grown) mask for both, as before, meant that
    # at rest, the ring between the tight true limb edge and the generous
    # erase edge got blended TWICE -- once when the torso's own colour was
    # mixed toward the fill colour there, and again when the patch (using
    # that same partial alpha) was composited back on top -- which does not
    # mathematically cancel out to the original colour for any partial
    # (feathered) alpha, only at fully 0 or fully 1. That mismatch is what
    # showed up as a persistent tan/gold arc at the limb's edge even at
    # rest. Decoupling the two masks removes the double-counted blend.
    def _mask_from(bool_arr, grow, feather):
        m = Image.fromarray(bool_arr.astype(np.uint8) * 255, mode="L")
        if grow > 0:
            m = m.filter(ImageFilter.MaxFilter(2 * grow + 1))
        if feather > 0:
            m = m.filter(ImageFilter.GaussianBlur(feather))
        return np.asarray(m, dtype=np.float32) / 255.0

    patch_alpha = _mask_from(solid_bool, grow=1, feather=1)
    hole_alpha = _mask_from(solid_bool_torso, grow=LIMB_HOLE_GROW, feather=LIMB_HOLE_FEATHER)

    # The patch itself is masked down to its tight silhouette so it only
    # ever draws the limb, never the slice of torso that shares its
    # bounding box -- otherwise that torso slice would still rotate along
    # with the limb every frame.
    r, g, b, pa = patch.split()
    pa_arr = np.asarray(pa, dtype=np.float32) * patch_alpha[: patch.size[1], : patch.size[0]]
    patch = Image.merge("RGBA", (r, g, b, Image.fromarray(pa_arr.astype(np.uint8), mode="L")))

    # Instead of erasing straight to transparent (which reveals whatever
    # sits behind the character -- the synthetic sky/grass background plate
    # -- through any gap between the hole and the redrawn patch), the hole
    # is filled with texture cloned from the surrounding body via OpenCV's
    # inpainting (Telea algorithm). A single flat sampled colour was tried
    # first, but the body art has gradient shading (highlights/shadow), so a
    # flat patch either mismatched the local gradient or -- when the sample
    # point landed on an anti-aliased edge pixel -- picked up a stray tinted
    # colour, both very visible. Inpainting extends the real neighbouring
    # gradient into the hole instead, which blends correctly regardless of
    # exactly where on the gradient the hole happens to sit.
    torso = cutout_img.copy()
    r, g, b, ta = torso.split()
    rgb_arr = np.array(torso.convert("RGB"), dtype=np.uint8)
    ta_arr = np.asarray(ta, dtype=np.float32)

    # box may extend past the image edges (e.g. a negative left, when the
    # limb is near the canvas border) -- clip to the actual image bounds
    # before indexing, and take the matching sub-slice of hole_alpha (whose
    # own local origin is box's top-left, padding included) so the two
    # stay aligned.
    ix0, iy0 = max(box[0], 0), max(box[1], 0)
    ix1, iy1 = min(box[2], torso.width), min(box[3], torso.height)
    hx0, hy0 = ix0 - box[0], iy0 - box[1]
    hx1, hy1 = hx0 + (ix1 - ix0), hy0 + (iy1 - iy0)
    h_full = hole_alpha[hy0:hy1, hx0:hx1]

    # Transparent AND semi-transparent (anti-aliased edge) pixels still
    # carry leftover matte/background colour in their RGB channels even
    # where alpha is low or partial -- cv2.inpaint only looks at RGB, not
    # alpha, so if those edge pixels are left unmasked it happily uses that
    # stray colour as "known" texture right next to the hole and bleeds it
    # in (this is what produced a pink/lavender blob for Zuzu: the cutout's
    # original background was a purple sky gradient, and the AA fringe
    # around the arm blends arm-green with sky-purple at partial alpha).
    # Only pixels that are SOLIDLY opaque are trustworthy body colour, so
    # anything below a high alpha bar is also marked "to fill" (excluded as
    # a source) -- the algorithm then only ever pulls from real solid body
    # pixels, however far it has to reach.
    orig_alpha_arr = np.asarray(cutout_img.split()[-1], dtype=np.float32)
    valid_source = orig_alpha_arr > 220

    # Thin dark ink outlines around the character's silhouette are just as
    # solidly opaque as the body fill, so without this they'd also count as
    # "known good" source colour -- and being right at the edge of almost
    # every opaque region, they get pulled in disproportionately, staining
    # the fill dark. Eroding the valid-source mask by a couple px removes
    # any region only 2-3px wide (the outline stroke) while leaving large
    # body-colour fields intact.
    valid_img = Image.fromarray((valid_source.astype(np.uint8)) * 255, mode="L")
    valid_img = valid_img.filter(ImageFilter.MinFilter(5))
    valid_source = np.asarray(valid_img, dtype=np.uint8) > 0

    # Both characters stand on a small opaque grass tuft baked into the
    # cutout art near their feet, right below/inside Titu's tail box -- it's
    # real, solid, opaque art, so the alpha/outline filters above don't
    # exclude it, but its green is unrelated to the body it would bleed
    # into. Excluding the bottom strip of the canvas as a source keeps the
    # fill pulling only from the body above (a plain gradient), never the
    # feet decoration. Both limb boxes sit well above this band, so it never
    # excludes the actual limb art itself.
    grass_band_y = int(torso.height * 0.88)
    valid_source[grass_band_y:, :] = False

    full_mask = np.zeros(ta_arr.shape, dtype=np.uint8)
    full_mask[iy0:iy1, ix0:ix1] = (h_full > 0.15).astype(np.uint8) * 255
    full_mask[~valid_source] = 255
    if full_mask.any():
        import cv2
        inpainted = cv2.inpaint(rgb_arr, full_mask, 5, cv2.INPAINT_TELEA)
    else:
        inpainted = rgb_arr

    # Replace with a binary (thresholded) mask rather than blending RGB by
    # h_full's continuous value: h_full is also what patch_alpha derives
    # from (same polygon, different grow/feather), and blending torso's own
    # colour by that same continuous value here, then AGAIN compositing
    # patch on top by its own continuous alpha, double-applies the fade and
    # does not cancel back out to the original colour at rest for any
    # partial alpha -- only at exactly 0 or 1. Swapping to a hard cutover
    # (still positioned using the smooth h_full contour, so its boundary
    # isn't jagged) removes that double-blend: outside the patch's own tight
    # coverage this area is deep inside the generously-grown erase region
    # anyway, where inpainted texture should already closely match the
    # surrounding body, so a hard edge there reads as a colour continuity,
    # not a seam.
    region_fill = inpainted[iy0:iy1, ix0:ix1].astype(np.float32)
    replace = h_full > 0.5
    region_orig = rgb_arr[iy0:iy1, ix0:ix1]
    region_orig[replace] = np.round(region_fill[replace]).astype(np.uint8)

    # Same binary cutover for alpha: a pixel below the threshold keeps its
    # original alpha (which, for a background gap pixel, may be low/zero --
    # correctly still invisible), instead of being nudged partially opaque
    # while still showing its old, possibly wrong-tinted colour (that
    # combination is exactly what let the original pink/lavender AA fringe
    # show through faintly even after the colour source fix above).
    box_ta = ta_arr[iy0:iy1, ix0:ix1]
    box_ta[replace] = 255.0

    torso = Image.merge("RGBA", (
        *Image.fromarray(rgb_arr, mode="RGB").split(),
        Image.fromarray(ta_arr.astype(np.uint8), mode="L"),
    ))
    return torso, patch


def _limb_angle(t, limb_def, seed):
    """Rotation angle (degrees) for one limb at time t: a slow sine wave
    (the main wave/wag) plus a smaller, faster sine ("flick") layered on
    top, so it reads as a wave/wag with a bit of extra wrist/tip snap
    rather than a single smooth metronome swing."""
    rest = limb_def.get("rest_deg", 0.0)
    amp = limb_def["amp_deg"]
    period = limb_def["period"]
    flick_deg = limb_def.get("flick_deg", 0.0)
    flick_period = limb_def.get("flick_period", 0.4)
    phase = seed
    wave = amp * math.sin(2 * math.pi * t / period + phase)
    flick = flick_deg * math.sin(2 * math.pi * t / flick_period + phase * 1.6)
    return rest + wave + flick


def _apply_limbs(torso, patches_and_defs, frame_idx, fps, seed=0.0):
    """Composite each limb patch back onto `torso`, rotated about its pivot
    for this frame. At angle 0 this exactly reproduces the original
    artwork; the pivot stays (almost) fixed while the rest of the patch
    sweeps an arc around it."""
    t = frame_idx / float(fps)
    frame = torso.copy()
    for patch, limb_def in patches_and_defs:
        angle = _limb_angle(t, limb_def, seed)
        box = limb_def["box"]
        pivot = limb_def["pivot"]
        local_pivot = (pivot[0] - box[0], pivot[1] - box[1])
        rotated = patch.rotate(angle, resample=Image.BICUBIC, center=local_pivot, fillcolor=(0, 0, 0, 0))
        frame.paste(rotated, (box[0], box[1]), rotated)
    return frame


def _apply_bob(img, frame_idx, fps, seed=0.0):
    """Small whole-character translate (idle weight-shift), not a warp --
    the character stays rigid/solid, just shifts position slightly."""
    t = frame_idx / float(fps)
    w, h = img.size
    dy = int(round(BOB_AMP_Y_FRAC * h * math.sin(2 * math.pi * t / BOB_PERIOD + seed)))
    dx = int(round(BOB_AMP_X_FRAC * w * math.sin(2 * math.pi * t / (BOB_PERIOD * 1.35) + seed + 0.7)))
    if dx == 0 and dy == 0:
        return img
    out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    out.paste(img, (dx, dy), img)
    return out


def _build_rig_states(character):
    """For each mouth state, load the cutout and pre-erase this character's
    limb box(es) once (cheap, done a handful of times, not per frame).
    Returns {state: (torso_with_holes, [(patch, limb_def), ...])}."""
    limb_defs = LIMB_DEFS.get(character, [])
    states = {}
    for state, path in CUTOUT_IMAGES[character].items():
        img = Image.open(path).convert("RGBA")
        torso = img
        patches = []
        for ld in limb_defs:
            torso, patch = _erase_and_extract_limb(torso, ld)
            patches.append(patch)
        states[state] = (torso, list(zip(patches, limb_defs)))
    return states


def _render_character_frame(rig_states, state, frame_idx, fps, seed=0.0):
    """One fully-composited character frame (limb rotated + idle bob
    applied), at the cutout's own original resolution."""
    torso, patches_and_defs = rig_states[state]
    frame = _apply_limbs(torso, patches_and_defs, frame_idx, fps, seed)
    frame = _apply_bob(frame, frame_idx, fps, seed)
    return frame


def _scale_to_height(img, target_h):
    w, h = img.size
    scale = target_h / float(h)
    return img.resize((max(1, int(w * scale)), target_h), Image.LANCZOS)


def animate(audio_path, character, out_video_path, frames_dir):
    if character not in CHARACTER_IMAGES:
        raise ValueError(
            "Unknown character '%s', expected one of %s" % (character, list(CHARACTER_IMAGES))
        )

    os.makedirs(frames_dir, exist_ok=True)
    smoothed, _duration = _read_wav_rms_states(audio_path)

    bg_plate = _solo_background(character)
    rig_states = _build_rig_states(character)

    for i, st in enumerate(smoothed):
        char_frame = _render_character_frame(rig_states, st, i, FPS, seed=0.0)
        frame = bg_plate.copy()
        frame.paste(char_frame, (0, 0), char_frame)
        frame.convert("RGB").save(os.path.join(frames_dir, "frame_%05d.png" % i))

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


def _paste_character(frame_rgba, char_img, anchor_x, ground_y, speaking, glow=True):
    """Composite one full-body character onto the frame: optional speaking
    glow + slight scale-up, contact shadow, then the character itself.

    `glow=False` (used by build_duo_composites, see its docstring) skips the
    cyan tinted-glow highlight entirely: that glow was designed as a
    "who's speaking" cue for the old per-frame rig-animation video, baked
    directly into a single still image and blurred well past the body's own
    edges (including down past the feet). Reused unchanged as the *source
    photo* fed to WaveSpeedAI, that same blur reads as a cyan halo/aura
    around the character and a cyan smudge at their feet -- exactly the
    "aura bleue" / "traces" artifacts reported after switching to the
    WaveSpeedAI pipeline. WaveSpeedAI's own animation of the speaking
    character already makes clear who's talking, so the glow cue is no
    longer needed; the scale-up and contact shadow are kept (not colour
    artifacts, don't cause this issue).

    The non-speaker brightness dimming (previously `_dim(img,
    DUO_NONSPEAKER_BRIGHTNESS)`) has been removed for the same reason:
    another "who's speaking" cue designed for the old still-frame rig
    video, made redundant -- and visually distracting ("il s'allume /
    l'autre devient terne") -- now that WaveSpeedAI's own body/mouth
    animation of the speaking character already makes that clear. Both
    characters now keep their natural brightness regardless of who's
    speaking."""
    img = char_img
    if speaking:
        w, h = img.size
        img = img.resize((int(w * DUO_SPEAKER_SCALE), int(h * DUO_SPEAKER_SCALE)), Image.LANCZOS)
        if glow:
            glow_img, gpad = _tinted_glow(img, DUO_GLOW_COLOR, DUO_GLOW_BLUR)
            gx = anchor_x - glow_img.size[0] // 2
            gy = ground_y - img.size[1] - gpad + int(img.size[1] * 0.06)
            frame_rgba.alpha_composite(glow_img, (gx, gy))

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


def build_duo_composites(background_path):
    """Build one static, fully-composited PNG per possible speaker for a duo
    scene: the AI-generated background with both characters standing at
    their positions (speaker glowing/scaled-up, other dimmed/static),
    using each character's raw cutout art at rest -- no limb rig, no idle
    bob. This is the "Round 4" pivot: instead of trying to fake body motion
    with a hand-traced cutout+rotate rig (Rounds 1-3, see docstring above --
    always left some visible seam or wrong-coloured patch no matter how
    much the mask/inpaint got tuned), this composite is only ever the
    *input* to a real video-generation model (WaveSpeedAI Wan2.2-S2V,
    already used for solo mode and confirmed natural), which animates the
    whole character itself. One composite per possible speaker (2 for a
    Zuzu/Titu duo) is enough, since the visual state only depends on WHO is
    currently speaking, not on which line -- the same composite is reused
    for every turn that speaker has.

    Round 5 (see module docstring): after pasting both characters, a strip
    cropped from the BOTTOM of this same generated background is redrawn
    on top of everything, overlapping the characters' ankles/lower legs.
    Whatever the background model drew there (it's explicitly asked for
    foreground detail near the bottom edges, see buildSceneBackgroundPrompt
    in worker.js) then reads as truly in FRONT of the characters instead of
    behind them, which is what actually makes them look embedded in the
    scene instead of pasted on top of it.

    Returns {character: PIL.Image (RGB)}.
    """
    canvas_w, canvas_h = DUO_CANVAS

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

    ground_y = int(canvas_h * DUO_GROUND_Y_FRAC)
    anchors_x = {c: int(canvas_w * frac) for c, frac in DUO_SLOT_X_FRAC.items()}
    target_h = max(1, int(canvas_h * DUO_CHAR_HEIGHT_FRAC))

    rest_cutouts = {
        character: Image.open(CUTOUT_IMAGES[character]["closed"]).convert("RGBA")
        for character in DUO_SLOT_X_FRAC
    }

    # Round 5 foreground occlusion strip -- see docstring above. Cropped
    # once from the (already canvas-sized) background, reused for every
    # speaker composite.
    foreground_y = int(canvas_h * DUO_FOREGROUND_Y_FRAC)
    foreground_strip = bg.crop((0, foreground_y, canvas_w, canvas_h))

    # Round 6 (20/21 juillet, chantier "plan continu") : en plus des 2 composites
    # "qui parle" (zuzu/titu, chacun avec le personnage concerné légèrement
    # agrandi), on construit aussi un composite "neutral" où AUCUN des deux
    # n'est agrandi. C'est cette version neutre qui sert désormais d'image
    # source UNIQUE envoyée à WaveSpeedAI pour toute la scène en un seul appel
    # (au lieu d'un appel par réplique, avec le composite du "speaker" à
    # chaque fois) -- voir buildDuoTimelinePrompt/wavespeed_submit dans
    # worker.js. Elle doit aussi passer par la retouche de pose (voir
    # pose_submit/pose_wait) comme les 2 autres, donc c'est bien une clé de
    # plus dans ce dict, pas un calcul à part.
    composites = {}
    for speaker in list(DUO_SLOT_X_FRAC) + [None]:
        frame = bg.copy()
        for character in DUO_SLOT_X_FRAC:
            is_speaker = character == speaker
            char_h = max(1, int(target_h * DUO_CHAR_HEIGHT_SCALE.get(character, 1.0)))
            scaled = _scale_to_height(rest_cutouts[character], char_h)
            _paste_character(frame, scaled, anchors_x[character], ground_y, speaking=is_speaker, glow=False)
        frame.alpha_composite(foreground_strip, (0, foreground_y))
        composites[speaker if speaker is not None else "neutral"] = frame.convert("RGB")
    return composites


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
    small idle weight-shift bob at all times, and whichever limb they have
    (Zuzu's arm, Titu's tail) rotates about its real attachment point.
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

    # Pre-erase each character's limb box(es) once (per mouth state).
    rig_states = {character: _build_rig_states(character) for character in DUO_SLOT_X_FRAC}

    ground_y = int(canvas_h * DUO_GROUND_Y_FRAC)
    anchors_x = {c: int(canvas_w * frac) for c, frac in DUO_SLOT_X_FRAC.items()}
    target_h = max(1, int(canvas_h * DUO_CHAR_HEIGHT_FRAC))

    frame_idx = 0
    for turn_i, turn in enumerate(turns):
        speaker = turn["character"]
        states = turn_states[turn_i]
        for state in states:
            frame = bg.copy()
            for character in DUO_SLOT_X_FRAC:
                is_speaker = character == speaker
                st = state if is_speaker else "closed"
                raw = _render_character_frame(
                    rig_states[character], st, frame_idx, FPS, seed=DUO_SEED_OFFSET[character]
                )
                char_h = max(1, int(target_h * DUO_CHAR_HEIGHT_SCALE.get(character, 1.0)))
                scaled = _scale_to_height(raw, char_h)
                _paste_character(frame, scaled, anchors_x[character], ground_y, speaking=is_speaker)
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
