import os
import base64
import tempfile
import shutil
import subprocess
import requests
import runpod
import torch
import torchaudio

import viseme

_cosy = None


def get_model():
    global _cosy
    if _cosy is None:
        from cosyvoice2_eu import load
        _cosy = load()
    return _cosy


def download_file(url, suffix):
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(r.content)
    return path


def _trim_hallucinated_tail(wav, sr, text):
    """CosyVoice2-EU (zero-shot cloning against a fixed reference voice clip)
    sometimes keeps generating audio well past the end of the real text --
    observed by Tristana as garbled/hallucinated speech (repeated numbers,
    unrelated words, or a short trailing "ho ho"-type interjection) tacked
    onto the end of an otherwise-correct sentence, especially when the
    reference clip is longer than the target text.

    v1 (length-only heuristic, kept below as the fallback) only hard-cut
    once the TOTAL duration overran the text-length estimate by 15% --
    good at catching long runaway hallucinations, but a short one-word
    aside like "Ho Ho" tacked on right at the end rarely pushes the total
    duration that far over budget, so it slipped through untouched
    (confirmed by Tristana: "apres 'avant qu'il fasse nuit' Zuzu dit Ho Ho
    alors qu'il y a un point a la fin de la phrase dans le script").

    v2 (this version): once we're already past a plausible normal-speech
    duration for the text, scan for the first sustained silence gap
    (~250ms) in the audio's RMS envelope and hard-cut there instead. A
    trailing hallucination is reliably preceded by a short pause once the
    real sentence has actually finished, so cutting at that pause removes
    it cleanly regardless of how short it is -- something a pure
    total-duration check can never catch. Only starts searching AFTER the
    estimated normal duration has elapsed, so a comma/breathing pause
    mid-sentence (which happens before that point) can't cause an early
    false cut."""
    n_chars = max(len((text or "").strip()), 1)
    est_seconds = (n_chars / 8.0) + 1.0
    total_samples = wav.shape[-1]

    window = max(1, int(0.02 * sr))
    start_sample = int(est_seconds * sr)
    if start_sample < total_samples - window:
        peak = wav.abs().max().item() or 1.0
        silence_thresh = 0.025
        gap_windows_needed = max(1, int(0.25 * sr / window))
        search = wav[..., start_sample:]
        n_windows = search.shape[-1] // window
        if n_windows > 0:
            envelope = search[..., : n_windows * window].reshape(-1, window).abs().mean(dim=-1) / peak
            run = 0
            cut_window = None
            for i in range(len(envelope)):
                if envelope[i] < silence_thresh:
                    run += 1
                    if run >= gap_windows_needed:
                        cut_window = i - run + 1
                        break
                else:
                    run = 0
            if cut_window is not None:
                cut_sample = start_sample + cut_window * window
                fade_samples = min(int(0.15 * sr), cut_sample)
                wav = wav[..., :cut_sample].clone()
                if fade_samples > 0:
                    fade = torch.linspace(1.0, 0.0, fade_samples)
                    wav[..., -fade_samples:] *= fade
                return wav

    # Fallback: no clean silence gap found (e.g. the hallucination runs
    # straight on with no pause) -- same hard-duration idea as v1, just
    # using the same (slightly tighter) estimate as the search above.
    max_seconds = est_seconds + 1.0
    max_samples = int(max_seconds * sr)
    if total_samples <= max_samples:
        return wav
    fade_samples = min(int(0.15 * sr), max_samples)
    wav = wav[..., :max_samples].clone()
    if fade_samples > 0:
        fade = torch.linspace(1.0, 0.0, fade_samples)
        wav[..., -fade_samples:] *= fade
    return wav


def generate(job):
    prompt_path = None
    frames_dir = None
    try:
        values = job["input"]
        mode = values.get("mode")

        # --- duo_composite: build the 2 static "who's speaking" composite
        # images (background + both characters, no audio/TTS involved at
        # all) used as the *input* to WaveSpeedAI for the new duo pipeline
        # (real video generation per speaking turn, instead of the old
        # cutout+rotate rig). No prompt_audio_url needed for this mode.
        if mode == "duo_composite":
            background_b64 = values.get("background_base64")
            background_color = values.get("background_color")
            frames_dir = tempfile.mkdtemp(prefix="viseme_composite_")
            background_path = os.path.join(frames_dir, "background.png")
            if background_color:
                from PIL import Image as _PILImage
                _PILImage.new("RGB", viseme.DUO_CANVAS, background_color).save(background_path)
            elif background_b64:
                with open(background_path, "wb") as f:
                    f.write(base64.b64decode(background_b64))
            else:
                return {"status": "ERROR", "error": "background_base64 ou background_color requis"}

            composites = viseme.build_duo_composites(background_path)
            out = {}
            for character, img in composites.items():
                png_path = os.path.join(frames_dir, "%s_composite.png" % character)
                img.save(png_path)
                with open(png_path, "rb") as f:
                    out[character] = base64.b64encode(f.read()).decode("utf-8")

            return {"status": "DONE", "composites": out, "mode": "duo_composite"}

        # --- concat_turns_audio (20/21 juillet, chantier "plan continu") : concatène
        # les pistes TTS déjà générées pour chaque réplique d'une scène duo (une par
        # tour de parole, dans l'ordre) en UNE SEULE piste audio continue, et renvoie
        # aussi la durée de chacune. Remplace l'ancienne architecture "1 appel
        # WaveSpeedAI par réplique + recollage vidéo après coup" (voir mode "stitch"
        # ci-dessous, gardé mais plus utilisé par le pipeline duo2) par "1 seul appel
        # WaveSpeedAI pour toute la scène" -- les durées renvoyées ici servent à
        # worker.js à construire un prompt qui décrit précisément, en secondes, quand
        # chaque personnage parle (voir buildDuoTimelinePrompt), pour que le modèle
        # anime le bon personnage au bon moment sur une seule vidéo continue.
        if mode == "concat_turns_audio":
            audio_b64_list = values["audio_base64_list"]
            frames_dir = tempfile.mkdtemp(prefix="viseme_concat_audio_")
            paths = []
            durations = []
            for i, b64 in enumerate(audio_b64_list):
                p = os.path.join(frames_dir, "seg_%d.wav" % i)
                with open(p, "wb") as f:
                    f.write(base64.b64decode(b64))
                paths.append(p)
                probe = subprocess.run(
                    [
                        "ffprobe", "-v", "error",
                        "-show_entries", "format=duration",
                        "-of", "csv=p=0", p,
                    ],
                    capture_output=True, text=True, check=True,
                )
                durations.append(float(probe.stdout.strip()))

            concat_list_path = os.path.join(frames_dir, "concat_list.txt")
            with open(concat_list_path, "w") as f:
                for p in paths:
                    f.write("file '%s'\n" % os.path.abspath(p))
            out_audio_path = os.path.join(frames_dir, "combined.wav")
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-f", "concat", "-safe", "0",
                    "-i", concat_list_path,
                    "-c", "copy",
                    out_audio_path,
                ],
                check=True,
            )
            with open(out_audio_path, "rb") as f:
                combined_b64 = base64.b64encode(f.read()).decode("utf-8")

            return {
                "status": "DONE",
                "audio_base64": combined_b64,
                "durations": durations,
                "mode": "concat_turns_audio",
            }

        # --- stitch: concatenate N already-generated WaveSpeedAI clips (one
        # per duo turn) into a single final video, in order. No TTS/prompt
        # audio needed -- each clip already has its own audio baked in from
        # WaveSpeedAI.
        #
        # 20 juillet -- retour Tristana : un simple concat "cul à cul" créait un jump cut
        # visible à chaque changement de tour de parole ("il y a comme un nouveau plan...
        # une coupe sur un montage"), normal puisque chaque clip est une génération
        # WaveSpeedAI indépendante (fond/éclairage/mouvement légèrement différents d'un
        # clip à l'autre même s'ils partent de la même image source). Remplacé le concat
        # brut par un enchaînement en fondu (xfade vidéo + acrossfade audio, 0.4s) : ça
        # adoucit la transition au lieu de couper net. Ça ne rend pas les 3 clips
        # "identiques" en fond/mouvement (ça resterait 3 générations indépendantes), mais
        # le changement devient un fondu au lieu d'un jump cut brutal.
        if mode == "stitch":
            clip_urls = values["clip_urls"]
            clip_paths = [download_file(url, ".mp4") for url in clip_urls]
            out_video_path = "/content/out_stitched.mp4"

            if len(clip_paths) == 1:
                shutil.copy(clip_paths[0], out_video_path)
            else:
                XFADE = 0.4  # secondes de fondu entre deux clips consécutifs
                durations = []
                for p in clip_paths:
                    probe = subprocess.run(
                        [
                            "ffprobe", "-v", "error",
                            "-show_entries", "format=duration",
                            "-of", "csv=p=0", p,
                        ],
                        capture_output=True, text=True, check=True,
                    )
                    durations.append(float(probe.stdout.strip()))

                inputs = []
                for p in clip_paths:
                    inputs += ["-i", p]

                filter_parts = []
                v_prev, a_prev = "0:v", "0:a"
                cumulative = durations[0]
                for i in range(1, len(clip_paths)):
                    v_out, a_out = "v%d" % i, "a%d" % i
                    fade = min(XFADE, durations[i - 1], durations[i])
                    offset = max(cumulative - fade, 0)
                    filter_parts.append(
                        "[%s][%d:v]xfade=transition=fade:duration=%s:offset=%s[%s]"
                        % (v_prev, i, fade, offset, v_out)
                    )
                    filter_parts.append(
                        "[%s][%d:a]acrossfade=d=%s[%s]" % (a_prev, i, fade, a_out)
                    )
                    v_prev, a_prev = v_out, a_out
                    cumulative = offset + durations[i]

                cmd = ["ffmpeg", "-y"] + inputs + [
                    "-filter_complex", ";".join(filter_parts),
                    "-map", "[%s]" % v_prev, "-map", "[%s]" % a_prev,
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-c:a", "aac",
                    out_video_path,
                ]
                subprocess.run(cmd, check=True)

            with open(out_video_path, "rb") as f:
                video_b64 = base64.b64encode(f.read()).decode("utf-8")

            return {"status": "DONE", "video_base64": video_b64, "mode": "stitch"}

        # --- add_music: mixe une musique IA (générée dans l'app, voir /generate-music) sur une
        # vidéo duo déjà recollée (mode "stitch"). Deux placements possibles (choisis par
        # Tristana dans l'app, 18 juillet) :
        #   - "background" : la musique joue en fond, en boucle si besoin, sous le dialogue
        #     existant, à un volume réglable (slider "Volume" de l'app, 0.0 à 1.0).
        #   - "intro" : la musique joue seule pendant quelques secondes (image figée sur la
        #     première frame de la vidéo) AVANT que la vidéo (avec son propre son) ne démarre.
        if mode == "add_music":
            video_url = values["video_url"]
            music_url = values["music_url"]
            placement = values.get("placement", "background")
            volume = float(values.get("volume", 0.5))
            frames_dir = tempfile.mkdtemp(prefix="viseme_addmusic_")
            video_path = download_file(video_url, ".mp4")
            music_path = download_file(music_url, ".mp3")
            out_video_path = "/content/out_with_music.mp4"

            if placement == "intro":
                intro_seconds = 4
                probe = subprocess.run(
                    [
                        "ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=width,height,r_frame_rate",
                        "-of", "csv=p=0", video_path,
                    ],
                    capture_output=True, text=True, check=True,
                )
                w_str, h_str, fr_str = probe.stdout.strip().split(",")
                frame_path = os.path.join(frames_dir, "first_frame.png")
                subprocess.run(
                    ["ffmpeg", "-y", "-i", video_path, "-frames:v", "1", frame_path],
                    check=True,
                )
                intro_path = os.path.join(frames_dir, "intro.mp4")
                subprocess.run(
                    [
                        "ffmpeg", "-y",
                        "-loop", "1", "-i", frame_path,
                        "-i", music_path,
                        "-t", str(intro_seconds),
                        "-vf", "scale=%s:%s,fps=%s" % (w_str, h_str, fr_str),
                        "-af", "volume=%s,afade=t=out:st=%s:d=0.5" % (volume, max(intro_seconds - 0.5, 0)),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-c:a", "aac",
                        "-shortest",
                        intro_path,
                    ],
                    check=True,
                )
                concat_list_path = os.path.join(frames_dir, "concat_list.txt")
                with open(concat_list_path, "w") as f:
                    f.write("file '%s'\n" % os.path.abspath(intro_path))
                    f.write("file '%s'\n" % os.path.abspath(video_path))
                subprocess.run(
                    [
                        "ffmpeg", "-y",
                        "-f", "concat", "-safe", "0",
                        "-i", concat_list_path,
                        "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-c:a", "aac",
                        out_video_path,
                    ],
                    check=True,
                )
            else:
                subprocess.run(
                    [
                        "ffmpeg", "-y",
                        "-i", video_path,
                        "-stream_loop", "-1", "-i", music_path,
                        "-filter_complex",
                        "[1:a]volume=%s[music];[0:a][music]amix=inputs=2:duration=first:dropout_transition=2[aout]" % volume,
                        "-map", "0:v", "-map", "[aout]",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-c:a", "aac",
                        "-shortest",
                        out_video_path,
                    ],
                    check=True,
                )

            with open(out_video_path, "rb") as f:
                video_b64 = base64.b64encode(f.read()).decode("utf-8")

            return {"status": "DONE", "video_base64": video_b64, "mode": "add_music"}

        prompt_audio_url = values["prompt_audio_url"]
        prompt_path = download_file(prompt_audio_url, ".wav")
        cosy = get_model()

        if mode == "duo":
            # Two characters (Zuzu/Titu) taking turns in the same frame,
            # over an AI-generated background.
            turns_in = values["turns"]
            background_b64 = values["background_base64"]

            frames_dir = tempfile.mkdtemp(prefix="viseme_duo_frames_")

            background_path = os.path.join(frames_dir, "background.png")
            with open(background_path, "wb") as f:
                f.write(base64.b64decode(background_b64))

            turns = []
            for i, t in enumerate(turns_in):
                character = t["character"].strip().lower()
                turn_text = t["text"]
                wav, sr = cosy.tts(text=turn_text, prompt=prompt_path)
                wav = _trim_hallucinated_tail(wav, sr, turn_text)
                turn_audio_path = os.path.join(frames_dir, "turn_%02d.wav" % i)
                torchaudio.save(turn_audio_path, wav, sr)
                turns.append({"character": character, "audio_path": turn_audio_path})

            out_video_path = "/content/out.mp4"
            viseme.animate_duo(turns, background_path, out_video_path, frames_dir)

            with open(out_video_path, "rb") as f:
                video_b64 = base64.b64encode(f.read()).decode("utf-8")

            return {"status": "DONE", "video_base64": video_b64, "mode": "duo"}

        text = values["text"]
        character = values.get("character")

        wav, sr = cosy.tts(text=text, prompt=prompt_path)
        wav = _trim_hallucinated_tail(wav, sr, text)

        out_audio_path = "/content/out.wav"
        torchaudio.save(out_audio_path, wav, sr)

        if character:
            character = character.strip().lower()
            frames_dir = tempfile.mkdtemp(prefix="viseme_frames_")
            out_video_path = "/content/out.mp4"

            viseme.animate(out_audio_path, character, out_video_path, frames_dir)

            with open(out_video_path, "rb") as f:
                video_b64 = base64.b64encode(f.read()).decode("utf-8")

            return {"status": "DONE", "video_base64": video_b64, "character": character}

        with open(out_audio_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode("utf-8")

        return {"status": "DONE", "audio_base64": audio_b64, "sample_rate": sr}
    except Exception as e:
        import traceback
        return {"status": "FAILED", "error": str(e), "traceback": traceback.format_exc()}
    finally:
        try:
            if prompt_path and os.path.exists(prompt_path):
                os.remove(prompt_path)
        except Exception:
            pass
        try:
            if frames_dir and os.path.exists(frames_dir):
                shutil.rmtree(frames_dir)
        except Exception:
            pass


runpod.serverless.start({"handler": generate})
