import os
import base64
import tempfile
import shutil
import subprocess
import requests
import runpod
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
            background_b64 = values["background_base64"]
            frames_dir = tempfile.mkdtemp(prefix="viseme_composite_")
            background_path = os.path.join(frames_dir, "background.png")
            with open(background_path, "wb") as f:
                f.write(base64.b64decode(background_b64))

            composites = viseme.build_duo_composites(background_path)
            out = {}
            for character, img in composites.items():
                png_path = os.path.join(frames_dir, "%s_composite.png" % character)
                img.save(png_path)
                with open(png_path, "rb") as f:
                    out[character] = base64.b64encode(f.read()).decode("utf-8")

            return {"status": "DONE", "composites": out, "mode": "duo_composite"}

        # --- stitch: concatenate N already-generated WaveSpeedAI clips (one
        # per duo turn) into a single final video, in order. No TTS/prompt
        # audio needed -- each clip already has its own audio baked in from
        # WaveSpeedAI. Re-encodes (rather than stream-copying) so clips with
        # slightly different encoder parameters still concatenate cleanly.
        if mode == "stitch":
            clip_urls = values["clip_urls"]
            frames_dir = tempfile.mkdtemp(prefix="viseme_stitch_")
            clip_paths = [download_file(url, ".mp4") for url in clip_urls]

            concat_list_path = os.path.join(frames_dir, "concat_list.txt")
            with open(concat_list_path, "w") as f:
                for p in clip_paths:
                    f.write("file '%s'\n" % os.path.abspath(p))

            out_video_path = "/content/out_stitched.mp4"
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

            with open(out_video_path, "rb") as f:
                video_b64 = base64.b64encode(f.read()).decode("utf-8")

            return {"status": "DONE", "video_base64": video_b64, "mode": "stitch"}

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
