import os
import base64
import tempfile
import requests
import runpod
import torchaudio

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
    try:
        values = job["input"]
        text = values["text"]
        prompt_audio_url = values["prompt_audio_url"]

        prompt_path = download_file(prompt_audio_url, ".wav")

        cosy = get_model()
        wav, sr = cosy.tts(text=text, prompt=prompt_path)

        out_path = "/content/out.wav"
        torchaudio.save(out_path, wav, sr)

        with open(out_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode("utf-8")

        return {"status": "DONE", "audio_base64": audio_b64, "sample_rate": sr}
    except Exception as e:
        import traceback
        return {"status": "FAILED", "error": str(e), "traceback": traceback.format_exc()}
    finally:
        try:
            if os.path.exists(prompt_path):
                os.remove(prompt_path)
        except Exception:
            pass


runpod.serverless.start({"handler": generate})
