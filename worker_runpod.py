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


# 28 juillet -- Tristana a comparé une vidéo doublée par ce pipeline à la vidéo Seedance
# d'origine (dont la voix a été extraite pour le clonage) : la voix générée sonne beaucoup
# moins dynamique/vivante -- attendu dans une certaine mesure, la synthèse cosyvoice2-eu génère
# sa propre prosodie et ne reproduit pas l'énergie de la performance d'origine, juste le timbre.
# Le package documentait un contrôle de style expérimental : un préfixe texte suivi du token
# spécial "<|endofprompt|>", censé être une instruction NON prononcée pour le modèle (avant le
# texte réellement synthétisé). 30 juillet -- Tristana a entendu ce préfixe lu à voix haute mot
# pour mot ("Parle avec entrain et dynamisme, comme si tu racontais quelque chose de génial à un
# ami...") au lieu d'être silencieux : le token <|endofprompt|> n'est visiblement PAS interprété
# comme un token de contrôle par ce package (cosyvoice2-eu, wrapper "zero-shot" simple autour du
# modèle -- pas l'API instruct dédiée de CosyVoice2). Retiré : mieux vaut une voix un peu plus
# plate qu'une phrase parasite récitée avant chaque réplique. _STYLE_PROMPT gardé en commentaire
# si jamais une vraie API instruct devient disponible plus tard.
# _STYLE_PROMPT = (
#     "Parle avec entrain et dynamisme, comme si tu racontais quelque chose de génial à un ami "
#     "avec le sourire. <|endofprompt|> "
# )


def _with_style_prompt(text):
    return text


# 29 juillet -- tâche #211 (Tristana : "passe au curseur pour ajuster les voix en aigu et en
# grave, et en rapide et lent"). cosyvoice2-eu expose un vrai paramètre `speed` sur cosy.tts()
# (voir l'option CLI --speed documentée sur PyPI) -- passé directement plus bas. En revanche le
# modèle n'expose AUCUN contrôle de hauteur (pitch) : ni sur cosy.tts(), ni sur la CLI. Pour que
# le curseur "Tonalité : Aigu <-> Grave" ait un vrai effet, on applique donc un pitch-shift en
# post-traitement ffmpeg (déjà installé, voir Dockerfile) sur le wav généré, via le montage
# classique asetrate (change la hauteur ET la vitesse de lecture) + atempo (compense la vitesse
# pour revenir à la durée d'origine) -- donc seule la hauteur change au final, pas la durée.
def _pitch_shift_wav(wav, sr, semitones):
    if abs(semitones) < 0.05:
        return wav, sr
    factor = 2.0 ** (semitones / 12.0)
    new_rate = max(1000, int(round(sr * factor)))
    tempo = 1.0 / factor
    # Le filtre ffmpeg "atempo" n'accepte qu'un facteur entre 0.5 et 2.0 par étage -- on chaîne
    # plusieurs étages si le facteur de compensation sort de cette plage (pas le cas pour la
    # plage de tonalité modeste utilisée ici, +/-4 demi-tons, mais on reste robuste).
    stages = []
    remaining = tempo
    while remaining > 2.0:
        stages.append(2.0)
        remaining /= 2.0
    while remaining < 0.5:
        stages.append(0.5)
        remaining /= 0.5
    stages.append(remaining)
    atempo_chain = ",".join("atempo=%.6f" % s for s in stages)

    in_path = tempfile.mktemp(suffix=".wav")
    out_path = tempfile.mktemp(suffix=".wav")
    try:
        torchaudio.save(in_path, wav, sr)
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", in_path,
                "-af", "asetrate=%d,aresample=%d,%s" % (new_rate, sr, atempo_chain),
                out_path,
            ],
            check=True, capture_output=True,
        )
        shifted, shifted_sr = torchaudio.load(out_path)
        return shifted, shifted_sr
    finally:
        for p in (in_path, out_path):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass


def _normalize_loudness(wav, target_peak=0.95, max_gain=12.0):
    """30 juillet -- retour de Tristana : "le son est super bas alors que j'ai le son a fond
    sur l'iPhone". CosyVoice2-EU ne sort aucun gain/normalisation -- le wav brut peut avoir un
    pic tres en dessous du max (0 dB), donc meme telephone a fond ca reste discret. On applique
    un gain lineaire simple base sur le pic (peak normalize) pour remonter le volume, plafonne
    pour ne pas amplifier du bruit si le pic est anormalement minuscule, et on ne baisse jamais
    (uniquement remonter un TTS trop discret)."""
    peak = wav.abs().max().item()
    if peak < 1e-6:
        return wav
    gain = min(target_peak / peak, max_gain)
    if gain > 1.01:
        wav = wav * gain
    return wav


# 30 juillet -- retour de Tristana (encore) : "Oh Dieu que c'est beau" -> "hohoho ho que
# c'est beau", puis "J'ai une super maman" -> premier mot rendu comme un grognement informe
# ("haaaa"), pas du tout "j'ai". Deux essais differents, meme symptome : le tout debut de la
# synthese (les toutes premieres syllabes) part en vrille -- CosyVoice2-EU zero-shot a
# visiblement besoin d'un instant pour "accrocher" la prosodie de la voix de reference avant de
# produire un son correct, et ce tout premier instant est justement celui qu'on entend en entier
# quand la phrase commence pile sur un mot court/exclamatif. _trim_hallucinated_head ne peut rien
# y faire seul : il ne coupe qu'apres un silence net, or ici il n'y a PAS de silence (le grognement
# s'enchaine directement sur la suite). Palliatif : on fait precede le texte reel d'un petit mot
# d'amorce neutre ("Alors, ") qui absorbe ce "faux depart" du modele a la place du vrai texte --
# le silence naturel apres la virgule donne enfin a _trim_hallucinated_head une vraie coupure nette
# a reperer et a supprimer. Si jamais la coupe ne se declenche pas (garde-fous trop stricts), on
# degrade proprement : la phrase sort juste avec un "Alors," en trop au debut, jamais du charabia.
_TTS_PRIMER = "Alors, "


def _prime_text_for_tts(text):
    return _TTS_PRIMER + (text or "")


def _normalize_text_for_tts(text):
    """30 juillet -- task #217 : Tristana a rapporte "j'ai une super maman" prononce
    "JA une super maman" par CosyVoice2-EU. Cause probable : le frontend texte-vers-
    phonemes du modele traite l'apostrophe droite ' (U+0027) plus comme une coupure/
    guillemet que comme un marqueur de liaison francaise, ce qui casse l'elision (j'ai,
    j'aime, qu'il, l'ai, aujourd'hui...) et fait sonner la syllabe comme deux mots separes.
    Remplacement par l'apostrophe typographique U+2019 (') avant synthese -- fix standard
    pour les frontends TTS/regles francaises (espeak et la plupart des normaliseurs FR la
    reconnaissent explicitement comme marqueur d'elision, contrairement a l'apostrophe
    droite)."""
    if not text:
        return text
    return text.replace("'", "’")


def _trim_hallucinated_head(wav, sr, text):
    """30 juillet -- variante "en tete" du fix ci-dessous (task #217) : Tristana a vu
    Snoopy dire "HAHAHA C'est vrai, je l'aime aussi." alors que "hahaha" n'etait pas dans
    le script. Meme cause probable que la hallucination de fin (CosyVoice2-EU zero-shot qui
    emprunte un peu de prosodie/contenu du clip de reference au demarrage), donc meme
    remede applique en miroir : chercher une coupure de silence nette (~200ms) tres tot
    dans l'audio, et si ce qui suit cette coupure a une duree plausible pour le texte reel,
    jeter tout ce qu'il y a avant. Volontairement conservateur (fenetre de recherche courte,
    exige du bruit AVANT le silence, et garde-fou sur la duree restante) : mieux vaut laisser
    passer une hallucination que couper un vrai debut de phrase."""
    n_chars = max(len((text or "").strip()), 1)
    est_seconds = (n_chars / 8.0) + 1.0
    total_samples = wav.shape[-1]

    window = max(1, int(0.02 * sr))
    max_head_seconds = min(1.2, est_seconds * 0.5)
    max_head_sample = int(max_head_seconds * sr)
    if max_head_sample < window * 2 or max_head_sample >= total_samples:
        return wav

    peak = wav.abs().max().item() or 1.0
    silence_thresh = 0.025
    gap_windows_needed = max(1, int(0.2 * sr / window))
    search = wav[..., :max_head_sample]
    n_windows = search.shape[-1] // window
    if n_windows == 0:
        return wav
    envelope = search[..., : n_windows * window].reshape(-1, window).abs().mean(dim=-1) / peak

    min_noise_windows = max(1, int(0.08 * sr / window))
    noise_run = 0
    run = 0
    cut_window = None
    for i in range(len(envelope)):
        if envelope[i] < silence_thresh:
            run += 1
            if noise_run >= min_noise_windows and run >= gap_windows_needed:
                cut_window = i + 1
                break
        else:
            noise_run += 1
            run = 0
    if cut_window is None:
        return wav

    cut_sample = cut_window * window
    remaining_seconds = (total_samples - cut_sample) / sr
    if remaining_seconds < est_seconds * 0.6:
        return wav

    fade_samples = min(int(0.05 * sr), cut_sample)
    wav = wav[..., cut_sample:].clone()
    if fade_samples > 0:
        fade = torch.linspace(0.0, 1.0, fade_samples)
        wav[..., :fade_samples] *= fade
    return wav


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


# 30 juillet, chantier #190 (multilingue reel) : construit un fichier .srt classique a partir
# d'une liste de "cues" {text, start, end} (secondes, floats) -- les timings viennent directement
# de scene.turnDurations (worker.js, meme timeline que celle utilisee pour le lip-sync), donc les
# sous-titres tombent exactement sur les bonnes repliques sans nouveau calcul de synchro.
def _seconds_to_srt_timestamp(seconds):
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000))
    hours, rem_ms = divmod(total_ms, 3600000)
    minutes, rem_ms = divmod(rem_ms, 60000)
    secs, ms = divmod(rem_ms, 1000)
    return "%02d:%02d:%02d,%03d" % (hours, minutes, secs, ms)


def _build_srt(cues):
    lines = []
    for i, cue in enumerate(cues, start=1):
        text = (cue.get("text") or "").strip()
        if not text:
            continue
        lines.append(str(i))
        lines.append("%s --> %s" % (
            _seconds_to_srt_timestamp(float(cue.get("start", 0))),
            _seconds_to_srt_timestamp(float(cue.get("end", 0))),
        ))
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def generate(job):
    prompt_path = None
    frames_dir = None
    try:
        values = job["input"]
        mode = values.get("mode")
        # 29 juillet -- tâche #211 : réglages par défaut (utilisés par le mode solo, et comme
        # repli pour un tour de duo qui n'a pas ses propres speed/pitch_semitones -- voir plus bas).
        default_speed = float(values.get("speed", 1.0) or 1.0)
        default_pitch_semitones = float(values.get("pitch_semitones", 0.0) or 0.0)

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
            # 21 juillet -- fix "voix hachées" + "accent bizarre" (retour Tristana sur le
            # premier test du plan continu). Root cause à deux volets, trouvée en repérant un
            # pic d'amplitude anormal exactement à la jonction entre 2 répliques dans la piste
            # finale :
            # 1) Concaténer les segments TTS bruts avec ffmpeg -c copy (aucun ré-encodage)
            #    laisse un saut brutal d'amplitude à chaque jonction entre répliques (les 2
            #    pistes ne se rejoignent jamais exactement en phase) -- audible comme un clic
            #    à chaque changement de tour de parole = "haché".
            # 2) -c copy suppose un format strictement identique (sample rate, canaux) sur
            #    TOUS les segments d'entrée -- rien ne garantit que CosyVoice2-EU renvoie
            #    exactement le même sample rate à chaque appel TTS ; un copy silencieux d'un
            #    segment à un sample rate différent du premier segment se joue à la mauvaise
            #    vitesse/hauteur, ce qui peut sonner comme un "accent" ou une voix changée.
            # Fix : on force maintenant un sample rate/nombre de canaux identiques sur CHAQUE
            # segment (ré-encodage, plus de simple copy) et on ajoute un fondu de 15ms en
            # entrée/sortie de chaque segment -- imperceptible sur la parole, mais supprime le
            # saut brutal à la jonction.
            target_sr = 24000
            fade_s = 0.015
            for i, b64 in enumerate(audio_b64_list):
                raw_path = os.path.join(frames_dir, "raw_%d.wav" % i)
                with open(raw_path, "wb") as f:
                    f.write(base64.b64decode(b64))
                probe = subprocess.run(
                    [
                        "ffprobe", "-v", "error",
                        "-show_entries", "format=duration",
                        "-of", "csv=p=0", raw_path,
                    ],
                    capture_output=True, text=True, check=True,
                )
                dur = float(probe.stdout.strip())
                durations.append(dur)

                p = os.path.join(frames_dir, "seg_%d.wav" % i)
                fade_out_start = max(0.0, dur - fade_s)
                subprocess.run(
                    [
                        "ffmpeg", "-y", "-i", raw_path,
                        "-ar", str(target_sr), "-ac", "1",
                        "-af", "afade=t=in:st=0:d=%.3f,afade=t=out:st=%.3f:d=%.3f" % (
                            fade_s, fade_out_start, fade_s,
                        ),
                        p,
                    ],
                    check=True,
                )
                paths.append(p)

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

        # --- burn_subtitles (30 juillet, chantier #190) : incruste des sous-titres (.srt genere a
        # la volee) sur une vidéo déjà générée, via le filtre ffmpeg "subtitles" (libass). Appelé
        # par worker.js sur le clip d'UNE scène (avant recollage multi-scènes), avec les cues
        # {text, start, end} calculées à partir des durées réelles de chaque réplique (mêmes
        # durées que celles utilisées pour le lip-sync) -- donc pas de nouveau calcul de synchro
        # à faire ici, juste écrire le .srt et laisser ffmpeg l'incruster.
        # Police forcée à "Noto Sans CJK SC" (voir Dockerfile, fonts-noto-cjk) : cette famille
        # couvre à la fois le latin (français/anglais/espagnol/portugais) ET le chinois/japonais
        # dans UNE seule police, donc pas de sous-titres "tofu" (carrés vides) selon la langue
        # choisie -- si le Dockerfile n'a pas encore été redéployé avec ce paquet, libass retombe
        # sur une police par défaut du système (les sous-titres latins restent lisibles, seuls les
        # sous-titres chinois/japonais seraient concernés).
        if mode == "burn_subtitles":
            video_url = values["video_url"]
            cues = values.get("cues") or []
            video_path = download_file(video_url, ".mp4")
            out_video_path = "/content/out_subtitled.mp4"

            srt_body = _build_srt(cues)
            if not srt_body.strip():
                shutil.copy(video_path, out_video_path)
            else:
                subs_dir = tempfile.mkdtemp(prefix="subs_")
                srt_path = os.path.join(subs_dir, "subs.srt")
                with open(srt_path, "w", encoding="utf-8") as f:
                    f.write(srt_body)
                style = (
                    "FontName=Noto Sans CJK SC,FontSize=22,PrimaryColour=&H00FFFFFF,"
                    "OutlineColour=&H00000000,BorderStyle=1,Outline=2,Shadow=0,"
                    "Alignment=2,MarginV=48"
                )
                subprocess.run(
                    [
                        "ffmpeg", "-y", "-i", video_path,
                        "-vf", "subtitles=%s:force_style='%s'" % (srt_path, style),
                        "-c:a", "copy",
                        out_video_path,
                    ],
                    check=True,
                )

            with open(out_video_path, "rb") as f:
                video_b64 = base64.b64encode(f.read()).decode("utf-8")

            return {"status": "DONE", "video_base64": video_b64, "mode": "burn_subtitles"}

        # --- crop_to_aspect (30 juillet, chantier #225, corrige le meme jour) : le pipeline
        # Avatar classique (Wan2.2-S2V / wavespeedGenerate cote worker.js) n'a AUCUN parametre
        # aspect_ratio -- la video generee suit toujours les dimensions de l'image source. Pour
        # que "Format" (menu 1) ait un vrai effet ici aussi (comme deja le cas pour Premium 3D via
        # l'aspect_ratio natif de Seedance), on adapte l'image AVANT l'appel WaveSpeedAI.
        # Version initiale (rognage "cover") coupait le bas/haut du personnage -> corrige en
        # rognage "contain" + fond flouté : on GARDE l'image entiere (aucun pixel du personnage
        # perdu), on l'agrandit sur un nouveau canevas au bon ratio, et on remplit l'espace ajoute
        # avec une version floutee/zoomee de la meme image (technique "fond flou", comme les
        # stories Instagram) plutot que des bandes noires.
        if mode == "crop_to_aspect":
            from PIL import Image as _PILImage, ImageFilter

            image_url = values["image_url"]
            aspect_ratio = values.get("aspect_ratio", "9:16")
            ar_map = {"9:16": 9.0 / 16.0, "1:1": 1.0, "16:9": 16.0 / 9.0}
            target_ratio = ar_map.get(aspect_ratio, 9.0 / 16.0)  # largeur / hauteur

            img_path = download_file(image_url, ".png")
            img = _PILImage.open(img_path).convert("RGB")
            w, h = img.size
            current_ratio = w / h

            if abs(current_ratio - target_ratio) < 0.01:
                canvas_w, canvas_h = w, h
            elif current_ratio > target_ratio:
                # Image deja plus large que la cible -- on garde la largeur d'origine et on
                # agrandit la hauteur du canevas (le personnage garde sa taille, du vide apparait
                # en haut/bas, comble par le fond floute).
                canvas_w = w
                canvas_h = max(1, int(round(w / target_ratio)))
            else:
                # Image plus haute que la cible (cas portrait -> carre/paysage, le plus frequent
                # ici) -- on garde la hauteur d'origine (donc le personnage entier, tete-pieds,
                # reste visible) et on agrandit la largeur du canevas.
                canvas_h = h
                canvas_w = max(1, int(round(h * target_ratio)))

            canvas = _PILImage.new("RGB", (canvas_w, canvas_h))

            # Fond : version floutee de l'image, agrandie en mode "cover" pour remplir tout le
            # canevas sans bande noire ni unie.
            cover_scale = max(canvas_w / w, canvas_h / h)
            bg_w = max(canvas_w, int(round(w * cover_scale)) + 2)
            bg_h = max(canvas_h, int(round(h * cover_scale)) + 2)
            bg = img.resize((bg_w, bg_h), _PILImage.LANCZOS)
            bg = bg.filter(ImageFilter.GaussianBlur(35))
            bg_left = (bg_w - canvas_w) // 2
            bg_top = (bg_h - canvas_h) // 2
            bg = bg.crop((bg_left, bg_top, bg_left + canvas_w, bg_top + canvas_h))
            canvas.paste(bg, (0, 0))

            # Premier plan : l'image ENTIERE, taille d'origine, centree -- rien n'est coupe.
            fg_left = (canvas_w - w) // 2
            fg_top = (canvas_h - h) // 2
            canvas.paste(img, (fg_left, fg_top))

            out_path = "/content/out_cropped.png"
            canvas.save(out_path, "PNG")
            with open(out_path, "rb") as f:
                image_b64 = base64.b64encode(f.read()).decode("utf-8")

            return {"status": "DONE", "image_base64": image_b64, "mode": "crop_to_aspect"}

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
            elif placement == "outro":
                # 28 juillet -- 3e placement demandé par Tristana en plus de "intro"/"background" :
                # la musique joue seule à la FIN, après la vidéo (miroir de "intro" mais sur la
                # dernière image figée au lieu de la première, et un fondu d'entrée au lieu de sortie).
                outro_seconds = 4
                probe = subprocess.run(
                    [
                        "ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=width,height,r_frame_rate",
                        "-of", "csv=p=0", video_path,
                    ],
                    capture_output=True, text=True, check=True,
                )
                w_str, h_str, fr_str = probe.stdout.strip().split(",")
                dur_probe = subprocess.run(
                    [
                        "ffprobe", "-v", "error",
                        "-show_entries", "format=duration",
                        "-of", "csv=p=0", video_path,
                    ],
                    capture_output=True, text=True, check=True,
                )
                video_duration = float(dur_probe.stdout.strip())
                frame_path = os.path.join(frames_dir, "last_frame.png")
                subprocess.run(
                    ["ffmpeg", "-y", "-ss", str(max(video_duration - 0.1, 0)), "-i", video_path, "-frames:v", "1", frame_path],
                    check=True,
                )
                outro_path = os.path.join(frames_dir, "outro.mp4")
                subprocess.run(
                    [
                        "ffmpeg", "-y",
                        "-loop", "1", "-i", frame_path,
                        "-i", music_path,
                        "-t", str(outro_seconds),
                        "-vf", "scale=%s:%s,fps=%s" % (w_str, h_str, fr_str),
                        "-af", "volume=%s,afade=t=in:st=0:d=0.5" % volume,
                        "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-c:a", "aac",
                        "-shortest",
                        outro_path,
                    ],
                    check=True,
                )
                concat_list_path = os.path.join(frames_dir, "concat_list.txt")
                with open(concat_list_path, "w") as f:
                    f.write("file '%s'\n" % os.path.abspath(video_path))
                    f.write("file '%s'\n" % os.path.abspath(outro_path))
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
            elif placement == "dub":
                # 28 juillet, chantier "doublage" (#141) : la vidéo Seedance source est MUETTE
                # (aucun flux audio, generate_audio=false côté Seedance) -- il n'y a donc rien à
                # mixer avec amix (ça plantait : "[0:a]" ne matchait aucun stream -> ffmpeg
                # échouait -> le pipeline retombait silencieusement sur la vidéo muette). Ici on
                # colle simplement notre voix TTS comme unique piste audio.
                subprocess.run(
                    [
                        "ffmpeg", "-y",
                        "-i", video_path,
                        "-i", music_path,
                        "-map", "0:v", "-map", "1:a",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-af", "volume=%s" % volume,
                        "-c:a", "aac",
                        "-shortest",
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
                wav, sr = cosy.tts(text=_with_style_prompt(_normalize_text_for_tts(_prime_text_for_tts(turn_text))), prompt=prompt_path)
                wav = _trim_hallucinated_head(wav, sr, turn_text)
                wav = _trim_hallucinated_tail(wav, sr, turn_text)
                wav = _normalize_loudness(wav)
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
        # 29 juillet -- tâche #211 : cette branche (audio seul, ou audio + viseme si un
        # `character` est fourni) est le VRAI point d'entrée TTS partagé, utilisé aussi bien pour
        # le solo que pour chaque réplique du pipeline duo -- voir worker.js runpodTTS() et
        # l'appel runpodSubmit() par tour dans /duo-generate. C'est donc ici, et uniquement ici,
        # qu'il faut appliquer speed/pitch pour que les curseurs aient un effet partout.
        # (default_speed/default_pitch_semitones déjà lus depuis `values` tout en haut de generate().)

        # 30 juillet -- retour de Tristana : même curseurs au milieu (= speed 1.0, donc
        # supposé neutre), les voix sonnaient "moins fun" / moins Pixar qu'avant #211. Cause :
        # on passait `speed=1.0` explicitement à cosy.tts() dans TOUS les cas, alors qu'avant
        # #211 l'appel ne passait jamais ce paramètre -- et le contrôle de durée activé par
        # `speed=` (même à 1.0) semble légèrement lisser la prosodie naturelle du modèle. On ne
        # passe donc `speed=` que si le réglage s'écarte vraiment du neutre, pour retrouver le
        # rendu d'origine quand les curseurs sont au centre.
        if abs(default_speed - 1.0) < 0.03:
            wav, sr = cosy.tts(text=_with_style_prompt(_normalize_text_for_tts(_prime_text_for_tts(text))), prompt=prompt_path)
        else:
            wav, sr = cosy.tts(text=_with_style_prompt(_normalize_text_for_tts(_prime_text_for_tts(text))), prompt=prompt_path, speed=default_speed)
        wav = _trim_hallucinated_head(wav, sr, text)
        wav = _trim_hallucinated_tail(wav, sr, text)
        wav, sr = _pitch_shift_wav(wav, sr, default_pitch_semitones)
        wav = _normalize_loudness(wav)

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
