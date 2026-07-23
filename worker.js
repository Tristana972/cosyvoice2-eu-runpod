// EasyVideo IA — Serveur intermédiaire (Cloudflare Worker)
// Reçoit une demande de l'app, appelle HeyGen avec la clé API secrète,
// renvoie le résultat à l'app. La clé API n'est JAMAIS visible côté téléphone.

// Zuzu et Titu sont routés vers le pipeline maison (CosyVoice2-EU + WaveSpeedAI)
// au lieu de HeyGen, car ce sont des mascottes trop stylisées pour MuseTalk (voir NOTES_migration_heygen.md).
const RUNPOD_ENDPOINT_ID = 'rg4z5rswwl2ibu';
// IMPORTANT: this must be the real human reference speaker sample used for
// zero-shot voice cloning, NOT audio/original-fr.wav — that file is the demo
// site's deliberately-bad "Baseline (No Fine-tuning)" example, explicitly
// labelled "Strong English Accent / Unnatural Prosody" on the CosyVoice2-EU
// demo page (it's a synthesized *output* sample used to show improvement,
// not a voice prompt). Using it as the cloning reference is almost
// certainly why generated French audio had a bad accent. The correct
// reference — the actual "French reference voice" / "Prompt Speaker" audio
// shown in the demo's architecture diagram — is common_voice_fr_40952142.wav
// (a real Mozilla Common Voice FR recording), served from the same site.
const RUNPOD_REFERENCE_VOICE_URL = 'https://hi-paris.github.io/CosyVoice2-EU/common_voice_fr_40952142.wav';
const RUNPOD_CHARACTERS = {
  '3fe7a4551b0d4909bdef9a1aaf0431aa': 'zuzu',
  '4a203ebf82c64475b36cb498516acdca': 'titu',
};

// WaveSpeedAI (Wan2.2-S2V) : moteur de génération vidéo naturelle (visage + corps),
// utilisé pour le mode solo ET (depuis la migration "Round 4") pour le mode duo.
// Le texte est d'abord synthétisé en français via CosyVoice2-EU sur RunPod
// (appel SANS le champ "character" → renvoie audio_base64 brut, pas de vidéo), puis cet audio est
// envoyé directement en base64 (data URI) à WaveSpeedAI avec l'image de référence du personnage.
// Voir NOTES_backend_technique.md pour le détail des tests de validation (11 juillet)
// et la section "migration duo vers WaveSpeedAI" (13 juillet) pour le nouveau pipeline duo.
const WAVESPEED_CHARACTER_IMAGES = {
  zuzu: 'https://cdn.jsdelivr.net/gh/Tristana972/cosyvoice2-eu-runpod@65d8eef/assets/zuzu_mouth_closed.png',
  titu: 'https://cdn.jsdelivr.net/gh/Tristana972/cosyvoice2-eu-runpod@65d8eef/assets/titu_mouth_closed.png',
};

// Lance une synthèse vocale CosyVoice2-EU sur RunPod (TTS seul, sans animation) et attend le résultat.
// Retourne l'audio en base64 (wav). Utilisé par le mode SOLO (bloquant, une seule réplique).
// Le mode DUO n'utilise PAS cette fonction (elle bloquerait le Worker trop longtemps pour
// plusieurs répliques) — voir runpodSubmit/runpodPoll et la machine à états dans /status.
async function runpodTTS(env, text) {
  const runRes = await fetch(`https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}/run`, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${env.RUNPOD_API_KEY}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ input: { text, prompt_audio_url: RUNPOD_REFERENCE_VOICE_URL } }),
  });
  const runData = await runRes.json();
  const jobId = runData?.id;
  if (!runRes.ok || !jobId) {
    throw new Error('RunPod TTS: impossible de lancer le job — ' + JSON.stringify(runData));
  }

  for (let i = 0; i < 40; i++) {
    await new Promise((resolve) => setTimeout(resolve, 2000));
    const statusRes = await fetch(`https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}/status/${jobId}`, {
      headers: { Authorization: `Bearer ${env.RUNPOD_API_KEY}` },
    });
    const statusData = await statusRes.json();
    if (statusData.status === 'COMPLETED') {
      const output = statusData.output || {};
      if (output.status === 'DONE' && output.audio_base64) {
        return output.audio_base64;
      }
      throw new Error('RunPod TTS: sortie invalide — ' + JSON.stringify(output));
    }
    if (statusData.status === 'FAILED' || statusData.status === 'CANCELLED') {
      throw new Error(`RunPod TTS: job ${statusData.status}`);
    }
  }
  throw new Error('RunPod TTS: délai dépassé (80s)');
}

// Lance une génération vidéo WaveSpeedAI (Wan2.2-S2V) et retourne l'id de prédiction (asynchrone).
// `prompt` (optionnel) est transmis tel quel à l'API — Wan2.2-S2V "peut suivre des prompts texte
// pour contrôler la scène/pose/comportement tout en gardant la synchro audio" (doc WaveSpeedAI),
// utilisé côté duo pour pousser vers plus de mouvement corps entier (voir /status, "wavespeed_submit").
async function wavespeedGenerate(env, imageUrl, audioBase64, prompt) {
  const body = {
    image: imageUrl,
    audio: `data:audio/wav;base64,${audioBase64}`,
    resolution: '480p',
    seed: -1,
  };
  if (prompt) body.prompt = prompt;
  const res = await fetch('https://api.wavespeed.ai/api/v3/wavespeed-ai/wan-2.2/speech-to-video', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${env.WAVESPEED_API_KEY}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  const wsId = data?.data?.id;
  if (!res.ok || !wsId) {
    throw new Error('WaveSpeedAI: impossible de lancer la génération — ' + JSON.stringify(data));
  }
  return wsId;
}

// Construit le prompt de mouvement WaveSpeedAI pour une réplique duo, en nommant explicitement
// qui parle (et sa position gauche/droite dans le composite -- Zuzu est toujours à gauche, Titu
// toujours à droite, voir DUO_SLOT_X_FRAC dans viseme.py) et en demandant explicitement à l'autre
// personnage de rester complètement immobile, bouche fermée. Nécessaire car le composite envoyé
// contient TOUJOURS les deux personnages mais une seule piste audio : un prompt générique ne
// précisait pas lequel des deux animer, et le modèle animait la bouche des deux sur le même audio
// (retour Tristana : Zuzu continuait à articuler alors que c'était Titu qui parlait).
//
// Deux correctifs supplémentaires (retour Tristana, même session) :
// 1) "its antennas or tail sway" était générique pour les deux personnages, alors que Zuzu n'a PAS
//    de queue (seulement des antennes) et Titu n'a PAS d'antennes (seulement une queue) -- le modèle
//    a fait apparaître une queue fantôme derrière Zuzu. On rend maintenant l'accessoire spécifique à
//    chaque personnage et on précise explicitement de ne pas ajouter de partie du corps absente.
// 2) La bouche de Titu se mettait à bouger avant même le début de son texte (anticipation). On
//    ajoute une instruction explicite : bouche fermée pendant tout silence en début d'audio, ne
//    bouger les lèvres qu'à l'instant précis où le son de la voix commence.
function buildDuoMotionPrompt(speakerCharacter) {
  const zuzu = {
    name: 'Zuzu',
    pos: 'on the LEFT',
    desc: 'a green alien-like creature with two antennas (it has NO tail)',
    accessory: 'its two antennas sway and bounce',
  };
  const titu = {
    name: 'Titu',
    pos: 'on the RIGHT',
    desc: 'a small blue/teal dog-like creature sitting on all fours, with two small antennas and a tail',
    accessory: 'its tail wags and its antennas sway',
  };
  const speaker = speakerCharacter === 'zuzu' ? zuzu : titu;
  const silent = speakerCharacter === 'zuzu' ? titu : zuzu;
  return (
    `In this image there are two cartoon characters: ${zuzu.name} (${zuzu.desc}, ${zuzu.pos}) and ` +
    `${titu.name} (${titu.desc}, ${titu.pos}). Only ${speaker.name}, ${speaker.pos.toLowerCase()}, ` +
    `is speaking right now -- animate ONLY ${speaker.name}'s mouth to match the speech audio exactly: ` +
    `keep ${speaker.name}'s mouth fully closed and still during any silence at the very start of the ` +
    `audio, and only start moving the lips at the exact instant the speech sound begins, not before. ` +
    `Just as importantly, stop moving ${speaker.name}'s lips and return the mouth to a closed/neutral ` +
    `position the moment the speech audio ends -- do not keep the mouth moving, twitching, or talking ` +
    `during any silence after the sentence is finished. ` +
    `Make ${speaker.name} gently sway and move naturally while talking: head tilts, shoulders and ` +
    `arms move, and ${speaker.accessory} along with the motion. Natural, lively, alive movement, not ` +
    `just the mouth. Do not add or show any body part ${speaker.name} does not have in the image. ` +
    `Meanwhile ${silent.name}, ${silent.pos.toLowerCase()}, is NOT speaking right now and its mouth ` +
    `must be treated as LOCKED closed for the entire video: zero lip movement, zero jaw movement, ` +
    `not even a small twitch or parting of the lips at any point, in any single frame, from the very ` +
    `first frame to the very last -- only extremely subtle idle breathing motion in the body is ` +
    `allowed. ${silent.name} is a completely silent bystander in this clip; do not animate its mouth ` +
    `to match the audio track under any circumstance, even briefly. Keep the ground/floor beneath both characters' feet clean and ` +
    `visually consistent with the rest of the background throughout the video -- do not introduce any ` +
    `new patch, stain, discoloration, dirt, or texture change near their feet that isn't already in ` +
    `the source image. ` +
    `Keep the camera completely static: no zoom in, no zoom out, no pan, no crop, same framing as the ` +
    `source image for the entire video. Both ${zuzu.name} and ${titu.name} must stay fully visible ` +
    `head to feet in every single frame, in the exact same on-screen position as the source image -- ` +
    `do not let any part of either character (especially their legs and feet) move outside the frame ` +
    `or become hidden behind any background element.`
  );
}

// 21 juillet -- retour Tristana : elle avait tapé "(fait un clin d'œil)" dans le texte d'une
// réplique pour que le personnage joue l'action, mais tout le texte tapé part tel quel au TTS et
// se retrouve donc LU à voix haute au lieu d'être joué visuellement (voir NOTES_backend_technique.md,
// 18 juillet). Nouvelle syntaxe : ce qui est entre crochets `[ ... ]` dans le texte d'une réplique
// est retiré avant l'envoi au TTS, et transmis à part comme instruction d'action/mouvement pour ce
// moment précis de la scène (voir buildDuoTimelinePrompt). Couvre aussi bien un geste solo (ex.
// "[fait un clin d'œil]") qu'une action impliquant les deux personnages (ex. "[Titu donne une
// claque à Zuzu qui reprend ses esprits]") -- le texte entre crochets est transmis tel quel au
// modèle vidéo, en langage naturel, sans autre syntaxe imposée que les crochets eux-mêmes.
function extractAction(rawText) {
  const text = rawText || '';
  const actions = [];
  const cleanText = text
    .replace(/\[([^\]]+)\]/g, (_, action) => {
      actions.push(action.trim());
      return '';
    })
    .replace(/\s{2,}/g, ' ')
    .trim();
  return { text: cleanText, action: actions.length ? actions.join('. ') : null };
}

// 20/21 juillet -- chantier "plan continu" (retour Tristana : "il y a comme un nouveau plan...
// une coupe sur un montage" entre chaque réplique). AVANT, chaque réplique déclenchait son
// propre appel WaveSpeedAI (buildDuoMotionPrompt ci-dessus, un composite "qui parle" par appel),
// puis tous les clips étaient recollés bout à bout -- chaque clip étant une génération IA
// indépendante, le fond/éclairage/mouvement dérivait légèrement d'un clip à l'autre, d'où la
// coupure visible à chaque changement de tour de parole. Wan2.2-S2V (WaveSpeedAI) supporte des
// clips jusqu'à 10 minutes en un seul appel : cette fonction construit UN SEUL prompt décrivant
// toute la scène (audio complet concaténé, voir mode "concat_turns_audio" côté RunPod), avec une
// vraie timeline en secondes de qui parle quand, pour un unique appel -- un seul plan continu, ni
// jump cut ni dérive de fond entre répliques. Ne garantit pas à 100% que le modèle respecte
// chaque fenêtre de parole à la lettre (aucun paramètre API ne permet de cibler un visage
// précisément, voir buildDuoMotionPrompt), mais une seule performance continue du modèle a de
// meilleures chances de rester cohérente que 3 générations indépendantes qui "redécident" chacune
// comment animer les deux personnages.
function buildDuoTimelinePrompt(turns, durations) {
  const zuzu = {
    name: 'Zuzu',
    pos: 'on the LEFT',
    desc: 'a green alien-like creature with two antennas (it has NO tail)',
  };
  const titu = {
    name: 'Titu',
    pos: 'on the RIGHT',
    desc: 'a small blue/teal dog-like creature sitting on all fours, with two small antennas and a tail',
  };
  const chars = { zuzu, titu };

  let t = 0;
  const segments = turns.map((turn, i) => {
    const start = t;
    const end = t + (durations[i] || 0);
    t = end;
    const speaker = chars[turn.character];
    const silent = turn.character === 'zuzu' ? titu : zuzu;
    let seg = (
      `From ${start.toFixed(1)}s to ${end.toFixed(1)}s, only ${speaker.name} (${speaker.pos.toLowerCase()}) ` +
      `speaks -- animate ONLY ${speaker.name}'s mouth to match the audio exactly during this window, ` +
      `while ${silent.name} (${silent.pos.toLowerCase()}) stays completely silent with its mouth locked closed`
    );
    // Action/geste explicite pour cette réplique (voir extractAction) -- peut impliquer un seul
    // personnage (un clin d'œil) ou les deux (une claque, un câlin...), transmis tel quel.
    if (turn.action) {
      seg += `. During this same time window, also act this out: ${turn.action}`;
    }
    return seg;
  });

  return (
    `In this image there are two cartoon characters: ${zuzu.name} (${zuzu.desc}, ${zuzu.pos}) and ` +
    `${titu.name} (${titu.desc}, ${titu.pos}). This is a single continuous shot of a full conversation ` +
    `where they take turns speaking, following this exact timeline: ${segments.join('. ')}. ` +
    `Whichever character is not speaking at a given moment must keep its mouth fully closed and still ` +
    `-- zero lip movement, zero jaw movement, not even a small twitch -- only extremely subtle idle ` +
    `breathing motion is allowed while silent. Both characters gently sway and move naturally the ` +
    `whole time even while not speaking: head tilts, shoulders and arms move, ${zuzu.name}'s two ` +
    `antennas sway and bounce, and ${titu.name}'s tail wags and its antennas sway. Do not add or show ` +
    `any body part either character does not have in the image -- no tail on ${zuzu.name}, no extra ` +
    `antennas beyond the two small ones on ${titu.name}. Keep the ground/floor beneath both ` +
    `characters' feet clean and visually consistent with the rest of the background throughout the ` +
    `video -- do not introduce any new patch, stain, discoloration, dirt, or texture change near ` +
    `their feet that isn't already in the source image. Keep the camera completely static: no zoom ` +
    `in, no zoom out, no pan, no crop, same framing as the source image for the entire video. Both ` +
    `${zuzu.name} and ${titu.name} must stay fully visible head to feet in every single frame, in the ` +
    `exact same on-screen position as the source image -- do not let any part of either character ` +
    `(especially their legs and feet) move outside the frame or become hidden behind any background ` +
    `element.`
  );
}

// Enrichit une description de décor (fournie par l'app ou par parseScenarioWithAI) avant de
// l'envoyer à flux-1-schnell, pour deux raisons distinctes remontées par Tristana (20 juillet) :
// 1) "les personnages ne s'animent pas en situation au 2e/3e plan, le fond est trop plat derrière
//    eux" -- un seul fond généré ne peut pas littéralement avoir plusieurs couches 3D pour un coût
//    raisonnable, mais on peut lui demander une VRAIE composition en plans (premier plan/plan
//    intermédiaire/arrière-plan avec de la profondeur de champ et de la perspective) au lieu d'un
//    aplat uniforme derrière les personnages -- gratuit (juste du prompt engineering, aucun appel
//    IA supplémentaire).
// 2) Constaté le même jour : sur un décor parisien, des pots de fleurs générés juste devant
//    l'endroit où les personnages sont plaqués (voir DUO_GROUND_Y_FRAC dans viseme.py) ont donné
//    l'impression que les jambes de Zuzu étaient coupées une fois la vidéo animée par WaveSpeedAI.
//    Le composite lui-même est correct (vérifié : personnage entier, rien de coupé à cette étape),
//    mais rien n'empêchait le fond généré de placer des objets juste dans la zone où les
//    personnages sont ensuite posés. On demande donc explicitement une bande dégagée au sol,
//    au centre, pour qu'ils aient toujours un endroit net où se tenir.
function buildSceneBackgroundPrompt(description) {
  return (
    `${description}. Flat 2D cartoon illustration background, children's animated show art style, ` +
    `bright saturated colors, simple shapes, no photorealism, no visible human characters, animals ` +
    `or faces (just scenery/props) -- this is a background plate for stylized cartoon characters to ` +
    `be composited on top and animated. Compose it with real depth and perspective like an animated ` +
    `film background: a foreground layer with a few larger, closer, slightly softer/blurred details ` +
    `near the bottom edges, a clear midground stage across the lower-center of the image at ground ` +
    `level, and a distant background layer (buildings, landscape, sky) receding toward a horizon or ` +
    `vanishing point -- not a single flat backdrop directly behind where the characters stand. ` +
    `In that foreground layer, place a couple of close, larger foreground props (like a low wall, a ` +
    `hedge, a lamp post, tall grass, or a plant) toward the LEFT and RIGHT edges/corners of the lower ` +
    `third of the image, framing the scene from the sides -- but leave the lower-CENTER ground area ` +
    `clear and unobstructed (no plants, furniture, signs, or other objects placed there) so two ` +
    `standing or sitting characters positioned in that center area are fully visible from head to ` +
    `feet, with nothing directly overlapping their own body.`
  );
}

// Détecte si l'histoire/décor demande une pose particulière (assis, allongé...) plutôt que la
// position debout par défaut des composites RunPod (voir build_duo_composites dans viseme.py,
// qui ne sait construire QUE la position debout — limitation architecturale confirmée). Si
// détecté, une étape supplémentaire (voir stages "pose_submit"/"pose_wait" dans /status) fait
// retoucher la pose du composite par une IA d'édition d'image (Nano Banana 2 / WaveSpeedAI) qui
// préserve l'identité des personnages (testé et validé le 15 juillet sur Zuzu et Titu).
function detectDuoPose(text) {
  if (!text) return null;
  const t = text.toLowerCase();
  if (/assis|assise|assis\(e\)|s'assoi|s'assey|sur (un|le) banc|sur (une|la) chaise|au sol|par terre|accroupi/.test(t)) {
    return 'sitting down naturally (for example on a bench, chair or on the ground), legs relaxed, still fully visible from head to feet';
  }
  if (/allong|couch[ée]|étendu/.test(t)) {
    return 'lying down comfortably in a relaxed pose, still fully visible';
  }
  return null;
}

// Soumet une édition de pose Nano Banana 2 (WaveSpeedAI) sur une image existante, SANS attendre
// le résultat (contrairement à la route de test /pose-test, qui bloque ~60-90s) -- suit le même
// principe non-bloquant que runpodSubmit/runpodPoll, pour ne jamais bloquer le Worker.
async function nanoBananaSubmit(env, imageUrl, prompt) {
  // IMPORTANT (bug constaté le 17 juillet) : sans aspect_ratio explicite, Nano Banana 2 renvoie
  // par défaut une image CARRÉE (1024x1024), alors que le composite d'origine (construit par
  // RunPod) est en format portrait 9:16 comme le reste de la vidéo. WaveSpeedAI (Wan2.2-S2V)
  // doit alors étirer/écraser cette image carrée pour l'animer, ce qui aplatit visiblement les
  // personnages pendant la réplique concernée (retour Tristana : "la 2e partie ils sont
  // aplatis"). On force donc explicitement le même format 9:16 que la vidéo finale.
  const res = await fetch('https://api.wavespeed.ai/api/v3/google/nano-banana-2/edit', {
    method: 'POST',
    headers: { Authorization: `Bearer ${env.WAVESPEED_API_KEY}`, 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt, images: [imageUrl], aspect_ratio: '9:16', resolution: '1k', output_format: 'png' }),
  });
  const data = await res.json();
  const requestId = data?.data?.id;
  if (!res.ok || !requestId) {
    throw new Error('Nano Banana 2: impossible de lancer l\'édition — ' + JSON.stringify(data));
  }
  return requestId;
}

// Vérifie l'état d'une édition Nano Banana 2 déjà soumise (un seul appel, ne boucle pas).
async function nanoBananaPoll(env, requestId) {
  const res = await fetch(`https://api.wavespeed.ai/api/v3/predictions/${requestId}/result`, {
    headers: { Authorization: `Bearer ${env.WAVESPEED_API_KEY}` },
  });
  return res.json();
}

// Transforme une histoire racontée en langage naturel (dictée par l'utilisatrice, mélangeant
// narration + dialogue, ex. "Zuzu et Titu sont devant un coffee shop... Zuzu dit : tu as vu ce qui
// se passe...") en un scénario exploitable par le pipeline existant (contexte visuel du décor +
// liste ordonnée de répliques attribuées à chaque personnage). Sans cette étape, il fallait que
// l'utilisatrice sépare elle-même "une ligne = une réplique par personnage" du "décor" — ce que
// Tristana a demandé de fusionner (14 juillet) pour permettre une vraie histoire avec plusieurs
// échanges, des passages d'action muette, une chute/morale, etc.
// Utilise Cloudflare Workers AI (déjà relié au Worker pour le fond IA, même compte, pas de
// nouvelle clé) plutôt qu'une API LLM tierce payante.
async function parseScenarioWithAI(env, story, cast) {
  const castList = cast.map((c) => `"${c.key}" (${c.name})`).join(' et ');
  const systemPrompt =
    `Tu transformes une histoire racontée en langage naturel par une utilisatrice en un scénario ` +
    `structuré pour une petite vidéo mettant en scène ${castList}. ` +
    `Réponds UNIQUEMENT avec un objet JSON valide, sans aucun texte autour, de la forme : ` +
    `{"context": "description visuelle courte du décor/lieu, en anglais, pour un générateur d'image", ` +
    `"turns": [{"character": "${cast[0].key}", "text": "ce que dit le personnage, en français"}, ...]}. ` +
    `Règles : "turns" ne contient QUE des répliques vraiment parlées par ${cast.map((c) => c.name).join(' ou ')} ` +
    `(pas de narration, pas de didascalies) ; garde-les dans l'ordre de l'histoire ; garde le ton ` +
    `enfantin, drôle et attachant des personnages ; les moments d'action sans parole peuvent être ` +
    `résumés en une courte réplique naturelle du personnage concerné plutôt qu'omis, pour que ` +
    `l'histoire reste compréhensible seulement à l'audio ; le champ "character" doit toujours être ` +
    `exactement une des clés suivantes : ${cast.map((c) => `"${c.key}"`).join(', ')}.`;

  const aiRes = await env.AI.run('@cf/meta/llama-3.3-70b-instruct-fp8-fast', {
    messages: [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: story },
    ],
  });
  let raw = aiRes?.response;
  if (typeof raw !== 'string') {
    raw = raw != null ? JSON.stringify(raw) : '';
  }
  const match = raw.match(/\{[\s\S]*\}/);
  if (!match) {
    throw new Error('Scénario IA : réponse du modèle sans JSON exploitable — ' + raw.slice(0, 300));
  }
  let parsed;
  try {
    parsed = JSON.parse(match[0]);
  } catch (err) {
    throw new Error('Scénario IA : JSON invalide — ' + String(err));
  }
  const validKeys = new Set(cast.map((c) => c.key));
  const turns = (Array.isArray(parsed.turns) ? parsed.turns : [])
    .filter((t) => t && validKeys.has(t.character) && (t.text || '').trim())
    .map((t) => ({ character: t.character, text: t.text.trim() }));
  if (turns.length === 0) {
    throw new Error('Scénario IA : aucune réplique exploitable extraite de l\'histoire.');
  }
  return { context: (parsed.context || '').trim(), turns };
}

// --- Helpers non-bloquants pour la machine à états du mode duo (voir /status, branche "duo2:") ---

// Soumet un job RunPod (n'importe quel mode : TTS seul, duo_composite, stitch...) et retourne
// juste son id, SANS attendre qu'il se termine (contrairement à runpodTTS, bloquant).
async function runpodSubmit(env, input) {
  const res = await fetch(`https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}/run`, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${env.RUNPOD_API_KEY}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ input }),
  });
  const data = await res.json();
  const jobId = data?.id;
  if (!res.ok || !jobId) {
    throw new Error('RunPod: impossible de lancer le job — ' + JSON.stringify(data));
  }
  return jobId;
}

// Vérifie l'état d'un job RunPod déjà soumis (un seul appel, ne boucle pas).
async function runpodPoll(env, jobId) {
  const res = await fetch(`https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}/status/${jobId}`, {
    headers: { Authorization: `Bearer ${env.RUNPOD_API_KEY}` },
  });
  return res.json();
}

// 23 juillet, chantier "vitesse multi-scènes" : AVANT, la machine à états "duo2:" ne faisait
// avancer QU'UNE scène à la fois (state.scenes[state.sceneIndex]) -- la scène 2 ne démarrait
// (audio, puis surtout la génération vidéo WaveSpeedAI, l'étape la plus longue) qu'une fois la
// scène 1 ENTIÈREMENT terminée, alors que son image composite était déjà prête depuis le début
// (toutes les scènes soumettent leur composite RunPod dès /duo-generate, voir plus bas). Ça
// donnait un temps total ≈ (nombre de scènes) × (durée d'une scène), soit 25-30min pour 2
// petites scènes et bien plus pour 5 -- largement au-delà du budget de poll côté app.
// Cette fonction fait avancer UNE scène d'UN pas, exactement comme avant, mais sur son PROPRE
// `scene.stage` (plus `state.stage`/`state.sceneIndex`) : /status peut donc appeler ceci pour
// TOUTES les scènes en attente EN PARALLÈLE (Promise.all) à chaque poll, ce qui fait tourner
// les N scènes de front. Le temps total tombe à ≈ 1 × (durée d'une scène) + recollage.
async function advanceDuoScene(env, url, uuid, scene, sceneNum) {
  if (scene.stage === 'composite') {
    const poll = await runpodPoll(env, scene.compositeJobId);
    if (poll.status === 'COMPLETED') {
      const out = poll.output || {};
      if (out.status === 'DONE' && out.composites) {
        const composites = {};
        for (const [character, b64] of Object.entries(out.composites)) {
          composites[character] = await uploadBase64ToR2(
            env, url.origin, b64, `duo-composite/${uuid}-s${sceneNum - 1}-${character}.png`, 'image/png'
          );
        }
        scene.composites = composites;
        if (scene.pose) {
          scene.poseKeys = Object.keys(composites);
          scene.poseIndex = 0;
          scene.stage = 'pose_submit';
        } else {
          scene.stage = 'audio';
        }
      } else {
        scene.stage = 'failed';
        scene.error = 'Composite RunPod invalide (scène ' + sceneNum + ') : ' + JSON.stringify(out).slice(0, 300);
      }
    } else if (poll.status === 'FAILED' || poll.status === 'CANCELLED') {
      scene.stage = 'failed';
      scene.error = `Composite RunPod (scène ${sceneNum}): ${poll.status} — ` + (poll.error || JSON.stringify(poll).slice(0, 300));
    }
  } else if (scene.stage === 'pose_submit') {
    const key = scene.poseKeys[scene.poseIndex];
    const posePrompt =
      `Keep this exact scene: identical background, identical cartoon character` +
      `${scene.poseKeys.length > 1 ? 's' : ''} (same colors, same face, same proportions, ` +
      `same flat cartoon art style, same left/right position), do not change their design or ` +
      `the background at all. Only change the pose: make the character(s) ${scene.pose}. Keep ` +
      `full body/bodies visible, keep the overall composition and camera framing the same.`;
    try {
      scene.poseRequestId = await nanoBananaSubmit(env, scene.composites[key], posePrompt);
      scene.stage = 'pose_wait';
    } catch (err) {
      scene.stage = 'failed';
      scene.error = 'Nano Banana 2 (pose) : ' + String(err);
    }
  } else if (scene.stage === 'pose_wait') {
    const key = scene.poseKeys[scene.poseIndex];
    const posePoll = await nanoBananaPoll(env, scene.poseRequestId);
    const poseStatus = posePoll?.data?.status;
    if (poseStatus === 'completed') {
      const outputUrl = posePoll?.data?.outputs?.[0];
      if (outputUrl) {
        scene.composites[key] = outputUrl;
      }
      scene.poseIndex += 1;
      scene.poseRequestId = null;
      scene.stage = scene.poseIndex < scene.poseKeys.length ? 'pose_submit' : 'audio';
    } else if (poseStatus === 'failed') {
      scene.stage = 'failed';
      scene.error = 'Nano Banana 2 (pose) : édition échouée pour ' + key;
    }
  } else if (scene.stage === 'audio') {
    const turn = scene.turns[scene.turnIndex];
    if (!turn.text || !turn.text.trim()) {
      scene.audioSegments.push(makeSilentWavBase64(ACTION_ONLY_SILENCE_SECONDS));
      scene.turnIndex += 1;
      scene.stage = scene.turnIndex < scene.turns.length ? 'audio' : 'concat_submit';
    } else if (!scene.ttsJobId) {
      scene.ttsJobId = await runpodSubmit(env, { text: turn.text, prompt_audio_url: RUNPOD_REFERENCE_VOICE_URL });
    } else {
      const poll = await runpodPoll(env, scene.ttsJobId);
      if (poll.status === 'COMPLETED') {
        const out = poll.output || {};
        if (out.status === 'DONE' && out.audio_base64) {
          scene.audioSegments.push(out.audio_base64);
          scene.ttsJobId = null;
          scene.turnIndex += 1;
          scene.stage = scene.turnIndex < scene.turns.length ? 'audio' : 'concat_submit';
        } else {
          scene.stage = 'failed';
          scene.error = 'TTS RunPod invalide pour la réplique ' + (scene.turnIndex + 1) + ' (scène ' + sceneNum + ')';
        }
      } else if (poll.status === 'FAILED' || poll.status === 'CANCELLED') {
        scene.stage = 'failed';
        scene.error = `TTS RunPod: ${poll.status} (scène ${sceneNum}, réplique ${scene.turnIndex + 1})`;
      }
    }
  } else if (scene.stage === 'concat_submit') {
    scene.concatJobId = await runpodSubmit(env, { mode: 'concat_turns_audio', audio_base64_list: scene.audioSegments });
    scene.stage = 'concat_wait';
  } else if (scene.stage === 'concat_wait') {
    const poll = await runpodPoll(env, scene.concatJobId);
    if (poll.status === 'COMPLETED') {
      const out = poll.output || {};
      if (out.status === 'DONE' && out.audio_base64 && out.durations) {
        scene.audioBase64 = out.audio_base64;
        scene.turnDurations = out.durations;
        scene.audioSegments = [];
        scene.stage = 'wavespeed_submit';
      } else {
        scene.stage = 'failed';
        scene.error = 'Concaténation audio RunPod invalide (scène ' + sceneNum + ') : ' + JSON.stringify(out).slice(0, 300);
      }
    } else if (poll.status === 'FAILED' || poll.status === 'CANCELLED') {
      scene.stage = 'failed';
      scene.error = `Concaténation audio RunPod (scène ${sceneNum}): ${poll.status}`;
    }
  } else if (scene.stage === 'wavespeed_submit') {
    const imageUrl = scene.composites['neutral'];
    const timelinePrompt = buildDuoTimelinePrompt(scene.turns, scene.turnDurations);
    scene.wsId = await wavespeedGenerate(env, imageUrl, scene.audioBase64, timelinePrompt);
    scene.audioBase64 = null;
    scene.stage = 'wavespeed_wait';
  } else if (scene.stage === 'wavespeed_wait') {
    const wsRes = await fetch(`https://api.wavespeed.ai/api/v3/predictions/${scene.wsId}/result`, {
      headers: { Authorization: `Bearer ${env.WAVESPEED_API_KEY}` },
    });
    const wsData = await wsRes.json();
    const wsStatus = wsData?.data?.status;
    if (wsStatus === 'completed') {
      const clipUrl = wsData?.data?.outputs?.[0];
      if (clipUrl) {
        scene.wsId = null;
        scene.clipUrl = clipUrl;
        scene.stage = 'clip_done';
      } else {
        scene.stage = 'failed';
        scene.error = 'Pas de vidéo en sortie (WaveSpeedAI, scène ' + sceneNum + ')';
      }
    } else if (wsStatus === 'failed') {
      scene.stage = 'failed';
      scene.error = wsData?.data?.error || ('Erreur WaveSpeedAI (scène ' + sceneNum + ')');
    }
  }
}

// 21 juillet -- fix bug audio "plan continu" : une réplique dont TOUT le texte est une action
// entre crochets (ex. "[fait un clin d'œil]", rien d'autre) donne un cleanText VIDE une fois
// extractAction() passé (le garde-fou `t.text || fallback` ne protège que le texte brut avant
// extraction, pas ce cas-là). Envoyer une chaîne vide au TTS RunPod (CosyVoice2-EU, clonage
// zero-shot) le fait halluciner/partir sur un contenu sans rapport (confirmé par Tristana :
// audio disant "enregistré par ofilum", Zuzu qui rigole avant de parler, durée totale anormalement
// courte). Pour une réplique 100% action/geste, il n'y a rien à synthétiser -- on génère un court
// silence localement (sans passer par RunPod) juste assez long pour laisser l'action se jouer dans
// ce segment de la timeline, au lieu d'appeler le TTS avec une entrée vide.
const ACTION_ONLY_SILENCE_SECONDS = 1.8;
function makeSilentWavBase64(seconds, sampleRate = 24000) {
  const numSamples = Math.max(1, Math.round(seconds * sampleRate));
  const dataSize = numSamples * 2; // PCM 16 bits mono
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);
  const writeStr = (offset, str) => {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  };
  writeStr(0, 'RIFF');
  view.setUint32(4, 36 + dataSize, true);
  writeStr(8, 'WAVE');
  writeStr(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeStr(36, 'data');
  view.setUint32(40, dataSize, true);
  // Les octets PCM restent à 0 par défaut (ArrayBuffer initialisé à zéro) = silence.
  const bytes = new Uint8Array(buffer);
  let binary = '';
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode.apply(null, bytes.subarray(i, i + chunk));
  }
  return btoa(binary);
}

// Décode un base64 et l'héberge sur le bucket R2 privé, renvoie l'URL publique (servie par /video/:key,
// qui fonctionne pour n'importe quel type de fichier, pas seulement les vidéos, malgré son nom).
async function uploadBase64ToR2(env, origin, base64, key, contentType) {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  await env.VIDEOS_BUCKET.put(key, bytes, { httpMetadata: { contentType } });
  return `${origin}/video/${key}`;
}

// L'état d'une génération duo (machine à états multi-étapes : fond → images composites →
// par réplique : audio puis vidéo WaveSpeedAI → recollage final) est stocké en JSON sur le
// même bucket R2, PAS dans le video_id renvoyé à l'app (l'app ne fait que renvoyer ce même
// video_id tel quel à chaque appel /status, elle ne peut pas transporter un état qui change).
async function readDuoState(env, uuid) {
  const obj = await env.VIDEOS_BUCKET.get(`duo-state/${uuid}.json`);
  if (!obj) return null;
  return JSON.parse(await obj.text());
}
async function writeDuoState(env, uuid, state) {
  await env.VIDEOS_BUCKET.put(`duo-state/${uuid}.json`, JSON.stringify(state), {
    httpMetadata: { contentType: 'application/json' },
  });
}

export default {
  async fetch(request, env) {
    // Autoriser les appels depuis l'app (CORS)
    const corsHeaders = {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'POST, GET, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Range',
      'Access-Control-Expose-Headers': 'Content-Range, Accept-Ranges, Content-Length',
    };

    if (request.method === 'OPTIONS') {
      return new Response(null, { headers: corsHeaders });
    }

    const url = new URL(request.url);

    // Servir un fichier (vidéo OU image, malgré le nom de la route historique) hébergé sur le
    // bucket R2 privé "easyvideo-generated" — c'est la SEULE façon d'accéder à ces fichiers
    // (le bucket lui-même n'est pas public) : GET /video/<clé>
    // IMPORTANT : les lecteurs vidéo natifs (iOS/Android) exigent le support des requêtes
    // "Range" (streaming par morceaux) pour lire une vidéo — sans ça, écran noir silencieux.
    // CloudFront (pipeline WaveSpeedAI) le fait automatiquement ; ici on doit le gérer nous-mêmes.
    if (url.pathname.startsWith('/video/')) {
      const key = decodeURIComponent(url.pathname.slice('/video/'.length));
      const rangeHeader = request.headers.get('range');

      const head = await env.VIDEOS_BUCKET.head(key);
      if (!head) {
        return new Response('Fichier introuvable', { status: 404, headers: corsHeaders });
      }
      const size = head.size;
      const contentType = head.httpMetadata?.contentType || 'video/mp4';

      if (rangeHeader) {
        const match = /bytes=(\d+)-(\d*)/.exec(rangeHeader);
        if (match) {
          const start = parseInt(match[1], 10);
          const end = match[2] ? Math.min(parseInt(match[2], 10), size - 1) : size - 1;
          const object = await env.VIDEOS_BUCKET.get(key, { range: { offset: start, length: end - start + 1 } });
          if (!object) {
            return new Response('Fichier introuvable', { status: 404, headers: corsHeaders });
          }
          const headers = new Headers(corsHeaders);
          headers.set('Content-Type', contentType);
          headers.set('Content-Range', `bytes ${start}-${end}/${size}`);
          headers.set('Accept-Ranges', 'bytes');
          headers.set('Content-Length', String(end - start + 1));
          headers.set('Cache-Control', 'public, max-age=31536000, immutable');
          return new Response(object.body, { status: 206, headers });
        }
      }

      const object = await env.VIDEOS_BUCKET.get(key);
      if (!object) {
        return new Response('Fichier introuvable', { status: 404, headers: corsHeaders });
      }
      const headers = new Headers(corsHeaders);
      headers.set('Content-Type', contentType);
      headers.set('Accept-Ranges', 'bytes');
      headers.set('Content-Length', String(size));
      headers.set('Cache-Control', 'public, max-age=31536000, immutable');
      return new Response(object.body, { headers });
    }

    // Test de vie du serveur : /health
    if (url.pathname === '/health') {
      return new Response(JSON.stringify({ status: 'ok', service: 'easyvideo-backend' }), {
        headers: { 'Content-Type': 'application/json', ...corsHeaders },
      });
    }

    // DEBUG TEMPORAIRE (16 juillet) — liste les voix HeyGen françaises disponibles sur le compte,
    // pour choisir de vraies voix françaises distinctes par style au lieu des 2 voix fixes actuelles.
    // À supprimer une fois le choix des voix fait.
    if (url.pathname === '/debug-voices') {
      try {
        const res = await fetch('https://api.heygen.com/v2/voices', {
          headers: { 'X-Api-Key': env.HEYGEN_API_KEY },
        });
        const data = await res.json();
        const all = data?.data?.voices || [];
        const fr = all.filter((v) => (v.language || '').toLowerCase().includes('french'));
        return new Response(JSON.stringify({ total: all.length, frenchCount: fr.length, french: fr }), {
          headers: { 'Content-Type': 'application/json', ...corsHeaders },
        });
      } catch (err) {
        return new Response(JSON.stringify({ error: String(err) }), {
          status: 500,
          headers: { 'Content-Type': 'application/json', ...corsHeaders },
        });
      }
    }

    // Génère une image de personnage à partir d'une description texte (bouton "Générer l'avatar
    // (IA)" dans l'onglet Créer > nouveau personnage, 17 juillet). Réutilise le même modèle
    // Workers AI que les fonds IA (flux-1-schnell, voir /duo-generate et /story-generate), avec
    // un prompt orienté "character design". Style choisi dans l'app (cartoon/réaliste/manga,
    // ajouté le 17 juillet suite au retour Tristana "on peut mettre dessin animé, réaliste, manga").
    if (url.pathname === '/generate-character-image' && request.method === 'POST') {
      try {
        const body = await request.json();
        const description = (body.description || '').trim();
        if (!description) {
          return new Response(JSON.stringify({ error: { message: 'Description manquante' } }), {
            status: 400,
            headers: { 'Content-Type': 'application/json', ...corsHeaders },
          });
        }
        const STYLE_PROMPTS = {
          cartoon: "Full body flat 2D cartoon character design, children's animated show art style, bright saturated colors, simple clean shapes, thick black outlines, plain light solid color background, centered, front-facing, no text, no watermark.",
          realistic: 'Full body photorealistic character portrait, natural lighting, realistic skin texture and proportions, high detail, plain light solid color background, centered, front-facing, no text, no watermark.',
          manga: 'Full body manga/anime character design, Japanese manga art style, clean linework, cel-shaded coloring, expressive features, plain light solid color background, centered, front-facing, no text, no watermark.',
        };
        const stylePrompt = STYLE_PROMPTS[body.style] || STYLE_PROMPTS.cartoon;
        const aiRes = await env.AI.run('@cf/black-forest-labs/flux-1-schnell', {
          prompt: `${description}. ${stylePrompt}`,
        });
        const imageBase64 = aiRes?.image;
        if (!imageBase64) {
          return new Response(JSON.stringify({ error: { message: "Impossible de générer l'image du personnage" } }), {
            status: 500,
            headers: { 'Content-Type': 'application/json', ...corsHeaders },
          });
        }
        return new Response(JSON.stringify({ data: { image_base64: imageBase64, mime: 'image/jpeg' } }), {
          status: 200,
          headers: { 'Content-Type': 'application/json', ...corsHeaders },
        });
      } catch (err) {
        return new Response(JSON.stringify({ error: { message: "Erreur génération avatar IA" }, details: String(err) }), {
          status: 500,
          headers: { 'Content-Type': 'application/json', ...corsHeaders },
        });
      }
    }

    // Transcrit une note vocale en texte (bouton "Maintenir pour dicter", champ script ET champ
    // décor, 20 juillet -- retour Tristana : la dictée existante était un simple placeholder qui
    // n'écrivait jamais le vrai texte prononcé, et échouait carrément sur les enregistrements
    // courts). Utilise Whisper via Cloudflare Workers AI (même binding "AI" que flux-1-schnell,
    // aucun compte/clé supplémentaire à gérer). L'app envoie l'audio enregistré en base64 (peu
    // importe le format exact -- m4a/wav/caf -- Whisper les gère tous), le Worker le décode en
    // octets bruts et le passe à Whisper qui renvoie le texte transcrit.
    // POST /transcribe  { audioBase64 }  ->  { data: { text } }
    if (url.pathname === '/transcribe' && request.method === 'POST') {
      try {
        const body = await request.json();
        const audioBase64 = body.audioBase64;
        if (!audioBase64) {
          return new Response(JSON.stringify({ error: { message: 'Audio manquant' } }), {
            status: 400,
            headers: { 'Content-Type': 'application/json', ...corsHeaders },
          });
        }
        const binary = atob(audioBase64);
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
        if (bytes.length < 2000) {
          // Enregistrement trop court pour contenir de la parole exploitable -- éviter un appel
          // Whisper inutile (et son résultat quasi toujours vide ou halluciné sur du silence pur).
          return new Response(JSON.stringify({ data: { text: '' } }), {
            status: 200,
            headers: { 'Content-Type': 'application/json', ...corsHeaders },
          });
        }
        const aiRes = await env.AI.run('@cf/openai/whisper', { audio: Array.from(bytes) });
        const text = (aiRes?.text || '').trim();
        return new Response(JSON.stringify({ data: { text } }), {
          status: 200,
          headers: { 'Content-Type': 'application/json', ...corsHeaders },
        });
      } catch (err) {
        return new Response(JSON.stringify({ error: { message: 'Erreur transcription audio' }, details: String(err) }), {
          status: 500,
          headers: { 'Content-Type': 'application/json', ...corsHeaders },
        });
      }
    }

    // Transforme une VRAIE photo (personnage réel) en avatar stylisé (manga ou dessin animé),
    // en conservant la ressemblance -- bouton "Styliser une photo" dans l'onglet Créer > nouveau
    // personnage, 18 juillet (retour Tristana : "à partir d'une vraie photo faire un avatar, en
    // anime ou en manga"). Réutilise Nano Banana 2 (déjà intégré pour retoucher la pose des
    // composites duo, voir nanoBananaSubmit/nanoBananaPoll) : une édition d'image qui préserve
    // l'identité de la personne tout en changeant le style de rendu, contrairement à
    // /generate-character-image qui génère une image FICTIVE à partir d'une description texte.
    // Asynchrone (soumission + poll), comme la musique et le duo -- voir /status "nanobanana:".
    // POST /stylize-avatar  { imageBase64, mimeType, style }
    if (url.pathname === '/stylize-avatar' && request.method === 'POST') {
      try {
        const body = await request.json();
        const { imageBase64 } = body;
        const mimeType = body.mimeType || 'image/jpeg';
        if (!imageBase64) {
          return new Response(JSON.stringify({ error: { message: 'Photo manquante' } }), {
            status: 400,
            headers: { 'Content-Type': 'application/json', ...corsHeaders },
          });
        }
        const STYLIZE_PROMPTS = {
          cartoon: "Redraw this exact person as a full body flat 2D cartoon character, children's animated show art style, bright saturated colors, simple clean shapes, thick black outlines, keep the same face, hairstyle, skin tone, and clothing colors so they stay recognizable, plain light solid color background, centered, front-facing, no text, no watermark.",
          manga: 'Redraw this exact person as a full body manga/anime character, Japanese manga art style, clean linework, cel-shaded coloring, keep the same face shape, hairstyle, skin tone, and clothing colors so they stay recognizable, plain light solid color background, centered, front-facing, no text, no watermark.',
        };
        const stylizePrompt = STYLIZE_PROMPTS[body.style] || STYLIZE_PROMPTS.cartoon;

        const ext = mimeType === 'image/png' ? 'png' : 'jpg';
        const key = `stylize-source/${crypto.randomUUID()}.${ext}`;
        const photoUrl = await uploadBase64ToR2(env, url.origin, imageBase64, key, mimeType);

        const requestId = await nanoBananaSubmit(env, photoUrl, stylizePrompt);
        return new Response(JSON.stringify({ data: { video_id: `nanobanana:${requestId}` } }), {
          status: 200,
          headers: { 'Content-Type': 'application/json', ...corsHeaders },
        });
      } catch (err) {
        return new Response(JSON.stringify({ error: { message: 'Erreur stylisation avatar IA' }, details: String(err) }), {
          status: 500,
          headers: { 'Content-Type': 'application/json', ...corsHeaders },
        });
      }
    }

    // Génère une musique d'ambiance à partir d'une description texte (bouton "Créer la musique
    // (IA)" dans l'onglet Créer > Musique et ambiance, 17 juillet). Utilise WaveSpeedAI ACE-Step
    // Prompt-to-Audio (même compte que la vidéo, pas de nouvelle clé) : instrumental par défaut
    // (pas de voix), pour servir de fond sonore. Soumission asynchrone, comme /duo-generate —
    // voir /status branche "wavespeed-music:" pour le polling du résultat.
    // POST /generate-music  { prompt, instrumental }
    if (url.pathname === '/generate-music' && request.method === 'POST') {
      try {
        const body = await request.json();
        const prompt = (body.prompt || '').trim();
        if (!prompt) {
          return new Response(JSON.stringify({ error: { message: 'Description manquante' } }), {
            status: 400,
            headers: { 'Content-Type': 'application/json', ...corsHeaders },
          });
        }
        const instrumental = body.instrumental !== false;
        const res = await fetch('https://api.wavespeed.ai/api/v3/wavespeed-ai/ace-step/prompt-to-audio', {
          method: 'POST',
          headers: { Authorization: `Bearer ${env.WAVESPEED_API_KEY}`, 'Content-Type': 'application/json' },
          body: JSON.stringify({ prompt, instrumental, duration: 60, seed: -1 }),
        });
        const data = await res.json();
        const id = data?.data?.id;
        if (!res.ok || !id) {
          return new Response(JSON.stringify({
            error: { message: 'Impossible de lancer la génération musicale' },
            details: JSON.stringify(data),
          }), {
            status: 500,
            headers: { 'Content-Type': 'application/json', ...corsHeaders },
          });
        }
        return new Response(JSON.stringify({ data: { music_id: `wavespeed-music:${id}` } }), {
          status: 200,
          headers: { 'Content-Type': 'application/json', ...corsHeaders },
        });
      } catch (err) {
        return new Response(JSON.stringify({ error: { message: 'Erreur génération musique IA' }, details: String(err) }), {
          status: 500,
          headers: { 'Content-Type': 'application/json', ...corsHeaders },
        });
      }
    }

    // Lister les looks (talking_photo_id) d'un avatar-personnage : GET /avatar-looks?group_id=...
    if (url.pathname === '/avatar-looks') {
      const groupId = url.searchParams.get('group_id');
      if (!groupId) {
        return new Response(JSON.stringify({ error: 'group_id manquant' }), {
          status: 400,
          headers: { 'Content-Type': 'application/json', ...corsHeaders },
        });
      }
      const looksRes = await fetch(`https://api.heygen.com/v2/avatar_group/${groupId}/avatars`, {
        headers: { 'X-Api-Key': env.HEYGEN_API_KEY },
      });
      const looksData = await looksRes.json();
      return new Response(JSON.stringify(looksData), {
        status: looksRes.status,
        headers: { 'Content-Type': 'application/json', ...corsHeaders },
      });
    }

    // Créer un nouvel avatar HeyGen à partir d'une photo (pour les personnages créés par les utilisateurs)
    // POST /create-avatar  { imageBase64, mimeType, name }
    if (url.pathname === '/create-avatar' && request.method === 'POST') {
      try {
        const body = await request.json();
        const { imageBase64, name } = body;
        const mimeType = body.mimeType || 'image/jpeg';
        if (!imageBase64 || !name) {
          return new Response(JSON.stringify({ error: 'imageBase64 ou name manquant' }), {
            status: 400,
            headers: { 'Content-Type': 'application/json', ...corsHeaders },
          });
        }

        // Étape 1 : décoder le base64 en fichier binaire
        const binary = atob(imageBase64);
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
        const ext = mimeType === 'image/png' ? 'png' : 'jpg';

        // Étape 2 : uploader la photo pour obtenir un asset_id
        const form = new FormData();
        form.append('file', new Blob([bytes], { type: mimeType }), `character.${ext}`);
        const uploadRes = await fetch('https://api.heygen.com/v3/assets', {
          method: 'POST',
          headers: { 'x-api-key': env.HEYGEN_API_KEY },
          body: form,
        });
        const uploadData = await uploadRes.json();
        const assetId = uploadData?.data?.asset_id;
        if (!uploadRes.ok || !assetId) {
          return new Response(JSON.stringify({ step: 'upload_asset', ...uploadData }), {
            status: uploadRes.status,
            headers: { 'Content-Type': 'application/json', ...corsHeaders },
          });
        }

        // Étape 3 : créer l'avatar-photo à partir de cette image
        const createRes = await fetch('https://api.heygen.com/v3/avatars', {
          method: 'POST',
          headers: {
            'x-api-key': env.HEYGEN_API_KEY,
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            type: 'photo',
            name,
            file: { type: 'asset_id', asset_id: assetId },
          }),
        });
        const createData = await createRes.json();
        return new Response(JSON.stringify({ step: 'create_avatar', ...createData }), {
          status: createRes.status,
          headers: { 'Content-Type': 'application/json', ...corsHeaders },
        });
      } catch (err) {
        return new Response(JSON.stringify({ error: String(err) }), {
          status: 500,
          headers: { 'Content-Type': 'application/json', ...corsHeaders },
        });
      }
    }

    // Génération vidéo : POST /generate  { text: "..." } ou { scenes: [{ text, avatarId, voiceId, characterType }, ...] }
    if (url.pathname === '/generate' && request.method === 'POST') {
      try {
        const body = await request.json();
        const hasCustomBackground = Boolean(body.backgroundColor || body.backgroundImageUrl);

        // Plusieurs personnages qui s'enchaînent dans une même vidéo (scènes successives)
        const scenes = Array.isArray(body.scenes) && body.scenes.length > 0
          ? body.scenes
          : [{ text: body.text, avatarId: body.avatarId, voiceId: body.voiceId, characterType: body.characterType }];

        // Routing Zuzu/Titu : si une seule scène et que son avatarId correspond à une mascotte connue,
        // on passe par CosyVoice2-EU (TTS français sur RunPod) + WaveSpeedAI (Wan2.2-S2V, lip-sync)
        // au lieu de HeyGen. Remplace l'ancien pipeline RunPod/viseme (voir NOTES_backend_technique.md).
        if (scenes.length === 1 && RUNPOD_CHARACTERS[scenes[0].avatarId]) {
          const character = RUNPOD_CHARACTERS[scenes[0].avatarId];
          const text = scenes[0].text || `Bonjour, je suis ${character} !`;
          // "Scénario IA" (mode texte + décor/contexte) : si un contexte a été fourni (pastille
          // rapide ou lieu décrit librement, ex. "un coffee shop cosy"), on le transmet à
          // WaveSpeedAI comme prompt de scène en plus de l'audio, pour que le personnage évolue
          // dans ce décor au lieu du fond neutre par défaut du mode "parle à la caméra".
          const context = (body.context || '').trim();
          const scenePrompt = context
            ? `Scene setting: ${context}. ${character} moves and acts naturally within this setting while speaking.`
            : undefined;
          // Musique IA optionnelle (voir /duo-generate pour la version 2 personnages, même
          // principe étendu ici au solo, 18 juillet) — appliquée après la vidéo WaveSpeedAI.
          const musicUrl = body.musicUrl || null;
          const musicPlacement = body.musicPlacement === 'intro' ? 'intro' : 'background';
          const musicVolume = typeof body.musicVolume === 'number' ? body.musicVolume : 0.5;

          try {
            const audioBase64 = await runpodTTS(env, text);
            const imageUrl = WAVESPEED_CHARACTER_IMAGES[character];
            const wsId = await wavespeedGenerate(env, imageUrl, audioBase64, scenePrompt);

            if (musicUrl) {
              const uuid = crypto.randomUUID();
              await writeDuoState(env, uuid, {
                stage: 'wait_wavespeed',
                wsId,
                videoUrl: null,
                musicUrl,
                musicPlacement,
                musicVolume,
                addMusicJobId: null,
                finalVideoUrl: null,
              });
              return new Response(JSON.stringify({ data: { video_id: `wavespeed2:${uuid}` } }), {
                status: 200,
                headers: { 'Content-Type': 'application/json', ...corsHeaders },
              });
            }

            return new Response(JSON.stringify({ data: { video_id: `wavespeed:${wsId}` } }), {
              status: 200,
              headers: { 'Content-Type': 'application/json', ...corsHeaders },
            });
          } catch (err) {
            return new Response(JSON.stringify({
              error: { message: 'Impossible de lancer la génération (CosyVoice2-EU / WaveSpeedAI)' },
              details: String(err),
            }), {
              status: 500,
              headers: { 'Content-Type': 'application/json', ...corsHeaders },
            });
          }
        }

        const buildVideoInput = (scene) => {
          const text = scene.text || 'Bonjour, je suis Zuzu !';
          const avatarId = scene.avatarId || 'Abigail_expressive_2024112501';
          const voiceId = scene.voiceId || '722e1d3f97434f4ba5a7ee3e1a8538d2';
          const characterType = scene.characterType || 'avatar';

          const character =
            characterType === 'talking_photo'
              ? { type: 'talking_photo', talking_photo_id: avatarId, use_avatar_iv_model: true, matting: hasCustomBackground }
              : { type: 'avatar', avatar_id: avatarId, avatar_style: 'normal' };

          const voiceObj = { type: 'text', input_text: text, voice_id: voiceId };
          // Réglages pitch/speed envoyés par l'app (panel de voix, 17 juillet) -- HeyGen les
          // applique par-dessus la voix choisie (pitch: -50 grave à +50 aigu, speed: 0.5 lent à
          // 1.5 rapide), ce qui permet de nuancer une même voix sans devoir changer de voice_id.
          if (typeof scene.voicePitch === 'number' && !Number.isNaN(scene.voicePitch)) {
            voiceObj.pitch = Math.max(-50, Math.min(50, scene.voicePitch));
          }
          if (typeof scene.voiceSpeed === 'number' && !Number.isNaN(scene.voiceSpeed)) {
            voiceObj.speed = Math.max(0.5, Math.min(1.5, scene.voiceSpeed));
          }
          const videoInput = {
            character,
            voice: voiceObj,
          };
          if (characterType === 'talking_photo') {
            videoInput.use_avatar_iv_model = true;
          }
          if (body.backgroundColor) {
            videoInput.background = { type: 'color', value: body.backgroundColor };
          } else if (body.backgroundImageUrl) {
            videoInput.background = { type: 'image', url: body.backgroundImageUrl };
          }
          return videoInput;
        };

        const anyTalkingPhoto = scenes.some((s) => (s.characterType || 'avatar') === 'talking_photo');

        const requestBody = {
          video_inputs: scenes.map(buildVideoInput),
          dimension: { width: 720, height: 1280 },
        };
        if (anyTalkingPhoto) {
          requestBody.use_avatar_iv_model = true;
        }

        // Appel à HeyGen pour lancer la génération vidéo
        const heygenRes = await fetch('https://api.heygen.com/v2/video/generate', {
          method: 'POST',
          headers: {
            'X-Api-Key': env.HEYGEN_API_KEY,
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(requestBody),
        });

        const heygenData = await heygenRes.json();

        return new Response(JSON.stringify(heygenData), {
          status: heygenRes.status,
          headers: { 'Content-Type': 'application/json', ...corsHeaders },
        });
      } catch (err) {
        return new Response(JSON.stringify({ error: String(err) }), {
          status: 500,
          headers: { 'Content-Type': 'application/json', ...corsHeaders },
        });
      }
    }

    // Génération "Duo" : Zuzu et Titu ensemble dans le même cadre, avec un fond généré par IA.
    // (Migration "Round 4" — 13 juillet) : au lieu de l'ancien rig cutout+rotation RunPod (bugué :
    // membres qui traînent un bout du corps, taches de couleur au raccord), chaque réplique est
    // maintenant animée par WaveSpeedAI (vraie IA vidéo, même moteur que le mode solo), sur une
    // image composite (fond + les 2 personnages positionnés) construite une fois par RunPod.
    // POST /duo-generate  { scenes: [{ turns: [{avatarId, text}, ...], backgroundPrompt, backgroundColor }, ...] }
    // 21 juillet -- chantier "scènes multiples" (retour Tristana : jusqu'à 5 scènes, chacune avec
    // son propre décor et son propre script, remplies manuellement -- pas de découpage IA d'une
    // histoire libre, ça c'est le rôle de /story-generate). Remplace l'ancienne forme à plat
    // { turns, backgroundPrompt } (= toujours une seule scène) par un tableau "scenes". Chaque
    // scène tourne dans son propre plan continu (voir buildDuoTimelinePrompt), sans coupe interne ;
    // les scènes entre elles sont recollées en fondu (mode "stitch" RunPod, déjà utilisé pour
    // adoucir les jump cuts) plutôt qu'avec une coupe sèche -- voir la nouvelle étape
    // "scenes_stitch_submit/wait" dans /status, branche "duo2:".
    // Cette route ne fait que LANCER le pipeline (fonds + soumission des jobs "composite" de
    // chaque scène) et rend la main immédiatement — tout le reste avance pas à pas à chaque appel
    // de GET /status, pour ne jamais bloquer une requête Worker sur plusieurs minutes de génération.
    if (url.pathname === '/duo-generate' && request.method === 'POST') {
      try {
        const body = await request.json();
        const scenesIn = Array.isArray(body.scenes) ? body.scenes : [];
        // Musique IA optionnelle à attacher à la vidéo FINALE (après recollage de toutes les
        // scènes) — voir le choix devant/en fond dans l'app, 18 juillet.
        const musicUrl = body.musicUrl || null;
        const musicPlacement = body.musicPlacement === 'intro' ? 'intro' : 'background';
        const musicVolume = typeof body.musicVolume === 'number' ? body.musicVolume : 0.5;

        if (scenesIn.length < 1) {
          return new Response(JSON.stringify({ error: { message: 'Il faut au moins une scène.' } }), {
            status: 400,
            headers: { 'Content-Type': 'application/json', ...corsHeaders },
          });
        }
        if (scenesIn.length > 5) {
          return new Response(JSON.stringify({ error: { message: '5 scènes maximum.' } }), {
            status: 400,
            headers: { 'Content-Type': 'application/json', ...corsHeaders },
          });
        }

        const scenes = [];
        for (const sceneIn of scenesIn) {
          const turnsIn = Array.isArray(sceneIn.turns) ? sceneIn.turns : [];
          if (turnsIn.length < 1) {
            return new Response(JSON.stringify({ error: { message: 'Chaque scène doit avoir au moins une réplique.' } }), {
              status: 400,
              headers: { 'Content-Type': 'application/json', ...corsHeaders },
            });
          }
          const turns = [];
          for (const t of turnsIn) {
            const character = RUNPOD_CHARACTERS[t.avatarId];
            if (!character) {
              return new Response(JSON.stringify({
                error: { message: `Personnage non reconnu pour la scène à deux (avatarId: ${t.avatarId})` },
              }), {
                status: 400,
                headers: { 'Content-Type': 'application/json', ...corsHeaders },
              });
            }
            const { text: cleanText, action } = extractAction(t.text || `Bonjour, je suis ${character} !`);
            turns.push({ character, text: cleanText, action });
          }

          const backgroundPrompt = sceneIn.backgroundPrompt || 'un décor coloré et chaleureux';
          const backgroundColor = sceneIn.backgroundColor || null;

          // Fond : couleur unie transmise directement à RunPod (PIL, pas besoin d'IA), sinon
          // génération IA (Workers AI flux-1-schnell) à partir du prompt texte de CETTE scène.
          let backgroundBase64 = null;
          if (!backgroundColor) {
            const aiRes = await env.AI.run('@cf/black-forest-labs/flux-1-schnell', {
              prompt: buildSceneBackgroundPrompt(backgroundPrompt),
            });
            backgroundBase64 = aiRes?.image;
            if (!backgroundBase64) {
              return new Response(JSON.stringify({ error: { message: 'Impossible de générer le fond IA (scène ' + (scenes.length + 1) + ')' } }), {
                status: 500,
                headers: { 'Content-Type': 'application/json', ...corsHeaders },
              });
            }
          }

          let compositeJobId;
          try {
            compositeJobId = await runpodSubmit(env, backgroundColor
              ? { mode: 'duo_composite', background_color: backgroundColor }
              : { mode: 'duo_composite', background_base64: backgroundBase64 });
          } catch (err) {
            return new Response(JSON.stringify({
              error: { message: 'Impossible de lancer la génération sur RunPod (scène ' + (scenes.length + 1) + ')' },
              details: String(err),
            }), {
              status: 500,
              headers: { 'Content-Type': 'application/json', ...corsHeaders },
            });
          }

          scenes.push({
            stage: 'composite',
            error: null,
            turns,
            pose: detectDuoPose(backgroundPrompt),
            compositeJobId,
            composites: null,
            poseKeys: null,
            poseIndex: 0,
            poseRequestId: null,
            turnIndex: 0,
            ttsJobId: null,
            audioBase64: null,
            audioSegments: [],
            turnDurations: null,
            concatJobId: null,
            wsId: null,
            clipUrl: null,
          });
        }

        const uuid = crypto.randomUUID();
        const state = {
          scenes,
          sceneClips: [],
          // 23 juillet : "scenes" = toutes les scènes tournent en parallèle (voir
          // advanceDuoScene) au lieu d'avancer une par une -- gros gain de vitesse pour les
          // scénarios à plusieurs scènes.
          stage: 'scenes',
          stitchJobId: null,
          stitchedVideoUrl: null,
          musicUrl,
          musicPlacement,
          musicVolume,
          addMusicJobId: null,
          finalVideoUrl: null,
          error: null,
        };
        await writeDuoState(env, uuid, state);

        return new Response(JSON.stringify({ data: { video_id: `duo2:${uuid}` } }), {
          status: 200,
          headers: { 'Content-Type': 'application/json', ...corsHeaders },
        });
      } catch (err) {
        return new Response(JSON.stringify({ error: String(err) }), {
          status: 500,
          headers: { 'Content-Type': 'application/json', ...corsHeaders },
        });
      }
    }

    // Génération "Scénario IA" : au lieu de faire écrire à l'utilisatrice une réplique par ligne
    // séparément du décor, elle raconte toute l'histoire en langage naturel (dictée ou tapée,
    // narration + dialogue mélangés, ex. "Zuzu et Titu sont devant un coffee shop... Zuzu dit...")
    // et un modèle de langage (Cloudflare Workers AI, voir parseScenarioWithAI) en extrait le
    // décor et la liste ordonnée des répliques de chaque personnage. Une fois extrait, ça rejoint
    // exactement le pipeline solo (/generate) ou duo (/duo-generate) existant selon le casting.
    // POST /story-generate  { avatarIds: [id1] ou [id1, id2], story }
    if (url.pathname === '/story-generate' && request.method === 'POST') {
      try {
        const body = await request.json();
        const avatarIds = Array.isArray(body.avatarIds) ? body.avatarIds : [];
        const story = (body.story || '').trim();

        if (avatarIds.length < 1) {
          return new Response(JSON.stringify({ error: { message: 'Choisis au moins un personnage pour le casting.' } }), {
            status: 400,
            headers: { 'Content-Type': 'application/json', ...corsHeaders },
          });
        }
        if (!story) {
          return new Response(JSON.stringify({ error: { message: "L'histoire est vide." } }), {
            status: 400,
            headers: { 'Content-Type': 'application/json', ...corsHeaders },
          });
        }

        const cast = [];
        for (const id of avatarIds) {
          const key = RUNPOD_CHARACTERS[id];
          if (!key) {
            return new Response(JSON.stringify({
              error: { message: `Personnage non reconnu pour un scénario (avatarId: ${id})` },
            }), {
              status: 400,
              headers: { 'Content-Type': 'application/json', ...corsHeaders },
            });
          }
          cast.push({ avatarId: id, key, name: key.charAt(0).toUpperCase() + key.slice(1) });
        }

        let scenario;
        try {
          scenario = await parseScenarioWithAI(env, story, cast);
        } catch (err) {
          return new Response(JSON.stringify({
            error: { message: "Impossible de comprendre l'histoire (analyse IA)." },
            details: String(err),
          }), {
            status: 500,
            headers: { 'Content-Type': 'application/json', ...corsHeaders },
          });
        }

        const distinctKeys = [...new Set(scenario.turns.map((t) => t.character))];

        // Casting à 1 personnage effectivement présent dans les répliques extraites : on
        // concatène ses répliques en un seul texte parlé (comme le mode solo normal), avec le
        // contexte du décor transmis à WaveSpeedAI comme prompt de scène.
        if (distinctKeys.length === 1) {
          const c = cast.find((c) => c.key === distinctKeys[0]);
          const fullText = scenario.turns.map((t) => t.text).join(' ');
          const scenePrompt = scenario.context
            ? `Scene setting: ${scenario.context}. ${c.name} moves and acts naturally within this setting while speaking.`
            : undefined;
          try {
            const audioBase64 = await runpodTTS(env, fullText);
            const imageUrl = WAVESPEED_CHARACTER_IMAGES[c.key];
            const wsId = await wavespeedGenerate(env, imageUrl, audioBase64, scenePrompt);
            return new Response(JSON.stringify({ data: { video_id: `wavespeed:${wsId}` } }), {
              status: 200,
              headers: { 'Content-Type': 'application/json', ...corsHeaders },
            });
          } catch (err) {
            return new Response(JSON.stringify({
              error: { message: 'Impossible de lancer la génération du scénario (CosyVoice2-EU / WaveSpeedAI)' },
              details: String(err),
            }), {
              status: 500,
              headers: { 'Content-Type': 'application/json', ...corsHeaders },
            });
          }
        }

        // Casting à 2 personnages : on rejoint exactement le pipeline duo existant (même machine
        // à états composite → audio/vidéo par réplique → recollage, voir /status "duo2:").
        const turns = scenario.turns.map((t) => {
          const { text: cleanText, action } = extractAction(t.text);
          return { character: t.character, text: cleanText, action };
        });

        const sceneDescription = scenario.context || 'un décor coloré et chaleureux';
        const aiRes = await env.AI.run('@cf/black-forest-labs/flux-1-schnell', {
          prompt: buildSceneBackgroundPrompt(sceneDescription),
        });
        const backgroundBase64 = aiRes?.image;
        if (!backgroundBase64) {
          return new Response(JSON.stringify({ error: { message: 'Impossible de générer le fond IA' } }), {
            status: 500,
            headers: { 'Content-Type': 'application/json', ...corsHeaders },
          });
        }

        let compositeJobId;
        try {
          compositeJobId = await runpodSubmit(env, { mode: 'duo_composite', background_base64: backgroundBase64 });
        } catch (err) {
          return new Response(JSON.stringify({
            error: { message: 'Impossible de lancer la génération sur RunPod' },
            details: String(err),
          }), {
            status: 500,
            headers: { 'Content-Type': 'application/json', ...corsHeaders },
          });
        }

        // 21 juillet -- /status "duo2:" attend désormais toujours un tableau "scenes" (voir
        // chantier "scènes multiples" sur /duo-generate) : le scénario IA reste pour l'instant
        // une seule scène (l'IA ne découpe pas encore l'histoire libre en plusieurs scènes avec
        // décors différents -- ça reste le rôle du mode manuel), donc on enveloppe simplement
        // tout ça dans un tableau à un seul élément pour rester compatible avec la machine à
        // états commune.
        const uuid = crypto.randomUUID();
        const state = {
          scenes: [{
            stage: 'composite',
            error: null,
            turns,
            pose: detectDuoPose(story + ' ' + sceneDescription),
            compositeJobId,
            composites: null,
            poseKeys: null,
            poseIndex: 0,
            poseRequestId: null,
            turnIndex: 0,
            ttsJobId: null,
            audioBase64: null,
            audioSegments: [],
            turnDurations: null,
            concatJobId: null,
            wsId: null,
            clipUrl: null,
          }],
          sceneClips: [],
          stage: 'scenes',
          stitchJobId: null,
          stitchedVideoUrl: null,
          musicUrl: null,
          musicPlacement: 'background',
          musicVolume: 0.5,
          addMusicJobId: null,
          finalVideoUrl: null,
          error: null,
        };
        await writeDuoState(env, uuid, state);

        return new Response(JSON.stringify({ data: { video_id: `duo2:${uuid}` } }), {
          status: 200,
          headers: { 'Content-Type': 'application/json', ...corsHeaders },
        });
      } catch (err) {
        return new Response(JSON.stringify({ error: String(err) }), {
          status: 500,
          headers: { 'Content-Type': 'application/json', ...corsHeaders },
        });
      }
    }

    // Génération vidéo via le moteur "Avatar IV" (dédié aux avatars-photo comme Zuzu/Titu)
    // POST /generate-iv  { text, imageUrl, voiceId, title }
    if (url.pathname === '/generate-iv' && request.method === 'POST') {
      try {
        const body = await request.json();
        const text = body.text || 'Bonjour !';
        const voiceId = body.voiceId || '722e1d3f97434f4ba5a7ee3e1a8538d2';
        const imageUrl = body.imageUrl;
        let imageKey = body.imageKey; // si déjà connu (uploadé au préalable), on saute l'étape 1/2
        const title = body.title || 'EasyVideo IA';

        if (!imageKey && !imageUrl) {
          return new Response(JSON.stringify({ error: 'imageKey ou imageUrl manquant' }), {
            status: 400,
            headers: { 'Content-Type': 'application/json', ...corsHeaders },
          });
        }

        if (!imageKey) {
          // Étape 1 : télécharger la photo (déjà hébergée chez HeyGen, format WEBP non accepté à l'upload)
          const imgRes = await fetch(imageUrl);
          if (!imgRes.ok) {
            return new Response(JSON.stringify({ step: 'download_image', error: 'Impossible de télécharger la photo' }), {
              status: 400,
              headers: { 'Content-Type': 'application/json', ...corsHeaders },
            });
          }
          const imgBlob = await imgRes.blob();

          // Étape 2 : uploader la photo pour obtenir un image_key (asset_id)
          const form = new FormData();
          form.append('file', imgBlob, 'avatar.webp');
          const uploadRes = await fetch('https://api.heygen.com/v3/assets', {
            method: 'POST',
            headers: { 'x-api-key': env.HEYGEN_API_KEY },
            body: form,
          });
          const uploadData = await uploadRes.json();
          imageKey = uploadData?.data?.asset_id;
          if (!uploadRes.ok || !imageKey) {
            return new Response(JSON.stringify({ step: 'upload_asset', ...uploadData }), {
              status: uploadRes.status,
              headers: { 'Content-Type': 'application/json', ...corsHeaders },
            });
          }
        }

        // Étape 3 : générer la vidéo Avatar IV avec cette image
        const heygenRes = await fetch('https://api.heygen.com/v2/video/av4/generate', {
          method: 'POST',
          headers: {
            'X-Api-Key': env.HEYGEN_API_KEY,
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            video_title: title,
            image_key: imageKey,
            script: text,
            voice_id: voiceId,
          }),
        });

        const heygenData = await heygenRes.json();

        return new Response(JSON.stringify({ step: 'generate', imageKey, ...heygenData }), {
          status: heygenRes.status,
          headers: { 'Content-Type': 'application/json', ...corsHeaders },
        });
      } catch (err) {
        return new Response(JSON.stringify({ error: String(err) }), {
          status: 500,
          headers: { 'Content-Type': 'application/json', ...corsHeaders },
        });
      }
    }

    // Vérifier l'avancement : GET /status?video_id=...
    if (url.pathname === '/status') {
      const videoId = url.searchParams.get('video_id');
      if (!videoId) {
        return new Response(JSON.stringify({ error: 'video_id manquant' }), {
          status: 400,
          headers: { 'Content-Type': 'application/json', ...corsHeaders },
        });
      }

      // Job routé vers WaveSpeedAI (Zuzu/Titu solo) : préfixe "wavespeed:"
      if (videoId.startsWith('wavespeed:')) {
        const wsId = videoId.slice('wavespeed:'.length);
        try {
          const wsRes = await fetch(`https://api.wavespeed.ai/api/v3/predictions/${wsId}/result`, {
            headers: { Authorization: `Bearer ${env.WAVESPEED_API_KEY}` },
          });
          const wsData = await wsRes.json();
          const wsStatus = wsData?.data?.status;

          if (wsStatus === 'completed') {
            const videoUrl = wsData?.data?.outputs?.[0];
            if (videoUrl) {
              return new Response(JSON.stringify({
                data: { status: 'completed', video_url: videoUrl },
              }), {
                status: 200,
                headers: { 'Content-Type': 'application/json', ...corsHeaders },
              });
            }
            return new Response(JSON.stringify({
              data: { status: 'failed', error: { message: 'Pas de vidéo en sortie (WaveSpeedAI)' } },
            }), {
              status: 200,
              headers: { 'Content-Type': 'application/json', ...corsHeaders },
            });
          }

          if (wsStatus === 'failed') {
            return new Response(JSON.stringify({
              data: { status: 'failed', error: { message: wsData?.data?.error || 'Erreur WaveSpeedAI' } },
            }), {
              status: 200,
              headers: { 'Content-Type': 'application/json', ...corsHeaders },
            });
          }

          // created / processing
          return new Response(JSON.stringify({ data: { status: 'processing' } }), {
            status: 200,
            headers: { 'Content-Type': 'application/json', ...corsHeaders },
          });
        } catch (err) {
          return new Response(JSON.stringify({ error: String(err) }), {
            status: 500,
            headers: { 'Content-Type': 'application/json', ...corsHeaders },
          });
        }
      }

      // Job WaveSpeedAI solo (Zuzu/Titu) AVEC musique IA attachée : préfixe "wavespeed2:".
      // Petite machine à états (état stocké sur R2 comme pour duo2:, voir readDuoState) :
      // attendre la vidéo WaveSpeedAI, puis lancer/attendre le mixage musique sur RunPod
      // (mode "add_music", même logique que pour le pipeline duo à 2 personnages).
      if (videoId.startsWith('wavespeed2:')) {
        const uuid = videoId.slice('wavespeed2:'.length);
        try {
          const state = await readDuoState(env, uuid);
          if (!state) {
            return new Response(JSON.stringify({
              data: { status: 'failed', error: { message: 'État de génération introuvable (expiré ou invalide).' } },
            }), {
              status: 200,
              headers: { 'Content-Type': 'application/json', ...corsHeaders },
            });
          }

          if (state.stage === 'wait_wavespeed') {
            const wsRes = await fetch(`https://api.wavespeed.ai/api/v3/predictions/${state.wsId}/result`, {
              headers: { Authorization: `Bearer ${env.WAVESPEED_API_KEY}` },
            });
            const wsData = await wsRes.json();
            const wsStatus = wsData?.data?.status;
            if (wsStatus === 'completed') {
              const videoUrl = wsData?.data?.outputs?.[0];
              if (videoUrl) {
                state.videoUrl = videoUrl;
                state.stage = 'add_music_submit';
              } else {
                state.stage = 'failed';
                state.error = 'Pas de vidéo en sortie (WaveSpeedAI)';
              }
            } else if (wsStatus === 'failed') {
              state.stage = 'failed';
              state.error = wsData?.data?.error || 'Erreur WaveSpeedAI';
            }
          } else if (state.stage === 'add_music_submit') {
            state.addMusicJobId = await runpodSubmit(env, {
              mode: 'add_music',
              video_url: state.videoUrl,
              music_url: state.musicUrl,
              placement: state.musicPlacement,
              volume: state.musicVolume,
            });
            state.stage = 'add_music_wait';
          } else if (state.stage === 'add_music_wait') {
            const poll = await runpodPoll(env, state.addMusicJobId);
            if (poll.status === 'COMPLETED') {
              const out = poll.output || {};
              if (out.status === 'DONE' && out.video_base64) {
                const key = `solo/${uuid}-${Date.now()}-music.mp4`;
                state.finalVideoUrl = await uploadBase64ToR2(env, url.origin, out.video_base64, key, 'video/mp4');
                state.stage = 'done';
              } else {
                // Mixage échoué : on renvoie quand même la vidéo (sans musique) plutôt que de
                // tout faire échouer.
                state.finalVideoUrl = state.videoUrl;
                state.stage = 'done';
              }
            } else if (poll.status === 'FAILED' || poll.status === 'CANCELLED') {
              state.finalVideoUrl = state.videoUrl;
              state.stage = 'done';
            }
          }

          await writeDuoState(env, uuid, state);

          if (state.stage === 'done') {
            return new Response(JSON.stringify({
              data: { status: 'completed', video_url: state.finalVideoUrl },
            }), {
              status: 200,
              headers: { 'Content-Type': 'application/json', ...corsHeaders },
            });
          }
          if (state.stage === 'failed') {
            return new Response(JSON.stringify({
              data: { status: 'failed', error: { message: state.error || 'Erreur inconnue' } },
            }), {
              status: 200,
              headers: { 'Content-Type': 'application/json', ...corsHeaders },
            });
          }
          return new Response(JSON.stringify({ data: { status: 'processing' } }), {
            status: 200,
            headers: { 'Content-Type': 'application/json', ...corsHeaders },
          });
        } catch (err) {
          return new Response(JSON.stringify({ error: String(err) }), {
            status: 500,
            headers: { 'Content-Type': 'application/json', ...corsHeaders },
          });
        }
      }

      // Job de génération musicale (ACE-Step Prompt-to-Audio) : préfixe "wavespeed-music:"
      if (videoId.startsWith('wavespeed-music:')) {
        const wsId = videoId.slice('wavespeed-music:'.length);
        try {
          const wsRes = await fetch(`https://api.wavespeed.ai/api/v3/predictions/${wsId}/result`, {
            headers: { Authorization: `Bearer ${env.WAVESPEED_API_KEY}` },
          });
          const wsData = await wsRes.json();
          const wsStatus = wsData?.data?.status;

          if (wsStatus === 'completed') {
            const audioUrl = wsData?.data?.outputs?.[0];
            if (audioUrl) {
              return new Response(JSON.stringify({
                data: { status: 'completed', audio_url: audioUrl },
              }), {
                status: 200,
                headers: { 'Content-Type': 'application/json', ...corsHeaders },
              });
            }
            return new Response(JSON.stringify({
              data: { status: 'failed', error: { message: 'Pas de musique en sortie (WaveSpeedAI)' } },
            }), {
              status: 200,
              headers: { 'Content-Type': 'application/json', ...corsHeaders },
            });
          }

          if (wsStatus === 'failed') {
            return new Response(JSON.stringify({
              data: { status: 'failed', error: { message: wsData?.data?.error || 'Erreur WaveSpeedAI' } },
            }), {
              status: 200,
              headers: { 'Content-Type': 'application/json', ...corsHeaders },
            });
          }

          return new Response(JSON.stringify({ data: { status: 'processing' } }), {
            status: 200,
            headers: { 'Content-Type': 'application/json', ...corsHeaders },
          });
        } catch (err) {
          return new Response(JSON.stringify({ error: String(err) }), {
            status: 500,
            headers: { 'Content-Type': 'application/json', ...corsHeaders },
          });
        }
      }

      // Job de stylisation d'avatar à partir d'une vraie photo (Nano Banana 2) : préfixe "nanobanana:"
      // Contrairement aux autres jobs Nano Banana 2 (retouche de pose, internes à la machine à
      // états duo), celui-ci est exposé directement à l'app -- on récupère l'image de sortie et on
      // la reconvertit en base64 (comme /generate-character-image) pour rester cohérent avec le
      // reste du flux de création de personnage côté app.
      if (videoId.startsWith('nanobanana:')) {
        const requestId = videoId.slice('nanobanana:'.length);
        try {
          const poll = await nanoBananaPoll(env, requestId);
          const status = poll?.data?.status;
          if (status === 'completed') {
            const outputUrl = poll?.data?.outputs?.[0];
            if (!outputUrl) {
              return new Response(JSON.stringify({
                data: { status: 'failed', error: { message: "Pas d'image en sortie (Nano Banana 2)" } },
              }), {
                status: 200,
                headers: { 'Content-Type': 'application/json', ...corsHeaders },
              });
            }
            const imgRes = await fetch(outputUrl);
            const buf = await imgRes.arrayBuffer();
            let binary = '';
            const bytes = new Uint8Array(buf);
            for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
            const imageBase64 = btoa(binary);
            return new Response(JSON.stringify({
              data: { status: 'completed', image_base64: imageBase64, mime: 'image/png' },
            }), {
              status: 200,
              headers: { 'Content-Type': 'application/json', ...corsHeaders },
            });
          }
          if (status === 'failed') {
            return new Response(JSON.stringify({
              data: { status: 'failed', error: { message: "Échec de la stylisation (Nano Banana 2)" } },
            }), {
              status: 200,
              headers: { 'Content-Type': 'application/json', ...corsHeaders },
            });
          }
          return new Response(JSON.stringify({ data: { status: 'processing' } }), {
            status: 200,
            headers: { 'Content-Type': 'application/json', ...corsHeaders },
          });
        } catch (err) {
          return new Response(JSON.stringify({ error: String(err) }), {
            status: 500,
            headers: { 'Content-Type': 'application/json', ...corsHeaders },
          });
        }
      }

      // Job de génération "duo" (nouveau pipeline WaveSpeedAI, "Round 4") : préfixe "duo2:"
      // Machine à états multi-étapes, stockée sur R2 (voir readDuoState/writeDuoState) : chaque
      // appel ici fait avancer d'UN pas (soumettre ou vérifier le sous-job courant) puis renvoie
      // "processing" tant que ce n'est pas fini, "completed" avec l'URL finale sinon.
      if (videoId.startsWith('duo2:')) {
        const uuid = videoId.slice('duo2:'.length);
        try {
          const state = await readDuoState(env, uuid);
          if (!state) {
            return new Response(JSON.stringify({
              data: { status: 'failed', error: { message: 'État de génération introuvable (expiré ou invalide).' } },
            }), {
              status: 200,
              headers: { 'Content-Type': 'application/json', ...corsHeaders },
            });
          }

          // 23 juillet, chantier "vitesse multi-scènes" : toutes les scènes tournent maintenant
          // EN PARALLÈLE (voir advanceDuoScene ci-dessus) au lieu d'une à la fois -- avant, la
          // scène 2 ne démarrait (TTS, vidéo WaveSpeedAI...) qu'une fois la scène 1 ENTIÈREMENT
          // terminée, d'où les 25-30min pour seulement 2 petites scènes.
          if (state.stage === 'scenes') {
            await Promise.all(state.scenes.map((s, i) =>
              (s.stage === 'clip_done' || s.stage === 'failed') ? Promise.resolve() : advanceDuoScene(env, url, uuid, s, i + 1)
            ));
            const failedScene = state.scenes.find((s) => s.stage === 'failed');
            if (failedScene) {
              state.stage = 'failed';
              state.error = failedScene.error || 'Erreur inconnue (scène)';
            } else if (state.scenes.every((s) => s.stage === 'clip_done')) {
              state.sceneClips = state.scenes.map((s) => s.clipUrl);
              if (state.sceneClips.length > 1) {
                state.stage = 'scenes_stitch_submit';
              } else if (state.musicUrl) {
                state.stitchedVideoUrl = state.sceneClips[0];
                state.stage = 'add_music_submit';
              } else {
                state.finalVideoUrl = state.sceneClips[0];
                state.stage = 'done';
              }
            }
          } else if (state.stage === 'scenes_stitch_submit') {
            // Plusieurs scènes : on les recolle en fondu enchaîné (transition douce plutôt qu'une
            // coupe sèche entre deux scènes) en réutilisant le mode "stitch" déjà présent côté
            // RunPod (worker_runpod.py), initialement prévu pour l'ancien pipeline par réplique.
            state.stitchJobId = await runpodSubmit(env, { mode: 'stitch', clip_urls: state.sceneClips });
            state.stage = 'scenes_stitch_wait';
          } else if (state.stage === 'scenes_stitch_wait') {
            const poll = await runpodPoll(env, state.stitchJobId);
            if (poll.status === 'COMPLETED') {
              const out = poll.output || {};
              if (out.status === 'DONE' && out.video_base64) {
                const key = `duo-scenes/${uuid}-stitched.mp4`;
                const stitchedUrl = await uploadBase64ToR2(env, url.origin, out.video_base64, key, 'video/mp4');
                if (state.musicUrl) {
                  state.stitchedVideoUrl = stitchedUrl;
                  state.stage = 'add_music_submit';
                } else {
                  state.finalVideoUrl = stitchedUrl;
                  state.stage = 'done';
                }
              } else {
                state.stage = 'failed';
                state.error = 'Recollage des scènes RunPod invalide : ' + JSON.stringify(out).slice(0, 300);
              }
            } else if (poll.status === 'FAILED' || poll.status === 'CANCELLED') {
              state.stage = 'failed';
              state.error = `Recollage des scènes RunPod: ${poll.status}`;
            }
          } else if (state.stage === 'add_music_submit') {
            state.addMusicJobId = await runpodSubmit(env, {
              mode: 'add_music',
              video_url: state.stitchedVideoUrl,
              music_url: state.musicUrl,
              placement: state.musicPlacement,
              volume: state.musicVolume,
            });
            state.stage = 'add_music_wait';
          } else if (state.stage === 'add_music_wait') {
            const poll = await runpodPoll(env, state.addMusicJobId);
            if (poll.status === 'COMPLETED') {
              const out = poll.output || {};
              if (out.status === 'DONE' && out.video_base64) {
                const key = `duo/${uuid}-${Date.now()}-music.mp4`;
                state.finalVideoUrl = await uploadBase64ToR2(env, url.origin, out.video_base64, key, 'video/mp4');
                state.stage = 'done';
              } else {
                // Le mixage musique a échoué : on ne perd pas la vidéo déjà prête, on la
                // renvoie telle quelle (sans musique) plutôt que de tout faire échouer.
                state.finalVideoUrl = state.stitchedVideoUrl;
                state.stage = 'done';
              }
            } else if (poll.status === 'FAILED' || poll.status === 'CANCELLED') {
              state.finalVideoUrl = state.stitchedVideoUrl;
              state.stage = 'done';
            }
          }

          await writeDuoState(env, uuid, state);

          if (state.stage === 'done') {
            return new Response(JSON.stringify({ data: { status: 'completed', video_url: state.finalVideoUrl } }), {
              status: 200,
              headers: { 'Content-Type': 'application/json', ...corsHeaders },
            });
          }
          if (state.stage === 'failed') {
            return new Response(JSON.stringify({ data: { status: 'failed', error: { message: state.error || 'Erreur inconnue' } } }), {
              status: 200,
              headers: { 'Content-Type': 'application/json', ...corsHeaders },
            });
          }
          return new Response(JSON.stringify({ data: { status: 'processing' } }), {
            status: 200,
            headers: { 'Content-Type': 'application/json', ...corsHeaders },
          });
        } catch (err) {
          return new Response(JSON.stringify({ error: String(err) }), {
            status: 500,
            headers: { 'Content-Type': 'application/json', ...corsHeaders },
          });
        }
      }

      // Job routé vers RunPod (ANCIEN pipeline "rig cutout" duo, conservé seulement pour compat
      // arrière avec d'éventuels jobs déjà en vol — /duo-generate n'en crée plus depuis la
      // migration "Round 4") : préfixe "runpod:"
      if (videoId.startsWith('runpod:')) {
        const jobId = videoId.slice('runpod:'.length);
        try {
          const statusRes = await fetch(`https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}/status/${jobId}`, {
            headers: { Authorization: `Bearer ${env.RUNPOD_API_KEY}` },
          });
          const statusData = await statusRes.json();
          const jobStatus = statusData?.status;

          if (jobStatus === 'COMPLETED') {
            const output = statusData.output || {};
            if (output.status === 'DONE' && output.video_base64) {
              try {
                const binary = atob(output.video_base64);
                const bytes = new Uint8Array(binary.length);
                for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
                const key = `duo/${jobId}-${Date.now()}.mp4`;
                await env.VIDEOS_BUCKET.put(key, bytes, {
                  httpMetadata: { contentType: 'video/mp4' },
                });
                const videoUrl = `${url.origin}/video/${key}`;
                return new Response(JSON.stringify({
                  data: { status: 'completed', video_url: videoUrl },
                }), {
                  status: 200,
                  headers: { 'Content-Type': 'application/json', ...corsHeaders },
                });
              } catch (err) {
                return new Response(JSON.stringify({
                  data: { status: 'failed', error: { message: 'Erreur hébergement vidéo (R2) : ' + String(err) } },
                }), {
                  status: 200,
                  headers: { 'Content-Type': 'application/json', ...corsHeaders },
                });
              }
            }
            return new Response(JSON.stringify({
              data: { status: 'failed', error: { message: output.error || 'Erreur pendant la génération RunPod' } },
            }), {
              status: 200,
              headers: { 'Content-Type': 'application/json', ...corsHeaders },
            });
          }

          if (jobStatus === 'FAILED' || jobStatus === 'CANCELLED') {
            const detail = statusData.error || statusData.output?.error || JSON.stringify(statusData).slice(0, 500);
            return new Response(JSON.stringify({
              data: { status: 'failed', error: { message: `Job RunPod: ${jobStatus} — ${detail}` } },
            }), {
              status: 200,
              headers: { 'Content-Type': 'application/json', ...corsHeaders },
            });
          }

          // IN_QUEUE ou IN_PROGRESS
          return new Response(JSON.stringify({ data: { status: 'processing' } }), {
            status: 200,
            headers: { 'Content-Type': 'application/json', ...corsHeaders },
          });
        } catch (err) {
          return new Response(JSON.stringify({ error: String(err) }), {
            status: 500,
            headers: { 'Content-Type': 'application/json', ...corsHeaders },
          });
        }
      }

      const statusRes = await fetch(`https://api.heygen.com/v1/video_status.get?video_id=${videoId}`, {
        headers: { 'X-Api-Key': env.HEYGEN_API_KEY },
      });
      const statusData = await statusRes.json();
      return new Response(JSON.stringify(statusData), {
        status: statusRes.status,
        headers: { 'Content-Type': 'application/json', ...corsHeaders },
      });
    }

    // Lister les looks disponibles (pour récupérer les IDs à passer à Cinematic Avatar) : GET /avatars-mine
    if (url.pathname === '/avatars-mine') {
      const looksRes = await fetch('https://api.heygen.com/v3/avatars/looks?ownership=private', {
        headers: { 'x-api-key': env.HEYGEN_API_KEY },
      });
      const looksData = await looksRes.json();
      return new Response(JSON.stringify(looksData), {
        status: looksRes.status,
        headers: { 'Content-Type': 'application/json', ...corsHeaders },
      });
    }

    // Génération "Cinematic Avatar" : plusieurs personnages dans le même cadre + décor décrit par texte
    // POST /cinematic-generate  { prompt, avatarIds: [id1, id2], aspectRatio, duration, title }
    if (url.pathname === '/cinematic-generate' && request.method === 'POST') {
      try {
        const body = await request.json();
        if (!body.prompt || !Array.isArray(body.avatarIds) || body.avatarIds.length === 0) {
          return new Response(JSON.stringify({ error: 'prompt et avatarIds (tableau) sont requis' }), {
            status: 400,
            headers: { 'Content-Type': 'application/json', ...corsHeaders },
          });
        }
        const heygenRes = await fetch('https://api.heygen.com/v3/videos', {
          method: 'POST',
          headers: {
            'x-api-key': env.HEYGEN_API_KEY,
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            type: 'cinematic_avatar',
            prompt: body.prompt,
            avatar_id: body.avatarIds,
            aspect_ratio: body.aspectRatio || '9:16',
            resolution: '720p',
            duration: body.duration || 8,
            title: body.title || 'EasyVideo IA — scène',
          }),
        });
        const heygenData = await heygenRes.json();
        return new Response(JSON.stringify(heygenData), {
          status: heygenRes.status,
          headers: { 'Content-Type': 'application/json', ...corsHeaders },
        });
      } catch (err) {
        return new Response(JSON.stringify({ error: String(err) }), {
          status: 500,
          headers: { 'Content-Type': 'application/json', ...corsHeaders },
        });
      }
    }

    // Vérifier l'avancement d'une vidéo Cinematic Avatar : GET /cinematic-status?video_id=...
    if (url.pathname === '/cinematic-status') {
      const videoId = url.searchParams.get('video_id');
      if (!videoId) {
        return new Response(JSON.stringify({ error: 'video_id manquant' }), {
          status: 400,
          headers: { 'Content-Type': 'application/json', ...corsHeaders },
        });
      }
      const statusRes = await fetch(`https://api.heygen.com/v3/videos/${videoId}`, {
        headers: { 'x-api-key': env.HEYGEN_API_KEY },
      });
      const statusData = await statusRes.json();
      return new Response(JSON.stringify(statusData), {
        status: statusRes.status,
        headers: { 'Content-Type': 'application/json', ...corsHeaders },
      });
    }

    return new Response('EasyVideo IA backend — routes: /health, /generate, /status, /duo-generate, /story-generate, /cinematic-generate, /cinematic-status, /video/:key', {
      headers: corsHeaders,
    });
  },
};
