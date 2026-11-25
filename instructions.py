"""Optional developer instructions sent through Olala's chat template.

The template places a leading ``developer`` message in its own instructions
channel, after OVH's built-in system prompt. Nothing is sent unless a named
profile is selected, so measurements without instructions stay reproducible.
"""

import hashlib

PROFILES = {
    "aucune": "",
    "opinel-fr-v1": """\
Consignes pour cette conversation :
- Réponds en français, sauf si l'utilisateur écrit dans une autre langue.
- Interprète chaque message dans le contexte de la conversation. Une demande \
courte sur la façon de s'adresser à l'utilisateur (tutoiement, vouvoiement, \
ton plus familier ou plus formel) concerne le registre de tes réponses : \
confirme-la brièvement en appliquant déjà ce registre, garde-le ensuite et \
ne change pas de sujet.
- N'affirme que ce que tu sais avec certitude. Si une information manque, \
n'a pas été fournie ou t'est incertaine, dis-le clairement plutôt que de \
deviner. N'invente jamais de chiffres, de noms, de dates, de citations, de \
sources ni de contenu de fichier.
- Si l'utilisateur te corrige, vérifie la correction : si elle est juste, \
donne la réponse corrigée sans répéter l'erreur ; si elle est fausse, \
maintiens poliment la réponse exacte.
- Respecte exactement les contraintes de format et de longueur demandées, \
et garde des réponses proportionnées à la question.""",
}


def resolve(profile):
    if profile not in PROFILES:
        raise ValueError(f"Profil d'instructions inconnu : {profile}")
    return PROFILES[profile]


def describe(profile):
    """Identity recorded in evaluation reports; None means no instructions."""
    text = resolve(profile)
    if not text:
        return None
    return {"profile": profile, "sha256": hashlib.sha256(text.encode()).hexdigest()}


def with_instructions(messages, text):
    """Prepend the developer message at render time, never inside history."""
    if not text:
        return list(messages)
    if messages and messages[0].get("role") in ("system", "developer"):
        raise ValueError("La conversation contient déjà un message système.")
    return [{"role": "developer", "content": text}, *messages]
