"""
Test standalone : RFQ planning avec qwen3.5:4b + thinking activé.

Lance avec :
  python test_rfq_planning.py

Montre la réponse complète (bloc <think> inclus) pour évaluer
ce que le modèle comprend réellement d'un RFQ complexe.
"""

import json
import requests

OLLAMA_URL = "http://localhost:11434"
MODEL = "qwen3.5:4b"

SYSTEM_PROMPT = """Tu es un expert technique en machines robotisées industrielles (WAAM, DED Laser poudre, Cold Spray, FSW, usinage hybride).

Analyse ce message commercial et extrais :
1. La ou les machines VLM concernées (SOLO, GEMINI, COMPAQT, HYMANCO — ou "inconnu")
2. Les technologies demandées (ex: DED Laser poudre, WAAM, Cold Spray, usinage, scan...)
3. Les contraintes techniques explicites (dimensions, effecteurs, fournisseurs imposés/exclus...)
4. Les termes exacts à rechercher dans un catalogue de composants industriels
5. Les questions à clarifier si des informations manquent

Réponds en JSON structuré uniquement."""

RFQ_MESSAGE = """Bonjour Bruno,

Nous sommes sollicités par Thomas Bauer un prospect qui nous connait depuis 2021 et qui est revenu vers nous à Formnext. Il a besoin d'investir cette année dans une cellule robotisée de WLAM car la machine Lasertech qu'il lui permettait de fabriquer ses impellers qualifiés a été vendue en même temps que le site.

Excellence (qui s'appelait avant Man Energy) a donc besoin de retrouver une capacité de fabrication de pièce sur une machine DED Laser poudre. Comme ils ont d'autres pièces en tête qui feront appel à d'autre technologie WAAM et Colspray, ils souhaitent investir dans une cellule robotisée. Thomas Bauer qui nous connait parfaitement sait que VLM est le bon fournisseur. Il nous demande en conséquence de chiffrer. C'est bien une RFQ (pas une RFI) ; son cahier des charges ci-joint est très ouvert à nos suggestions.

Son besoin :
- Une SOLO sans axe linéaire
- Effecteur / Laserline mais open sur autres producteurs majeurs de laser et optique comme Precitec, IPG…, pas sur OSCAR
- Weld pool / Fraunhofer Dresden EMAQs

Son budget :
- 1 M€ en 2026. Upgrading after WAAM, Milling like CS."""


def test_rfq_planning():
    print("=" * 70)
    print(f"Modèle : {MODEL}")
    print(f"Think  : activé")
    print("=" * 70)
    print("\nRFQ envoyé...")
    print("-" * 70)

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": RFQ_MESSAGE},
        ],
        "stream": False,
        "think": True,
        "options": {"temperature": 0},
    }

    try:
        resp = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"ERREUR : {e}")
        return

    message = data.get("message", {})
    thinking = message.get("thinking", "")
    content = message.get("content", "")

    # Stats
    eval_data = data.get("eval_count", "?")
    prompt_eval = data.get("prompt_eval_count", "?")
    total_duration_s = data.get("total_duration", 0) / 1e9

    print(f"\n📊 STATS : {prompt_eval} tokens input | {eval_data} tokens output | {total_duration_s:.1f}s\n")

    if thinking:
        print("🧠 BLOC THINKING :")
        print("-" * 70)
        print(thinking)
        print("-" * 70)
    else:
        print("⚠️  Pas de bloc thinking (think non supporté ou vide)")

    print("\n📋 RÉPONSE JSON :")
    print("-" * 70)
    # Tenter de parser et reformatter le JSON
    try:
        # Enlever les ``` si présents
        clean = content.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[1].rsplit("```", 1)[0]
        parsed = json.loads(clean)
        print(json.dumps(parsed, ensure_ascii=False, indent=2))
    except Exception:
        print(content)
    print("-" * 70)


if __name__ == "__main__":
    test_rfq_planning()
