from fastapi import FastAPI
from pydantic import BaseModel
import httpx
import os
import ast
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

class QuestionRequest(BaseModel):
    question: str

class FinalResponse(BaseModel):
    action: str
    prompt: str
    url: str = None
    content: str = None
    entities: list[str] = []

async def call_huggingface_model(prompt: str) -> str:
    print("🧠 Appel Hugging Face lancé")
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {"inputs": prompt}

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                "https://api-inference.huggingface.co/models/google/flan-t5-base",
                headers=headers,
                json=payload
            )
            print(f"✅ Réponse HF status: {response.status_code}")
            print(f"📦 Contenu brut HF: {response.text[:300]}")  # évite les pavés
            return response.json()[0]["generated_text"]
    except Exception as e:
        print(f"🛑 ERREUR Hugging Face: {e}")
        raise

async def scrape_and_summarize(search_query: str) -> tuple[str, str]:
    print(f"🕸️ Début du scraping pour : {search_query}")
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(
                "https://root-web-scraper.fly.dev/scrape",
                json={"query": search_query}
            )
            print(f"✅ Réponse Scraper status: {response.status_code}")
            print(f"📦 Contenu brut Scraper: {response.text[:300]}")
            data = response.json()
            return data.get("url", ""), data.get("content", "")
    except Exception as e:
        print(f"🛑 ERREUR Scraper: {e}")
        return "", "Erreur lors du scraping."

@app.post("/analyze", response_model=FinalResponse)
async def analyze_question(data: QuestionRequest):
    print("📩 Question reçue :", data.question)

    prompt = f"""Analyse la phrase suivante et donne :
- une intention (recherche, commande, salutation, autre)
- les mots-clés principaux (sous forme de liste)
- une requête de recherche optimisée pour un moteur

Phrase : "{data.question}" """

    try:
        response_text = await call_huggingface_model(prompt)
        print("🧠 Résultat NLP brut :", response_text)
    except Exception as e:
        print("🛑 Échec analyse NLP :", e)
        return FinalResponse(
            action="just_groq",
            prompt=f"Une erreur est survenue dans le module NLP. Réponds normalement à : {data.question}",
            entities=[]
        )

    try:
        lines = response_text.strip().split("\n")
        intention = lines[0].split(":")[1].strip()
        entities_raw = lines[1].split(":")[1].strip()
        search_query = lines[2].split(":")[1].strip()

        print("🧩 Données extraites NLP :", intention, entities_raw, search_query)

        entities = ast.literal_eval(entities_raw) if entities_raw.startswith("[") else []
        if not isinstance(entities, list):
            print("⚠️ Entités non liste !")
            entities = []
    except Exception as e:
        print("⚠️ Erreur d'analyse NLP :", e)
        intention = "recherche"
        entities = []
        search_query = data.question

    print(f"🎯 Intention : {intention}")
    print(f"🔑 Entités : {entities}")
    print(f"🔍 Requête optimisée : {search_query}")

    if intention == "recherche":
        url, content = await scrape_and_summarize(search_query)
        print(f"🔗 URL récupérée : {url}")
        print(f"📃 Résumé contenu (début) : {content[:300]}...")

        prompt_final = f"""Tu es ROOT, une intelligence artificielle experte en réponse contextuelle fiable.

L'utilisateur demande : "{data.question}"

Voici un contenu provenant de la page : {url}

=== DÉBUT DU CONTENU ===
{content}
=== FIN DU CONTENU ===

Ta tâche :
- Réponds à la question en te basant uniquement sur ce contenu
- Si aucune info utile n’est présente, dis-le honnêtement
- Donne une réponse claire, naturelle, et bien formulée

(Source : {url})"""

        return FinalResponse(
            action="scrape+groq",
            prompt=prompt_final.strip(),
            url=url,
            content=content,
            entities=entities
        )
    else:
        prompt_final = f"""L'utilisateur dit : "{data.question}" 

Réponds naturellement, de façon utile et concise."""
        return FinalResponse(
            action="just_groq",
            prompt=prompt_final.strip(),
            entities=entities
        )
