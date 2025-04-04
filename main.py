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
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}"
    }
    payload = {
        "inputs": prompt
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api-inference.huggingface.co/models/google/flan-t5-base",
            headers=headers,
            json=payload
        )

        print("📭 Contenu brut HF:", response.text)
        print("📭 Réponse HF status:", response.status_code)

        try:
            return response.json()[0]["generated_text"]
        except Exception as e:
            print("❌ ERREUR Hugging Face:", e)
            return "Intention: autre\nMots-clés: []\nRequête: " + prompt  # Fallback safe



async def scrape_and_summarize(search_query: str) -> tuple[str, str]:
    async with httpx.AsyncClient() as client:
        try:
            scrape_response = await client.post(
                "https://root-web-scraper.fly.dev/scrape",
                json={"query": search_query}
            )
            print("📦 Contenu brut Scraper:", scrape_response.text)
            print("📦 Réponse Scraper status:", scrape_response.status_code)

            if scrape_response.status_code != 200:
                return "", ""

            data = scrape_response.json()
            return data.get("url", ""), data.get("content", "")
        except Exception as e:
            print("🔥 Erreur scraper:", e)
            return "", ""


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
        intention = "autre"
        entities_raw = "[]"
        search_query = data.question

        for line in lines:
            if "intention" in line.lower():
                intention = line.split(":")[1].strip() if ":" in line else "autre"
            elif "mots-clés" in line.lower() or "keywords" in line.lower():
                entities_raw = line.split(":")[1].strip() if ":" in line else "[]"
            elif "requête" in line.lower() or "query" in line.lower():
                search_query = line.split(":")[1].strip() if ":" in line else data.question


        entities = ast.literal_eval(entities_raw) if entities_raw.startswith("[") else []
        if not isinstance(entities, list):
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
        if not content:
            return FinalResponse(
                action="just_groq",
                prompt=f"Je n'ai pas pu récupérer d'informations utiles pour : '{data.question}'. Essaie de reformuler ou pose une autre question.",
                url=url,
                content="",
                entities=entities
            )


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
