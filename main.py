
from fastapi import FastAPI
from pydantic import BaseModel
import httpx
import os
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
        return response.json()[0]["generated_text"]

async def scrape_and_summarize(search_query: str) -> tuple[str, str]:
    async with httpx.AsyncClient() as client:
        scrape_response = await client.post(
            "https://root-web-scraper.fly.dev/scrape",
            json={"query": search_query}
        )
        data = scrape_response.json()
        return data.get("url", ""), data.get("content", "")

@app.post("/analyze", response_model=FinalResponse)
async def analyze_question(data: QuestionRequest):
    prompt = f"""Analyse la phrase suivante et donne :
- une intention (recherche, commande, salutation, autre)
- les mots-clés principaux (sous forme de liste)
- une requête de recherche optimisée pour un moteur

Phrase : "{data.question}" """

    response_text = await call_huggingface_model(prompt)

    try:
        lines = response_text.strip().split("\n")
        intention = lines[0].split(":")[1].strip()
        entities = eval(lines[1].split(":")[1].strip())
        search_query = lines[2].split(":")[1].strip()
    except:
        intention = "recherche"
        entities = []
        search_query = data.question

    if intention == "recherche":
        url, content = await scrape_and_summarize(search_query)
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
            content=content
        )
    else:
        prompt_final = f"""L'utilisateur dit : "{data.question}" 
    

Réponds naturellement, de façon utile et concise."""
        return FinalResponse(
            action="just_groq",
            prompt=prompt_final.strip()
        )
