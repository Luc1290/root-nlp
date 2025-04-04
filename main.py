from fastapi import FastAPI, Request
from pydantic import BaseModel
import httpx
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

class QuestionRequest(BaseModel):
    question: str

class NLPResponse(BaseModel):
    intention: str
    entities: list[str]
    search_query: str

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

@app.post("/analyze", response_model=NLPResponse)
async def analyze_question(data: QuestionRequest):
    prompt = f"""Analyse la phrase suivante et donne :
- une intention (recherche, commande, salutation, autre)
- les mots-clés principaux (sous forme de liste)
- une requête de recherche optimisée pour un moteur

Phrase : "{data.question}" """

    response_text = await call_huggingface_model(prompt)

    # 💡 Très simple parser naïf pour ce MVP
    try:
        lines = response_text.strip().split("\n")
        intention = lines[0].split(":")[1].strip()
        entities = eval(lines[1].split(":")[1].strip())
        search_query = lines[2].split(":")[1].strip()
    except:
        intention = "recherche"
        entities = []
        search_query = data.question

    return NLPResponse(
        intention=intention,
        entities=entities,
        search_query=search_query
    )
