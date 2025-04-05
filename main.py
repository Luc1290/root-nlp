from fastapi import FastAPI
from pydantic import BaseModel
import httpx
import os
import json
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

class QuestionRequest(BaseModel):
    question: str

class IntentResult(BaseModel):
    intent: str
    entities: dict

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
        try:
            return response.json()[0]["generated_text"]
        except Exception as e:
            print("❌ Hugging Face Error:", e)
            return '{ "intent": "autre", "entities": {} }'

@app.post("/analyze", response_model=IntentResult)
async def analyze_question(data: QuestionRequest):
    print("📩 Question reçue :", data.question)

    prompt = f"""
Tu es un assistant d'analyse sémantique. Ton travail est de détecter l’intention d’une phrase et d’identifier les entités importantes.

⚠️ Tu dois répondre uniquement en JSON, sans ajouter d’explication ou de texte supplémentaire.

Voici le format que tu dois respecter, toujours :
{{
  "intent": "nom_de_l_intention",
  "entities": {{
    "clé1": "valeur1",
    "clé2": "valeur2"
  }}
}}

Phrase : "{data.question}"
"""

    try:
        response_text = await call_huggingface_model(prompt)
        print("🧠 Réponse brute HF:", response_text)

        parsed = json.loads(response_text.strip())  # Sécurisé
        intent = parsed.get("intent", "autre")
        entities = parsed.get("entities", {})

        return {
            "intent": intent,
            "entities": entities
        }

    except Exception as e:
        print("⚠️ Parsing error:", e)
        return {
            "intent": "autre",
            "entities": {}
        }
