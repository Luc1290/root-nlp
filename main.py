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
            return "intention: autre\nentities: {}"

@app.post("/analyze", response_model=IntentResult)
async def analyze_question(data: QuestionRequest):
    print("📩 Question reçue :", data.question)

    prompt = f"""Analyse l’intention de la phrase ci-dessous et identifie les éléments utiles.

Donne ta réponse au format JSON comme ceci :
{{
  "intent": "recherche_web",
  "entities": {{
    "lieu": "Nantes",
    "type_info": "météo",
    "date": "aujourd’hui"
  }}
}}

Phrase : "{data.question}"
"""

    try:
        response_text = await call_huggingface_model(prompt)
        print("🧠 Réponse brute HF:", response_text)
        parsed = ast.literal_eval(response_text.strip())
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
