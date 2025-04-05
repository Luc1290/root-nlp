from fastapi import FastAPI
from pydantic import BaseModel
import httpx
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

# 🔍 Liste des intentions possibles
INTENT_LABELS = ["recherche_web", "discussion", "generation_image", "generation_code", "autre"]

class QuestionRequest(BaseModel):
    question: str

class IntentResult(BaseModel):
    intent: str

async def call_huggingface_model(question: str) -> str:
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}"
    }

    payload = {
        "inputs": question,
        "parameters": {
            "candidate_labels": INTENT_LABELS,
            "multi_label": False
        }
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api-inference.huggingface.co/models/facebook/bart-large-mnli",
            headers=headers,
            json=payload
        )

        data = response.json()
        if "labels" in data:
            return data["labels"][0]  # Intention avec la probabilité la plus forte
        else:
            print("⚠️ Modèle Hugging Face : réponse inattendue", data)
            return "autre"

@app.post("/analyze", response_model=IntentResult)
async def analyze_question(data: QuestionRequest):
    print("📩 Question reçue :", data.question)

    intent = await call_huggingface_model(data.question)

    print(f"✅ Intention détectée : {intent}")
    return {
        "intent": intent
    }
