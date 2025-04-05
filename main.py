from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import httpx
import os
from dotenv import load_dotenv
import time
from fastapi.middleware.cors import CORSMiddleware
import logging

# Configurer le logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI()

# Ajouter CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Ou spécifier les domaines exacts
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
if not HF_API_TOKEN:
    logger.error("🚨 HF_API_TOKEN manquant dans les variables d'environnement")
    # On continue l'exécution, mais on avertit 

# 🔍 Liste des intentions possibles
INTENT_LABELS = ["recherche_web", "discussion", "generation_image", "generation_code", "autre"]

# Modèle de secours si HuggingFace échoue
DEFAULT_INTENT_RULES = {
    "code": "generation_code",
    "programme": "generation_code",
    "script": "generation_code",
    "function": "generation_code",
    "cherche": "recherche_web",
    "trouve": "recherche_web",
    "recherche": "recherche_web",
    "quand": "recherche_web",
    "qui est": "recherche_web",
    "qu'est-ce que": "recherche_web",
    "dessine": "generation_image",
    "image": "generation_image",
    "photo": "generation_image",
}

class QuestionRequest(BaseModel):
    question: str

class IntentResult(BaseModel):
    intent: str
    confidence: float = 0.0
    
# Cache simple pour éviter d'appeler HF trop souvent
intent_cache = {}
MAX_CACHE_SIZE = 1000

async def call_huggingface_model(question: str) -> tuple[str, float]:
    """Appelle le modèle Hugging Face pour classer l'intention"""
    # Vérifier si l'intention est déjà en cache
    if question in intent_cache:
        cached_intent, confidence = intent_cache[question]
        logger.info(f"🔄 Utilisation du cache pour: '{question[:30]}...' -> {cached_intent}")
        return cached_intent, confidence
    
    # Si pas de token, utilisez le fallback
    if not HF_API_TOKEN:
        logger.warning("⚠️ HF_API_TOKEN non trouvé, utilisation des règles par défaut")
        return fallback_intent_detection(question)
    
    start_time = time.time()
    logger.info(f"📤 Envoi à HuggingFace: '{question[:50]}...'")
    
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

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api-inference.huggingface.co/models/facebook/bart-large-mnli",
                headers=headers,
                json=payload,
                timeout=10.0  # Timeout de 10 secondes
            )

            elapsed = time.time() - start_time
            logger.info(f"⏱️ Réponse HuggingFace reçue en {elapsed:.2f}s")
            
            if response.status_code != 200:
                logger.error(f"🚨 Erreur HuggingFace: {response.status_code} {response.text[:100]}")
                return fallback_intent_detection(question)
                
            data = response.json()
            if "labels" in data and "scores" in data:
                intent = data["labels"][0]
                confidence = data["scores"][0]
                
                # Mettre en cache
                if len(intent_cache) >= MAX_CACHE_SIZE:
                    # Supprimer une entrée aléatoire si le cache est plein
                    intent_cache.pop(next(iter(intent_cache)))
                intent_cache[question] = (intent, confidence)
                
                logger.info(f"✅ Intention détectée: {intent} (confiance: {confidence:.2f})")
                return intent, confidence
            else:
                logger.warning(f"⚠️ Réponse HF inattendue: {data}")
                return fallback_intent_detection(question)
    except Exception as e:
        logger.error(f"🚨 Exception lors de l'appel HF: {str(e)}")
        return fallback_intent_detection(question)

def fallback_intent_detection(question: str) -> tuple[str, float]:
    """Méthode de secours pour détecter l'intention si HuggingFace échoue"""
    question_lower = question.lower()
    
    # Recherche de mots-clés dans la question
    for keyword, intent in DEFAULT_INTENT_RULES.items():
        if keyword in question_lower:
            logger.info(f"🔍 Intention détectée par règles: {intent} (mot-clé: {keyword})")
            return intent, 0.7  # Confiance arbitraire
    
    # Par défaut, on considère que c'est une discussion
    logger.info("🔄 Aucun mot-clé trouvé, utilisation de l'intention par défaut: discussion")
    return "discussion", 0.5

@app.post("/analyze", response_model=IntentResult)
async def analyze_question(data: QuestionRequest, request: Request):
    start_time = time.time()
    client_ip = request.client.host if request.client else "unknown"
    
    logger.info(f"📩 Question reçue de {client_ip}: '{data.question[:50]}...'")

    # Ajouter un délai artificiel en cas d'appels trop rapides
    if hasattr(app, "last_request_time"):
        time_since_last = time.time() - app.last_request_time
        if time_since_last < 0.1:  # Si moins de 100ms depuis la dernière requête
            await asyncio.sleep(0.2)  # Attendre 200ms

    app.last_request_time = time.time()

    try:
        intent, confidence = await call_huggingface_model(data.question)
        
        elapsed = time.time() - start_time
        logger.info(f"✅ Traitement terminé en {elapsed:.2f}s: {intent} ({confidence:.2f})")
        
        return {
            "intent": intent,
            "confidence": confidence
        }
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'analyse: {str(e)}")
        # En cas d'erreur, retourner une intention par défaut
        return {
            "intent": "discussion",
            "confidence": 0.5
        }

@app.get("/health")
async def health_check():
    """Endpoint de vérification de santé"""
    return {"status": "ok", "service": "root-nlp-service"}

if __name__ == "__main__":
    import uvicorn
    import asyncio
    
    app.last_request_time = time.time()
    
    port = int(os.getenv("PORT", 8000))
    logger.info(f"🚀 Démarrage du service NLP sur le port {port}")
    
    uvicorn.run(app, host="0.0.0.0", port=port)