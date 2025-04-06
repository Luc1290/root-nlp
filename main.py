from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import httpx
import os
from dotenv import load_dotenv
import time
from fastapi.middleware.cors import CORSMiddleware
import logging
import asyncio
import re

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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
if not HF_API_TOKEN:
    logger.warning("🚨 HF_API_TOKEN manquant dans les variables d'environnement")

# 🔍 Liste des intentions possibles
INTENT_LABELS = [
    "recherche_web",           # chercher une info
    "discussion",            # discuter, parler
    "generation_image",        # créer une image
    "generation_code",         # générer du code
    "generation_texte",        # rédiger, inventer
    "analyse_donnee",          # comprendre ou synthétiser des infos
    "planification",           # demander de l'organisation
    "conseil_emotionnel",      # besoin de soutien ou de motivation
    "question_personnelle",    # introspection ou autoanalyse
    "autre"                    # tout ce qui ne rentre dans rien
]


# Règles de secours au cas où Hugging Face échoue
FALLBACK_RULES = {
    # Mots-clés qui indiquent fortement une intention
    "keywords": {
        "generation_code": ["code", "programme", "script", "fonction", "programmer", "développer"],
        "recherche_web": ["cherche", "météo", "président", "capitale", "définition"],
        "generation_image": ["dessine", "image", "visualise", "dessin"]
    },
    
    # Patterns qui indiquent fortement une intention
    "patterns": {
        "recherche_web": [
            r"(?i).*météo.*",
            r"(?i).*quel temps.*à.*",
            r"(?i).*qui est le président.*",
            r"(?i).*quelle est la capitale.*",
            r"(?i).*où se trouve.*",
            r"(?i).*combien.*coûte.*",
            r"(?i).*parapluie.*demain.*",
            r"(?i).*faut[- ]il.*parapluie.*",
            r"(?i).*vais[- ]je.*prendre.*parapluie.*",
            r"(?i).*pleuvoir.*demain.*",
            r"(?i).*pluie.*demain.*",
            

        ],
        "generation_image": [
            r"(?i)dessine[- ]moi.*",
            r"(?i)génère[- ]moi une image.*",
        ],
        "generation_code": [
            r"(?i)écris[- ]moi un (code|programme|script).*",
            r"(?i)comment coder.*",
        ]
    }
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
    
    # Si pas de token, utiliser le fallback
    if not HF_API_TOKEN:
        logger.warning("⚠️ HF_API_TOKEN non trouvé, utilisation des règles par défaut")
        return fallback_intent_detection(question)
    
    start_time = time.time()
    logger.info(f"📤 Envoi à HuggingFace: '{question[:50]}...'")
    
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}"
    }

    # Amélioration du prompt pour mieux diriger le modèle
    payload = {
        "inputs": question,
        "parameters": {
            "candidate_labels": INTENT_LABELS,
            "multi_label": False,
            "hypothesis_template": "Dans le contexte de cette requête, l'utilisateur souhaite principalement obtenir un résultat relevant de la catégorie suivante : {}."
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
                
                # Si la confiance est très faible, on vérifie avec les règles
                if confidence < 0.6:
                    logger.warning(f"⚠️ Confiance faible ({confidence:.2f}), vérification avec règles")
                    fallback_intent, fallback_confidence = fallback_intent_detection(question)
                    
                    # Si les règles ont une confiance plus élevée, on les utilise
                    if fallback_confidence > confidence + 0.1:  # +0.1 pour favoriser HF quand c'est proche
                        logger.info(f"🔄 Utilisation du fallback: {fallback_intent} (confiance: {fallback_confidence:.2f})")
                        intent = fallback_intent
                        confidence = fallback_confidence
                
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
    
    # Vérifier d'abord les patterns
    for intent, patterns in FALLBACK_RULES["patterns"].items():
        for pattern in patterns:
            if re.match(pattern, question):
                logger.info(f"🔍 Pattern détecté pour {intent}: {pattern}")
                return intent, 0.85
    
    # Ensuite vérifier les mots-clés
    for intent, keywords in FALLBACK_RULES["keywords"].items():
        for keyword in keywords:
            if keyword.lower() in question_lower:
                logger.info(f"🔑 Mot-clé détecté pour {intent}: {keyword}")
                return intent, 0.8
    
    # Analyse des mots interrogatifs pour la recherche web
    if question.strip().endswith("?") and any(question_lower.startswith(w) for w in 
                                             ["qui", "que", "quoi", "quel", "quelle", 
                                              "quels", "quelles", "où", "comment", 
                                              "pourquoi", "quand", "combien"]):
        logger.info("❓ Question détectée, suggestion de 'recherche_web'")
        return "recherche_web", 0.7
    
    # Par défaut, on considère que c'est une discussion
    logger.info("💬 Aucun pattern spécifique trouvé, utilisation de l'intention par défaut 'discussion'")
    return "discussion", 0.6

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
    
    app.last_request_time = time.time()
    
    port = int(os.getenv("PORT", 8000))
    logger.info(f"🚀 Démarrage du service NLP sur le port {port}")
    
    uvicorn.run(app, host="0.0.0.0", port=port)