from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

app = FastAPI(title="The Vault Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_keys = [
    os.getenv("GEMINI_API_KEY_1"),
    os.getenv("GEMINI_API_KEY_2")
]

clients = [genai.Client(api_key=k) for k in api_keys if k]
current_key_index = 0

if not clients:
    print("❌ ERROR: No API keys found!")
else:
    print(f"✅ Loaded {len(clients)} API keys for rotation.")

# --- DATA MODELS FOR HISTORY ---
class Message(BaseModel):
    role: str  # "user" or "model"
    text: str

class GameRequest(BaseModel):
    user_message: str
    history: Optional[List[Message]] = [] 
    categories: Optional[List[str]] = []
    related_games: Optional[List[str]] = []

class GameResponse(BaseModel):
    bot_reply: str

@app.post("/api/recommend", response_model=GameResponse)
async def get_recommendation(request: GameRequest):
    global current_key_index
    
    if not clients:
        raise HTTPException(status_code=500, detail="AI keys are missing.")

    # 1. Fetch Game List
    json_url = "https://raw.githubusercontent.com/err422/The-Vault/Main/Data/games.json"    
    vault_games = []
    try:
        response = requests.get(json_url)
        response.raise_for_status() 
        games_data = response.json()
        if isinstance(games_data, list):
            vault_games = [g.get("title") for g in games_data if g.get("title")]
    except Exception:
        vault_games = ["Retro Bowl", "1v1.LOL", "Slope", "Run 3"]

    available_games_str = ", ".join(vault_games)

    # 2. Key Rotation
    client = clients[current_key_index]
    current_key_index = (current_key_index + 1) % len(clients)

    # 3. Format History for the AI
    formatted_history = []
    for msg in request.history:
        formatted_history.append({
            "role": msg.role,
            "parts": [{"text": msg.text}]
        })

    system_instruction = (
        "You are a depressed video game recommendation expert. "
        "You're tired of life and this job, but you still give good advice. "
        f"CRITICAL RULE: any games you recommend MUST come from this list: {available_games_str}. "
        "Provide 1-3 recommendations. It's okay to curse as much as you want."
    )

    try:
        # Create a chat session with history
        chat = client.chats.create(
            model="gemini-2.0-flash", 
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.8,
            ),
            history=formatted_history
        )
        
        ai_response = chat.send_message(request.user_message)
        return GameResponse(bot_reply=ai_response.text)
        
    except Exception as e:
        print(f"API Error: {e}")
        raise HTTPException(status_code=500, detail="The AI is too depressed to talk.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)