from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load the secret API keys from your .env file
load_dotenv()

app = FastAPI(title="The Vault Chatbot API")

# Add CORS middleware so your GitHub site can talk to your PC
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- KEY ROTATION SETUP ---
api_keys = [
    os.getenv("GEMINI_API_KEY_1"),
    os.getenv("GEMINI_API_KEY_2")
]

# Create a list of working clients
clients = [genai.Client(api_key=k) for k in api_keys if k]
current_key_index = 0

if not clients:
    print("❌ ERROR: No API keys found! Make sure .env has GEMINI_API_KEY_1 and _2")
else:
    print(f"✅ Loaded {len(clients)} API keys for rotation.")

class GameRequest(BaseModel):
    user_message: str
    categories: Optional[List[str]] = []
    related_games: Optional[List[str]] = []

class GameResponse(BaseModel):
    bot_reply: str

@app.post("/api/recommend", response_model=GameResponse)
async def get_recommendation(request: GameRequest):
    global current_key_index
    
    # Check if we have any working AI clients
    if not clients:
        raise HTTPException(status_code=500, detail="AI keys are missing.")

    # 1. Fetch your live game list from GitHub
    json_url = "https://raw.githubusercontent.com/err422/The-Vault/Main/Data/games.json"    
    vault_games = []

    try:
        response = requests.get(json_url)
        response.raise_for_status() 
        games_data = response.json()
        
        # Extract titles (handles single objects or lists)
        if isinstance(games_data, dict):
            if games_data.get("title"): vault_games.append(games_data.get("title"))
        elif isinstance(games_data, list):
            for game in games_data:
                if isinstance(game, dict) and game.get("title"):
                    vault_games.append(game.get("title"))

        print(f"Successfully loaded {len(vault_games)} games from GitHub.")
        
    except Exception as e:
        print(f"JSON Fetch Failed: {e}. Using fallback list.")
        vault_games = ["Retro Bowl", "1v1.LOL", "Slope", "Run 3"]

    available_games_str = ", ".join(vault_games)

    # 2. Pick the NEXT AI client in the rotation
    client = clients[current_key_index]
    print(f"Using Key Index: {current_key_index}") # Helps you see it switching!
    current_key_index = (current_key_index + 1) % len(clients)

    # 3. Setup the Depressed Expert Persona
    system_instruction = (
        "You are a depressed video game recommendation expert. "
        "You're tired of life and this job, but you still give good advice. "
        f"CRITICAL RULE: any games you recommend MUST come from this list: {available_games_str}. "
        "Provide 1-3 recommendations. It's okay to curse as much as you want "
    )

    try:
        # Generate the response
        ai_response = client.models.generate_content(
            model="gemini-2.5-flash", 
            contents=f"The user wants: {request.user_message}",
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.8,
            )
        )
        
        return GameResponse(bot_reply=ai_response.text)
        
    except Exception as e:
        print(f"Gemini API Error: {e}")
        raise HTTPException(status_code=500, detail="The AI is too depressed to talk right now.")

if __name__ == "__main__":
    import uvicorn
    # This runs the server on your local PC
    uvicorn.run(app, host="0.0.0.0", port=8000)