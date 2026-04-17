from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
import requests

# Load the secret API key from the .env file into the environment
load_dotenv()

# Initialize the FastAPI application
app = FastAPI(title="Game Recommender Chatbot API")

# Add CORS middleware to allow your frontend to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, you'd put your frontend's exact URL here. "*" allows everything for testing.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Initialize the Gemini Client
# This automatically looks for the GEMINI_API_KEY in your environment variables
try:
    client = genai.Client()
except Exception as e:
    print(f"Error initializing Gemini Client: {e}")
    client = None

# The data structure we expect from your frontend
class GameRequest(BaseModel):
    user_message: str
    categories: Optional[List[str]] = []
    related_games: Optional[List[str]] = []

# The data structure we will send back to your frontend
class GameResponse(BaseModel):
    bot_reply: str

@app.post("/api/recommend", response_model=GameResponse)
async def get_recommendation(request: GameRequest):
    if not client:
        raise HTTPException(status_code=500, detail="AI client not configured.")

    # 1. Fetch your live game data directly from GitHub
    raw_json_url = "https://raw.githubusercontent.com/err422/The-Vault/main/Data/games.json"    
    
    try:
        response = requests.get(raw_json_url)
        response.raise_for_status() # Throws an error if the URL is broken
        games_data = response.json()
        
        # 2. Extract the titles from the JSON
        # IMPORTANT: This assumes your JSON is a list of objects like [{"title": "Slope"}, {"title": "Run 3"}]
        # If your JSON uses "name" instead of "title", change game.get("title") to game.get("name")
        vault_games = [game.get("title") for game in games_data if game.get("title")]
        
    except Exception as e:
        print(f"Error fetching games list: {e}")
        # Fallback list just in case GitHub goes down
        vault_games = ["Retro Bowl", "1v1.LOL", "Slope", "Run 3"]

    # Convert the list into a comma-separated string
    available_games_str = ", ".join(vault_games)

    # 2. System Instructions: This is how we program the chatbot's personality and rules
    system_instruction = (
        "You are a depressed, video game recommendation expert. "
        "Your goal is to recommend games based on the user's message, preferred categories, and related games they enjoy. "
        "CRITICAL RULE: You must ONLY recommend games from this exact list: {available_games_str}. "
        "Provide 1 to 3 solid recommendations. For each, give a brief, engaging reason why it fits their tastes. "
        "Keep your formatting clean and readable, but its ok to curse"
    )
# 4. User Prompt
    user_prompt = (
        f"User Message: {request.user_message}\n\n"
        "Please recommend a game from the approved list."
    )

    try:
        response = requests.get(json_url)
        response.raise_for_status()
        games_data = response.json()
        
        vault_games = []
        
        # If the JSON is a single object (starts with { )
        if isinstance(games_data, dict):
            title = games_data.get("title")
            if title:
                vault_games.append(title)
        
        # If the JSON is a list (starts with [ )
        elif isinstance(games_data, list):
            for game in games_data:
                # Check for "title" or "id" just to see what we find
                title = game.get("title")
                if title:
                    vault_games.append(title)

        print(f"SUCCESSFULLY LOADED {len(vault_games)} GAMES.")
        
    except Exception as e:
        print(f"FAILED TO FETCH JSON: {e}")
        vault_games = ["Retro Bowl", "1v1.LOL", "Slope"] # The "I give up" list
        
        # 5. Call Gemini
        ai_response = client.models.generate_content(
            model="gemini-3-flash-preview", # Or whichever model string you successfully used earlier
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.7,
            )
        )
        
        return GameResponse(bot_reply=ai_response.text)
        
    except Exception as e:
        print(f"Gemini API Error: {e}")
        raise HTTPException(status_code=500, detail="Sorry, I couldn't think of a recommendation right now.")
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
