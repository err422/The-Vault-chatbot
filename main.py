from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

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
    """
    Endpoint that receives frontend data, formats a prompt, 
    calls Gemini, and returns the AI's game recommendation.
    """
    if not client:
        raise HTTPException(status_code=500, detail="AI client not configured.")

    # 1. Format the lists into readable strings for the AI
    categories_str = ", ".join(request.categories) if request.categories else "None specified"
    related_games_str = ", ".join(request.related_games) if request.related_games else "None specified"

    # 2. System Instructions: This is how we program the chatbot's personality and rules
    system_instruction = (
        "You are a depressed, video game recommendation expert. "
        "Your goal is to recommend games based on the user's message, preferred categories, and related games they enjoy. "
        "Provide 1 to 3 solid recommendations. For each, give a brief, engaging reason why it fits their tastes. "
        "Keep your formatting clean and readable, but its ok to curse"
    )

    # 3. User Prompt: This combines the user's chat message with their frontend selections
    user_prompt = (
        f"User Message: {request.user_message}\n"
        f"Categories: {categories_str}\n"
        f"Related Games: {related_games_str}\n\n"
        "Please recommend some games!"
    )

    try:
        # 4. Make the API Call to Gemini
        # We are using gemini-3-flash because it is extremely fast and cost-effective
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.7, # 0.0 is robotic/strict, 1.0 is highly creative. 0.7 is a good balance.
            )
        )
        
        # 5. Return the AI's text back to the frontend!
        return GameResponse(bot_reply=response.text)
        
    except Exception as e:
        print(f"Gemini API Error: {e}")
        raise HTTPException(status_code=500, detail="Sorry, I couldn't think of a recommendation right now. Try again later!")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)