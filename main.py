from typing import Union, Dict, Any, List
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from models.recommender import NetflixRecommender

app = FastAPI()

recommender = NetflixRecommender()

@app.get("/")
def read_root():
    return {"message": "Netflix Recommender API is running"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

# Pydantic models for Dialogflow request structure
class Intent(BaseModel):
    displayName: str

class QueryResult(BaseModel):
    intent: Intent
    parameters: Dict[str, Any]

class DialogflowRequest(BaseModel):
    queryResult: QueryResult

@app.post("/webhook")
async def dialogflow_webhook(request: DialogflowRequest):
    # Extract information from Dialogflow request
    intent = request.queryResult.intent.displayName
    parameters = request.queryResult.parameters
    
    # Process recommendation based on intent
    response_text = ""
    try:
        if intent == "recommend_similar_content":
            title = parameters.get("title", "")
            recommendations = recommender.recommend_similar_content(title)
            response_text = format_recommendations(recommendations, "Similar Content")
            
        elif intent == "recommend_by_director":
            director = parameters.get("director", "")
            recommendations = recommender.recommend_by_director(director)
            response_text = format_recommendations(recommendations, f"Movies by Director {director}")
            
        elif intent == "recommend_by_actor":
            actor = parameters.get("actor", "")
            recommendations = recommender.recommend_by_actor(actor)
            response_text = format_recommendations(recommendations, f"Movies starring {actor}")
            
        elif intent == "recommend_by_rating":
            rating = parameters.get("rating", "")
            recommendations = recommender.recommend_by_rating(rating)
            response_text = format_recommendations(recommendations, f"Content with {rating} Rating")
            
        elif intent == "recommend_by_genre":
            genre = parameters.get("genre", "")
            recommendations = recommender.recommend_by_genre(genre)
            response_text = format_recommendations(recommendations, f"{genre} Genre Content")
            
        elif intent == "recommend_by_multi":
            genre = parameters.get("genre", None)
            director = parameters.get("director", None)
            actor = parameters.get("actor", None)
            rating = parameters.get("rating", None)
            
            recommendations = recommender.recommend_by_multi(
                genre=genre,
                director=director,
                actor=actor,
                rating=rating
            )
            
            # Combine conditions into a string
            conditions = []
            if genre:
                conditions.append(f"Genre: {genre}")
            if director:
                conditions.append(f"Director: {director}")
            if actor:
                conditions.append(f"Actor: {actor}")
            if rating:
                conditions.append(f"Rating: {rating}")
                
            category = "Recommended content based on the following conditions (" + ", ".join(conditions) + ")"
            response_text = format_recommendations(recommendations, category)
        elif intent == "recommend_by_text":
            # Flexible free-text recommendation using NER
            user_input = parameters.get("text", "")
            recommendations = recommender.recommend_by_ner(user_input)
            response_text = format_recommendations(recommendations, f"Recommendations based on your input: \"{user_input}\"")
    
        else:
            response_text = "Sorry, we couldn't find the requested recommendation feature."
            
    except Exception as e:
        response_text = f"Sorry, an error occurred: {str(e)}"
    
    # Dialogflow response format
    return {
        "fulfillmentText": response_text,
        "fulfillmentMessages": [
            {
                "text": {
                    "text": [response_text]
                }
            }
        ]
    }

def format_recommendations(recommendations: List[Dict], category: str) -> str:
    """Format recommendation results for readability."""
    if not recommendations:
        return f"Sorry, we couldn't find any {category}."

    # Check if the first item is a fallback message
    if recommendations[0].get("title") == "No matching recommendations found.":
        # Start with the fallback message's description
        result = f"{recommendations[0].get('description', '')}\n\n"
        # Skip the fallback item and continue formatting the rest
        recommendations = recommendations[1:]
    else:
        result = f"Here are the recommended {category}:\n\n"

    # Format each recommendation with title and optional release year
    for i, item in enumerate(recommendations, 1):
        year = f" ({item['release_year']})" if item.get("release_year") else ""
        result += f"{i}. {item['title']}{year}\n"

    return result
