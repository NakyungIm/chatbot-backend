from fastapi import APIRouter
from models.schemas import DialogflowRequest
from models.recommender import NetflixRecommender
from utils.formatter import format_recommendations
from typing import Dict, Any

router = APIRouter()
recommender = NetflixRecommender()

@router.post("/webhook")
async def dialogflow_webhook(request: DialogflowRequest):
    intent = request.queryResult.intent.displayName
    parameters = request.queryResult.parameters
    
    try:
        response_text = process_intent(intent, parameters)
    except Exception as e:
        response_text = f"Sorry, an error occurred: {str(e)}"
    
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

def process_intent(intent: str, parameters: Dict[str, Any]) -> str:
    intent_processors = {
        "recommend_similar_content": process_similar_content,
        "recommend_by_director": process_director_recommendation,
        "recommend_by_actor": process_actor_recommendation,
        "recommend_by_genre": process_genre_recommendation,
        "recommend_by_multi": process_multi_recommendation,
        "recommend_by_text": process_text_recommendation
    }
    
    processor = intent_processors.get(intent)
    if processor:
        return processor(parameters)
    return "Sorry, we couldn't find the requested recommendation feature."

def process_similar_content(parameters: Dict[str, Any]) -> str:
    title = parameters.get("title")
    if isinstance(title, list):
        title = title[0] if title else ""
    
    if not title:
        return "Please provide a movie or show title."
    
    recommendations = recommender.recommend_similar_content(str(title))
    if not recommendations:
        return f"Sorry, we couldn't find any content similar to '{title}'. Please try another title."
    
    return format_recommendations(recommendations, f"Similar to '{title}'")

def process_director_recommendation(parameters: Dict[str, Any]) -> str:
    """Process director-based recommendation request"""
    director = parameters.get("director_name", "")
    if isinstance(director, list):
        director = director[0] if director else ""
    
    if not director:
        return "Please provide a director name."
    
    recommendations = recommender.recommend_by_director(str(director))
    return format_recommendations(recommendations, f"Movies by Director {director}")

def process_actor_recommendation(parameters: Dict[str, Any]) -> str:
    """Process actor-based recommendation request"""
    actor = parameters.get("cast_name", "")
    if isinstance(actor, list):
        actor = actor[0] if actor else ""
    
    if not actor:
        return "Please provide an actor name."
    
    recommendations = recommender.recommend_by_actor(str(actor))
    return format_recommendations(recommendations, f"Movies starring {actor}")

def process_genre_recommendation(parameters: Dict[str, Any]) -> str:
    """Process genre-based recommendation request"""
    genre = parameters.get("genre")
    if isinstance(genre, list):
        genre = genre[0] if genre else ""
    
    if not genre:
        return "Please specify a genre."
    
    recommendations = recommender.recommend_by_genre(str(genre))
    return format_recommendations(recommendations, f"{genre} genre")

def process_multi_recommendation(parameters: Dict[str, Any]) -> str:
    """Process multi-criteria recommendation request"""
    # Handle parameters that might be lists
    genre = parameters.get("genre")
    director = parameters.get("director")
    actor = parameters.get("actor")
    rating = parameters.get("rating")
    country = parameters.get("country")
    
    # Convert list parameters to strings if necessary
    for param in [genre, director, actor, rating, country]:
        if isinstance(param, list):
            param = param[0] if param else None
    
    # Convert all parameters to strings if they exist
    genre = str(genre) if genre else None
    director = str(director) if director else None
    actor = str(actor) if actor else None
    rating = str(rating) if rating else None
    country = str(country) if country else None
    
    recommendations = recommender.recommend_by_multi(
        genre=genre,
        director=director,
        actor=actor,
        rating=rating,
        country=country
    )
    
    # Combine conditions into string
    conditions = []
    if country:
        conditions.append(f"Country: {country}")
    if genre:
        conditions.append(f"Genre: {genre}")
    if director:
        conditions.append(f"Director: {director}")
    if actor:
        conditions.append(f"Actor: {actor}")
    if rating:
        conditions.append(f"Rating: {rating}")
    
    category = "Content matching the following criteria (" + ", ".join(conditions) + ")"
    return format_recommendations(recommendations, category)

# ... other processing functions ... 
def process_text_recommendation(parameters):
    user_input = parameters.get("text", "")
    recommendations = recommender.recommend_by_ner(user_input)
    category = f'"{user_input}"'
    return format_recommendations(recommendations, category)