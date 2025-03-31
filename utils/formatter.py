from typing import List, Dict

def format_recommendations(recommendations: List[Dict], category: str) -> str:
    """Format recommendation results for readability."""
    if not recommendations:
        return f"Sorry, we couldn't find any {category}."

    if recommendations[0].get("title") == "No matching recommendations found.":
        result = f"{recommendations[0].get('description', '')}\n\n"
        recommendations = recommendations[1:]
    else:
        result = f"Here are the recommended {category}:\n\n"

    for i, item in enumerate(recommendations, 1):
        year = f" ({item['release_year']})" if item.get("release_year") else ""
        result += f"{i}. {item['title']}{year}\n"

    return result 