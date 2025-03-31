import sys
import os

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.recommender import NetflixRecommender

def test_recommender():
    # Initialize the recommendation system
    recommender = NetflixRecommender()
    
    # 1. Test similar content recommendation
    print("\n1. Similar content recommendation (Based on 'Stranger Things'):")
    similar = recommender.recommend_similar_content("Stranger Things")
    for item in similar:
        print(f"- {item['title']} ({item['type']})")
    
    # 2. Test director-based recommendation
    print("\n2. Director-based recommendation (Movies by 'Bong Joon-ho'):")
    director_recs = recommender.recommend_by_director("Bong Joon-ho")
    for item in director_recs:
        print(f"- {item['title']} ({item['release_year']})")
    
    # 3. Test actor-based recommendation
    print("\n3. Actor-based recommendation (Movies featuring 'Tom Hanks'):")
    actor_recs = recommender.recommend_by_actor("Tom Hanks")
    for item in actor_recs:
        print(f"- {item['title']} ({item['release_year']})")
    
    # 4. Test rating-based recommendation
    print("\n4. Rating-based recommendation (TV-MA rating):")
    rating_recs = recommender.recommend_by_rating("TV-MA")
    for item in rating_recs:
        print(f"- {item['title']} ({item['rating']})")
    
    # 5. Test genre-based recommendation
    print("\n5. Genre-based recommendation ('Action' genre):")
    genre_recs = recommender.recommend_by_genre("Action")
    for item in genre_recs:
        print(f"- {item['title']} ({item['listed_in']})")
    
    # 6. Test multi-factor recommendation
    print("\n6. Multi-factor recommendation (Action genre & TV-MA rating):")
    multi_recs = recommender.recommend_by_multi(genre="Action", rating="TV-MA")
    for item in multi_recs:
        print(f"- {item['title']} ({item['rating']}, {item['listed_in']})")

    # 7. Test NER-based recommendation
    print("\n7. NER-based recommendation:")
    ner_results = recommender.recommend_by_ner("I love Tom Hanks movies")
    for item in ner_results:
        print(f"- {item['title']} ({item['release_year']})")

def test_named_entity_extraction():
    # Initialize the recommendation system
    recommender = NetflixRecommender()
    input_text = "I want something with Emma Watson"
    result = recommender.recommend_by_ner(input_text)
    assert "Emma Watson" in result or "Sorry" not in result
    print("test_named_entity_extraction passed.")

def test_no_entity():
    # Initialize the recommendation system
    recommender = NetflixRecommender()
    input_text = "Blah blah nothing known here"
    result = recommender.recommend_by_ner(input_text)
    assert "Sorry" in result
    print("test_no_entity passed.")

def test_actor_recommendation_format():
    # Initialize the recommendation system
    recommender = NetflixRecommender()
    input_text = "Brad Pitt"
    result = recommender.recommend_by_ner(input_text)
    assert result.startswith("Here are the recommended"), "Format check failed"
    print("test_actor_recommendation_format passed.")

if __name__ == "__main__":
    try:
        test_recommender()
        test_named_entity_extraction()
        test_no_entity()
        test_actor_recommendation_format()
    except Exception as e:
        print(f"Error occurred: {str(e)}")