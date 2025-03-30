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

if __name__ == "__main__":
    try:
        test_recommender()
    except Exception as e:
        print(f"Error occurred: {str(e)}")