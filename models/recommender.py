import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any

class NetflixRecommender:
    def __init__(self):
        # Load datasets
        self.movies_df = pd.read_csv('netflix_titles.csv')
        
        # Create TF-IDF vectorizer for content-based filtering
        self.tfidf = TfidfVectorizer(stop_words='english')
        
        # Prepare content features
        self.prepare_content_features()
        
    def prepare_content_features(self):
        # Combine relevant features for content-based filtering
        self.movies_df['content'] = (
            self.movies_df['description'].fillna('') + ' ' +
            self.movies_df['cast'].fillna('') + ' ' +
            self.movies_df['director'].fillna('') + ' ' +
            self.movies_df['listed_in'].fillna('')  # genres
        )
        
        # Create TF-IDF matrix
        self.tfidf_matrix = self.tfidf.fit_transform(self.movies_df['content'])
        
        # Calculate similarity matrix
        self.content_similarity = cosine_similarity(self.tfidf_matrix)

    def recommend_similar_content(self, title: str, n_recommendations: int = 5) -> List[Dict]:
        """Content-based filtering: Recommendation based on title"""
        try:
            # Find the index of the given title
            idx = self.movies_df[self.movies_df['title'] == title].index[0]
            
            # Get similarity scores
            sim_scores = list(enumerate(self.content_similarity[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:n_recommendations+1]
            
            # Get movie indices
            movie_indices = [i[0] for i in sim_scores]
            
            # Return recommended movies
            recommendations = self.movies_df.iloc[movie_indices]
            return recommendations.to_dict('records')
        except:
            return []

    def recommend_by_director(self, director: str, n_recommendations: int = 5) -> List[Dict]:
        """Recommendation based on director"""
        # Convert to lowercase for case-insensitive matching
        director_lower = director.lower().replace('-', ' ').strip()
        
        director_movies = self.movies_df[
            self.movies_df['director'].str.lower().str.replace('-', ' ').str.contains(director_lower, na=False)
        ].sort_values('release_year', ascending=False)
        
        if director_movies.empty:
            return []
        
        return director_movies.head(n_recommendations).to_dict('records')

    def recommend_by_actor(self, actor: str, n_recommendations: int = 5) -> List[Dict]:
        """Recommendation based on actor"""
        actor_content = self.movies_df[
            self.movies_df['cast'].str.contains(actor, na=False)
        ].sort_values('rating', ascending=False)
        
        return actor_content.head(n_recommendations).to_dict('records')

    def recommend_by_rating(self, rating: str, n_recommendations: int = 5) -> List[Dict]:
        """Recommendation based on rating (e.g., 'TV-MA', 'PG-13', 'R', etc.)"""
        rated_content = self.movies_df[
            self.movies_df['rating'] == rating
        ].sort_values('release_year', ascending=False)  # Prioritize recent content
        
        return rated_content.head(n_recommendations).to_dict('records')

    def recommend_by_genre(self, genre: str, n_recommendations: int = 5) -> List[Dict]:
        """Recommendation based on genre"""
        genre_content = self.movies_df[
            self.movies_df['listed_in'].str.contains(genre, na=False)
        ].sort_values('rating', ascending=False)
        
        return genre_content.head(n_recommendations).to_dict('records')

    def recommend_by_multi(self, 
                         genre: str = None, 
                         director: str = None, 
                         actor: str = None,
                         rating: str = None,  # Changed from float to str
                         n_recommendations: int = 5) -> List[Dict]:
        """Recommendation based on multiple conditions"""
        filtered_df = self.movies_df.copy()
        
        if genre:
            filtered_df = filtered_df[
                filtered_df['listed_in'].str.contains(genre, na=False)
            ]
        if director:
            filtered_df = filtered_df[
                filtered_df['director'].str.contains(director, na=False)
            ]
        if actor:
            filtered_df = filtered_df[
                filtered_df['cast'].str.contains(actor, na=False)
            ]
        if rating:
            filtered_df = filtered_df[
                filtered_df['rating'] == rating  # Changed >= operator to ==
            ]
            
        return filtered_df.sort_values('release_year', ascending=False).head(n_recommendations).to_dict('records')