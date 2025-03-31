import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any
import spacy
from difflib import SequenceMatcher

from models.entity_extractor import EntityExtractor

class NetflixRecommender:
    def __init__(self):
        # Load spaCy model
        self.nlp = spacy.load("en_core_web_sm")
        
        # Load datasets
        self.movies_df = pd.read_csv('netflix_titles.csv')
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.prepare_content_features()
        self.extractor = EntityExtractor(self.movies_df)
        
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

    def recommend_by_director(self, director_name: str, n: int = 5) -> List[Dict]:
        """Strict recommendation based on exact director name"""
        director_name = director_name.strip().lower()
        df_directors = self.movies_df[self.movies_df['director'].notna()]

        # 1. Try exact match (case-insensitive)
        exact_match = df_directors[
            df_directors['director'].str.lower() == director_name
        ]
        print("[DEBUG] Director match results:", exact_match)
        if not exact_match.empty:
            return exact_match.head(n).to_dict('records')

        # 2. If no match, return empty to trigger fallback
        return []


    def recommend_by_actor(self, actor_name: str, n: int = 5) -> List[Dict]:
        actor_name = actor_name.strip().lower()
        df_cast = self.movies_df[self.movies_df['cast'].notna()]

        # Only return exact matches
        exact_match = df_cast[df_cast['cast'].str.lower().str.split(', ').apply(lambda x: actor_name in [a.strip().lower() for a in x])]
        print("[DEBUG] Actor match results:", exact_match)
        if not exact_match.empty:
            return exact_match.head(n).to_dict('records')
        
        return []


    def recommend_by_rating(self, rating: str, n_recommendations: int = 5) -> List[Dict]:
        """Recommendation based on rating (e.g., 'TV-MA', 'PG-13', 'R', etc.)"""
        rated_content = self.movies_df[
            self.movies_df['rating'] == rating
        ].sort_values('release_year', ascending=False)  # Prioritize recent content
        
        return rated_content.head(n_recommendations).to_dict('records')

    def recommend_by_genre(self, genre: str, keyword: str = None, n_recommendations: int = 5) -> List[Dict]:
        df = self.movies_df
        df['listed_in'] = df['listed_in'].fillna('')  # filter NaN
        
        genre_filtered = df[df['listed_in'].str.contains(genre, case=False, na=False)]
        
        if keyword:
            genre_filtered = genre_filtered[
                genre_filtered['description'].str.contains(keyword, case=False, na=False)
            ]
        
        if genre_filtered.empty:
            return []
        
        return genre_filtered.sample(n=min(n_recommendations, len(genre_filtered))).to_dict('records')


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
    
    def recommend_by_ner(self, message: str, n: int = 5) -> List[Dict]:
        """Use extracted entities to recommend content"""
        entities = self.extractor.extract_entities(message)

        # 1. Attempt to recommend based on person's name (actor or director)
        if entities["person"]:
            person = entities["person"][0]
            results = self.recommend_by_actor(person)
            if results:
                return results
            
            results = self.recommend_by_director(person)
            print("[DEBUG] Ner match results:", results)  # 这行很关键！！
            if results:
                return results

        # 2. Attempt to recommend based on movie/show title
        if entities["title"]:
            title = entities["title"][0]
            results = self.recommend_similar_content(title)
            if results:
                return results

        # 3. Attempt to recommend based on genre
        if entities["genre"]:
            genre = entities["genre"][0]
            return self.recommend_by_genre(genre)

        # 4. Fallback: custom message + random drama
        fallback_header = [{
            "title": "No matching recommendations found.",
            "description": "We couldn't find any matching content based on your input. Here are some popular drama titles you might like:",
            "release_year": ""
        }]

        # Ensure all required fields are present
        drama_df = self.movies_df[
            self.movies_df['listed_in'].str.contains("Drama", case=False, na=False)
        ].dropna(subset=["title", "description", "release_year"])

        # Generate recommendations safely
        random_recs = drama_df.sample(n=min(n, len(drama_df))).apply(
            lambda row: {
                "title": row["title"],
                "description": row["description"],
                "release_year": int(row["release_year"]) if pd.notna(row["release_year"]) else ""
            },
            axis=1
        ).tolist()

        return fallback_header + random_recs
