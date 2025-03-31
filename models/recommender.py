import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any
import spacy

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
        """Prepare content features for similarity calculation"""
        # Combine relevant features for content-based filtering
        self.movies_df['content'] = (
            self.movies_df['description'].fillna('') + ' ' +
            self.movies_df['cast'].fillna('') + ' ' +
            self.movies_df['director'].fillna('') + ' ' +
            self.movies_df['listed_in'].fillna('')  # genres
        ).str.lower()  # Convert to lowercase
        
        # Create TF-IDF matrix
        self.tfidf_matrix = self.tfidf.fit_transform(self.movies_df['content'])
        
        # Calculate similarity matrix
        self.content_similarity = cosine_similarity(self.tfidf_matrix)

    def recommend_similar_content(self, title: str, n_recommendations: int = 5) -> List[Dict]:
        """Content-based recommendation based on title"""
        try:
            # Ensure title is a string and convert to lowercase
            title = str(title).lower()
            
            # Find movies with similar titles using flexible matching
            matching_titles = self.movies_df[
                self.movies_df['title'].str.lower().str.contains(title, na=False, regex=False)
            ]
            
            if matching_titles.empty:
                # Try more flexible matching
                words = title.split()
                for word in words:
                    if len(word) > 3:  # Only use words longer than 3 characters
                        matching_titles = self.movies_df[
                            self.movies_df['title'].str.lower().str.contains(word, na=False, regex=False)
                        ]
                        if not matching_titles.empty:
                            break
            
            if matching_titles.empty:
                return []
            
            # Get the first matching title's index
            idx = matching_titles.index[0]
            
            # Get similarity scores
            sim_scores = list(enumerate(self.content_similarity[idx]))
            
            # Sort movies by similarity score
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            
            # Get top N similar movies (excluding the input movie)
            sim_scores = [s for s in sim_scores if s[0] != idx and s[1] > 0][:n_recommendations]
            
            if not sim_scores:
                return []
            
            # Get movie indices
            movie_indices = [i[0] for i in sim_scores]
            
            # Return recommended movies
            recommendations = self.movies_df.iloc[movie_indices]
            return recommendations.to_dict('records')
        
        except Exception as e:
            print(f"Error in recommend_similar_content: {str(e)}")
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

    def recommend_by_genre(self, genre: str, n_recommendations: int = 5) -> List[Dict]:
        """Genre-based recommendation"""
        try:
            # Ensure genre is a string and convert to lowercase
            genre = str(genre).lower()
            
            # Genre mapping dictionary
            genre_mapping = {
                'western': 'westerns',
                'romantic': 'romantic movies',
                'comedy': 'comedies',
                'action': 'action & adventure',
                'drama': 'dramas',
                'horror': 'horror movies',
                'documentary': 'documentaries',
                'sci-fi': 'sci-fi & fantasy',
                'thriller': 'thrillers',
                'crime': 'crime tv shows',
                'mystery': 'mysteries',
                'fantasy': 'sci-fi & fantasy',
                'family': 'family movies',
                'anime': 'anime features',
                'sports': 'sports movies',
                'animated': 'children & family movies',
                'animation': 'children & family movies',
                'cartoon': 'children & family movies',
                'kids': 'children & family movies',
                'romance': 'romantic movies',
                'adventure': 'action & adventure',
                'animation': 'anime features',
                'children': 'family movies',
                'crime': 'crime tv shows',
                'documentary': 'documentaries',
                'music': 'music & musicals',
                'reality-tv': 'reality tv',
                'short': 'short films',
                'talk': 'talk shows',
                'war': 'war & politics',
            }
            
            # Get the mapped genre or use original if no mapping exists
            search_genre = genre_mapping.get(genre, genre)
            
            # Find content with matching genre (case-insensitive)
            genre_content = self.movies_df[
                self.movies_df['listed_in'].str.lower().str.contains(search_genre, na=False)
            ]
            
            if genre_content.empty:
                # Try searching with original genre if mapped genre returned no results
                genre_content = self.movies_df[
                    self.movies_df['listed_in'].str.lower().str.contains(genre, na=False)
                ]
            
            if genre_content.empty:
                return []
            
            # Sort by release year (most recent first)
            genre_content = genre_content.sort_values('release_year', ascending=False)
            
            return genre_content.head(n_recommendations).to_dict('records')
        
        except Exception as e:
            print(f"Error in recommend_by_genre: {str(e)}")
            return []


    def recommend_by_multi(self, 
                          genre: str = None, 
                          director: str = None, 
                          actor: str = None,
                          rating: str = None,
                          release_year: int = None,
                          country: str = None,
                          n_recommendations: int = 5) -> List[Dict]:
        """Multi-criteria based recommendation"""
        filtered_df = self.movies_df.copy()
        
        if country:
            # Handle common variations of country names
            country_mapping = {
                'indian': 'india',
                'american': 'united states',
                'british': 'united kingdom',
                'korean': 'south korea',
                'chinese': 'china',
                'japanese': 'japan'
            }
            search_country = country_mapping.get(country.lower(), country.lower())
            filtered_df = filtered_df[
                filtered_df['country'].str.lower().str.contains(search_country, na=False)
            ]
        
        if genre:
            filtered_df = filtered_df[
                filtered_df['listed_in'].str.contains(genre, na=False, case=False)
            ]
        if director:
            filtered_df = filtered_df[
                filtered_df['director'].str.contains(director, na=False, case=False)
            ]
        if actor:
            filtered_df = filtered_df[
                filtered_df['cast'].str.contains(actor, na=False, case=False)
            ]
        if rating:
            filtered_df = filtered_df[
                filtered_df['rating'] == rating
            ]
        if release_year:  # Modify year filtering
            filtered_df = filtered_df[
                filtered_df['release_year'].astype(str).str.contains(str(release_year), na=False)
            ]
            
        # Return empty list if no results found
        if filtered_df.empty:
            return []
        
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
            print("[DEBUG] Ner match results:", results)  # This line is critical!!
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
