import re
import spacy
import pandas as pd
from typing import Dict, List

class EntityExtractor:
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe
        self.nlp = spacy.load("en_core_web_sm")

        # Preprocess genre and title
        self.all_genres = set(g.strip().lower() for g in ','.join(self.df['listed_in'].dropna()).split(','))
        self.titles = set(self.df['title'].dropna().str.lower())

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        doc = self.nlp(text)
        entities = {
            "person": [],
            "genre": [],
            "year": [],
            "title": []
        }

        # Extract person names
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                entities["person"].append(ent.text)

        # Extract year (fix regex)
        entities["year"] = re.findall(r"\\\\b(?:19|20)\\\\d{2}\\\\b", text)

        # Genre detection from full dataset genres
        entities["genre"] = [
            genre for genre in self.all_genres if genre in text.lower()
        ]

        # Extra genre keyword alias mapping
        genre_keywords = {
            "horror": "horror movies",
            "comedy": "comedies",
            "romance": "romantic movies",
            "documentary": "documentaries",
            "thriller": "thrillers",
            "anime": "anime features",
            "drama": "dramas"
        }
        for keyword, genre in genre_keywords.items():
            if keyword in text.lower() and genre not in entities["genre"]:
                entities["genre"].append(genre)

        # Match known titles (case-insensitive)
        lower_text = text.lower()
        entities["title"] = [title for title in self.titles if title in lower_text]
        return entities