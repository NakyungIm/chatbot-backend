import re
import spacy
import pandas as pd
from typing import Dict, List

class EntityExtractor:
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe
        self.nlp = spacy.load("en_core_web_sm")

        # pre-processing genre and title 
        self.genres = set(g.strip().lower() for g in ','.join(self.df['listed_in'].dropna()).split(','))
        self.titles = set(self.df['title'].dropna().str.lower())

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        doc = self.nlp(text)
        entities = {
            "person": [],
            "genre": [],
            "year": [],
            "title": []
        }

        # 1. person name
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                entities["person"].append(ent.text)

        # genre aliases
        self.genre_aliases = {
            "horror": "horror movies",
            "comedy": "comedies",
            "romance": "romantic movies",
            "documentary": "documentaries",
            "thriller": "thrillers",
            "anime": "anime features",
            "drama": "dramas"
        }

        entities["genre"] = [
            real_genre for keyword, real_genre in self.genre_aliases.items()
            if keyword in text.lower()
        ]

        # 3. year
        entities["year"] = re.findall(r"\b(19|20)\d{2}\b", text)

        # 4. title
        entities["title"] = [t for t in self.titles if t in text.lower()]

        return entities
