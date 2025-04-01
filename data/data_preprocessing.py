import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Tuple
import os

def load_and_clean_data(file_path: str = 'data/raw/netflix_titles.csv') -> pd.DataFrame:
    """
    Load and clean Netflix titles dataset
    
    Args:
        file_path (str): Path to the netflix_titles.csv file
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    # Load the dataset
    print("Loading dataset...")
    df = pd.read_csv(file_path)
    initial_shape = df.shape
    
    # Basic cleaning
    print("\nPerforming basic cleaning...")
    
    # Handle missing values
    df['description'] = df['description'].fillna('')
    df['director'] = df['director'].fillna('Unknown Director')
    df['cast'] = df['cast'].fillna('Unknown Cast')
    df['country'] = df['country'].fillna('Unknown Country')
    df['rating'] = df['rating'].fillna('Not Rated')
    
    # Clean text fields
    df['description'] = df['description'].str.strip()
    df['title'] = df['title'].str.strip()
    df['director'] = df['director'].str.strip()
    df['cast'] = df['cast'].str.strip()
    
    # Standardize text fields to lowercase
    df['description'] = df['description'].str.lower()
    df['director'] = df['director'].str.lower()
    df['cast'] = df['cast'].str.lower()
    df['country'] = df['country'].str.lower()
    df['listed_in'] = df['listed_in'].str.lower()
    
    # Clean country field (take first country if multiple)
    df['country'] = df['country'].str.split(',').str[0].str.strip()
    
    # Convert release_year to numeric
    df['release_year'] = pd.to_numeric(df['release_year'], errors='coerce')
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['title', 'director', 'release_year'])
    
    print(f"Removed {initial_shape[0] - df.shape[0]} duplicate entries")
    
    return df

def create_text_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, TfidfVectorizer, np.ndarray]:
    """
    Create text features using TF-IDF vectorization
    
    Args:
        df (pd.DataFrame): Cleaned DataFrame
    
    Returns:
        Tuple containing:
        - DataFrame with added features
        - Fitted TfidfVectorizer
        - TF-IDF matrix
    """
    print("\nCreating text features...")
    
    # Combine relevant text fields for content-based features
    df['content'] = (
        df['description'] + ' ' +
        df['cast'] + ' ' +
        df['director'] + ' ' +
        df['listed_in']
    )
    
    # Create TF-IDF features
    tfidf = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2)  # Include both unigrams and bigrams
    )
    
    tfidf_matrix = tfidf.fit_transform(df['content'])
    print(f"Created TF-IDF matrix with shape: {tfidf_matrix.shape}")
    
    return df, tfidf, tfidf_matrix

def create_additional_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create additional features for the dataset
    
    Args:
        df (pd.DataFrame): DataFrame with basic cleaning applied
    
    Returns:
        pd.DataFrame: DataFrame with additional features
    """
    print("\nCreating additional features...")
    
    # Create decade categories
    df['decade'] = (df['release_year'] // 10) * 10
    
    # Create age of content
    current_year = pd.Timestamp.now().year
    df['content_age'] = current_year - df['release_year']
    
    # Extract duration numeric value and unit
    def extract_number(x):
        num = ''.join(filter(str.isdigit, str(x)))
        return float(num) if num else np.nan
    
    def extract_unit(x):
        unit = ''.join(filter(str.isalpha, str(x)))
        return unit.strip() if unit else ''
    
    # Extract duration numeric value and unit with proper handling of empty values
    df['duration_num'] = df['duration'].apply(extract_number)
    df['duration_unit'] = df['duration'].apply(extract_unit)
    
    # Create genre list from listed_in column
    df['genres'] = df['listed_in'].apply(lambda x: str(x).split(','))
    
    # Count number of genres
    df['genre_count'] = df['genres'].apply(len)
    
    # Count cast members
    df['cast_count'] = df['cast'].str.count(',') + 1
    
    return df

def main():
    """Main function to execute the preprocessing pipeline"""
    # Create directories if they don't exist
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # Load and clean data
    df = load_and_clean_data()
    
    # Create text features
    df, tfidf, tfidf_matrix = create_text_features(df)
    
    # Create additional features
    df = create_additional_features(df)
    
    # Save processed data and TF-IDF artifacts
    print("\nSaving processed data and TF-IDF artifacts...")
    df.to_csv('data/processed/processed_netflix_titles.csv', index=False)
    
    # Save TF-IDF matrix and vectorizer
    from scipy.sparse import save_npz
    import joblib
    
    save_npz('data/processed/tfidf_matrix.npz', tfidf_matrix)
    joblib.dump(tfidf, 'data/processed/tfidf_vectorizer.joblib')
    
    print("\nFinal dataset shape:", df.shape)
    print("\nMissing values summary:")
    print(df.isnull().sum())
    
    return df, tfidf, tfidf_matrix

if __name__ == "__main__":
    df, tfidf, tfidf_matrix = main() 