import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
from models.recommender import NetflixRecommender

# Initialize recommender
recommender = NetflixRecommender()

# Define evaluation ground truth for 5-query version
eval_5_queries = {
    "recommend_similar_content": {
        "The Matrix": ["The Matrix Reloaded", "The Matrix Revolutions", "Cloud Atlas"],
        "Inception": ["Interstellar", "The Prestige", "Tenet"],
        "The Irishman": ["GoodFellas", "Casino", "Raging Bull"],
        "Finding Nemo": ["Finding Dory", "Shark Tale", "Up"],
        "The Dark Knight": ["Batman Begins", "The Dark Knight Rises", "Joker"]
    },
    "recommend_by_actor": {
        "Tom Hanks": ["Forrest Gump", "Cast Away", "Saving Private Ryan"],
        "Scarlett Johansson": ["Lucy", "Marriage Story", "Black Widow"],
        "Brad Pitt": ["Fight Club", "Seven", "World War Z"],
        "Natalie Portman": ["Black Swan", "V for Vendetta", "Closer"],
        "Leonardo DiCaprio": ["Inception", "Shutter Island", "The Revenant"]
    },
    "recommend_by_ner": {
        "I want to watch something with Brad Pitt": ["Fight Club", "World War Z", "Seven"],
        "Show me some animated movies like Finding Nemo": ["Finding Dory", "Shark Tale", "Up"],
        "Any good movies directed by Christopher Nolan?": ["Inception", "The Prestige", "Tenet"],
        "Can you suggest something similar to The Irishman?": ["GoodFellas", "Casino", "The Godfather"],
        "Recommend something from Scarlett Johansson": ["Marriage Story", "Lucy", "Black Widow"]
    }
}


eval_10_queries = {
    "recommend_similar_content": {
        "The Matrix": ["The Matrix Reloaded", "The Matrix Revolutions", "Cloud Atlas"],
        "Inception": ["Interstellar", "The Prestige", "Tenet"],
        "The Irishman": ["GoodFellas", "Casino", "Raging Bull"],
        "Finding Nemo": ["Finding Dory", "Shark Tale", "Up"],
        "The Dark Knight": ["Batman Begins", "The Dark Knight Rises", "Joker"],
        "Titanic": ["The Notebook", "A Walk to Remember", "Dear John"],
        "Pulp Fiction": ["Reservoir Dogs", "Kill Bill: Vol. 1", "Django Unchained"],
        "The Social Network": ["The Great Hack", "The Big Short", "Jobs"],
        "Black Panther": ["Avengers: Infinity War", "Captain Marvel", "Doctor Strange"],
        "The Godfather": ["The Godfather Part II", "GoodFellas", "Scarface"]
    },
    "recommend_by_actor": {
        "Tom Hanks": ["Forrest Gump", "Cast Away", "Saving Private Ryan"],
        "Scarlett Johansson": ["Lucy", "Marriage Story", "Black Widow"],
        "Brad Pitt": ["Fight Club", "Seven", "World War Z"],
        "Natalie Portman": ["Black Swan", "V for Vendetta", "Closer"],
        "Leonardo DiCaprio": ["Inception", "Shutter Island", "The Revenant"],
        "Morgan Freeman": ["The Shawshank Redemption", "Se7en", "Bruce Almighty"],
        "Robert De Niro": ["The Irishman", "Taxi Driver", "Raging Bull"],
        "Will Smith": ["Men in Black", "I Am Legend", "The Pursuit of Happyness"],
        "Meryl Streep": ["The Devil Wears Prada", "Doubt", "Mamma Mia!"],
        "Keanu Reeves": ["The Matrix", "John Wick", "Constantine"]
    },
    "recommend_by_ner": {
        "I want to watch something with Brad Pitt": ["Fight Club", "World War Z", "Seven"],
        "Show me some animated movies like Finding Nemo": ["Finding Dory", "Shark Tale", "Up"],
        "Any good movies directed by Christopher Nolan?": ["Inception", "The Prestige", "Tenet"],
        "Can you suggest something similar to The Irishman?": ["GoodFellas", "Casino", "The Godfather"],
        "Recommend something from Scarlett Johansson": ["Marriage Story", "Lucy", "Black Widow"],
        "What should I watch if I liked Interstellar?": ["Inception", "Tenet", "Ad Astra"],
        "Movies about high school romance?": ["To All the Boys I’ve Loved Before", "The Kissing Booth", "Love, Simon"],
        "Suggest a good heist film": ["Ocean's Eleven", "Now You See Me", "Baby Driver"],
        "Find me something with Keanu Reeves": ["John Wick", "The Matrix", "Constantine"],
        "Any dramas featuring Meryl Streep?": ["The Devil Wears Prada", "Doubt", "August: Osage County"]
    }
}


# Evaluation function with F1
def evaluate_strategy(strategy_name: str, model_func: Callable, ground_truth: dict, top_k: int = 5):
    hit_count = 0
    total_expected = 0
    total_hits = 0
    total_returned = 0

    for query, expected in ground_truth.items():
        try:
            if strategy_name == "recommend_by_actor":
                recs = model_func(query, n=top_k)
            elif strategy_name == "recommend_by_ner":
                recs = model_func(query, n=top_k)
            else:
                recs = model_func(query, n_recommendations=top_k)

            rec_titles = [rec["title"] for rec in recs]
            hits = set(expected) & set(rec_titles)
            hit_count += 1 if hits else 0
            total_expected += len(expected)
            total_hits += len(hits)
            total_returned += len(rec_titles)
        except Exception as e:
            print(f"Error processing query '{query}': {e}")

    hit_rate = hit_count / len(ground_truth)
    precision = total_hits / total_returned if total_returned else 0
    recall = total_hits / total_expected if total_expected else 0
    f1 = (2 * precision * recall) / (precision + recall) if precision + recall else 0

    return hit_rate, precision, recall, f1

# Collect results
def collect_results(ground_truth_sets):
    strategies = {
        "recommend_similar_content": recommender.recommend_similar_content,
        "recommend_by_actor": recommender.recommend_by_actor,
        "recommend_by_ner": recommender.recommend_by_ner,
    }

    results = {}
    for strategy, func in strategies.items():
        gt = ground_truth_sets[strategy]
        metrics = evaluate_strategy(strategy, func, gt)
        results[strategy] = {
            "HitRate": round(metrics[0], 3),
            "Precision": round(metrics[1], 3),
            "Recall": round(metrics[2], 3),
            "F1": round(metrics[3], 3)
        }
    return pd.DataFrame(results).T.reset_index().rename(columns={"index": "Strategy"})

df_5 = collect_results(eval_5_queries)
df_10 = collect_results(eval_10_queries)

# Plot comparison (side-by-side bars)
fig, axes = plt.subplots(1, 4, figsize=(22, 5))
metrics = ["HitRate", "Precision", "Recall", "F1"]
x = np.arange(len(df_5["Strategy"]))
width = 0.35

for i, metric in enumerate(metrics):
    bars1 = axes[i].bar(x - width/2, df_5[metric], width, label="5 Queries", alpha=0.7)
    bars2 = axes[i].bar(x + width/2, df_10[metric], width, label="10 Queries", alpha=0.7)

    for bar in bars1:
        height = bar.get_height()
        axes[i].annotate(f'{height:.3f}',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3),
                         textcoords="offset points",
                         ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        axes[i].annotate(f'{height:.3f}',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3),
                         textcoords="offset points",
                         ha='center', va='bottom', fontsize=8)

    axes[i].set_title(metric)
    axes[i].set_xticks(x)
    axes[i].set_xticklabels(df_5["Strategy"], rotation=30)
    axes[i].set_ylim(0, 1)
    axes[i].set_ylabel("Score")

fig.suptitle("Evaluation Metrics Comparison: 5 vs. 10 Ground Truth Queries", fontsize=16)
fig.legend(loc="upper center", ncol=2)
fig.tight_layout(rect=[0, 0, 1, 0.93])
plt.show()
