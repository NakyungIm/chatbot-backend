import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load dataset
df = pd.read_csv("netflix_titles.csv")
df = df.dropna(subset=["description"])

# 2. Text feature extraction
tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
X = tfidf.fit_transform(df["description"])

# 3. Apply KMeans clustering
k = 5  # or tune this later
model = KMeans(n_clusters=k, random_state=42)
labels = model.fit_predict(X)

df["cluster"] = labels

# 4. Evaluate clustering
sil_score = silhouette_score(X, labels)
print(f"Silhouette Score: {sil_score:.3f}")

# 5. Use PCA to reduce to 2D for plotting
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X.toarray())
df["pca_1"] = X_pca[:, 0]
df["pca_2"] = X_pca[:, 1]

# 6. Plot the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x="pca_1", y="pca_2", hue="cluster", data=df, palette="Set2")
plt.title("KMeans Clustering of Netflix Titles Based on Description")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title="Cluster")
plt.tight_layout()
plt.savefig("cluster_plot.png")
plt.show()

# 7. Optional: Show top titles in each cluster
for i in range(k):
    print(f"\n📍 Cluster {i} Top Titles:")
    cluster_titles = df[df["cluster"] == i]["title"].head(5).tolist()
    for title in cluster_titles:
        print("-", title)
