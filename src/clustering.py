"""Segmentación de clientes con K-Means sobre variables RFM."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from src.features import PROCESSED

RFM_COLS = ["Recency", "Frequency", "Monetary"]
RANDOM_STATE = 42


def prepare_matrix(feats: pd.DataFrame, cols=RFM_COLS):
    """log1p (reduce el sesgo de F y M) + estandarización (media 0, desv. 1)."""
    X = np.log1p(feats[cols].astype(float))
    scaler = StandardScaler()
    return scaler.fit_transform(X), scaler


def evaluate_k(X, k_range=range(2, 11), sample_size=6000):
    """Inercia (codo) y métricas de validación interna para cada k."""
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X)
        rows.append({
            "k": k,
            "inertia": km.inertia_,
            "silhouette": silhouette_score(X, km.labels_, sample_size=sample_size,
                                           random_state=RANDOM_STATE),
            "calinski_harabasz": calinski_harabasz_score(X, km.labels_),
            "davies_bouldin": davies_bouldin_score(X, km.labels_),
        })
    return pd.DataFrame(rows)


def fit_kmeans(X, k):
    return KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE).fit(X)


def pca_2d(X):
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    return pca.fit_transform(X), pca


def name_segments(feats: pd.DataFrame, label_col="Cluster") -> dict:
    """Nombra cada cluster según su perfil medio relativo al promedio global.

    Puntaje = -z(Recency) + z(Frequency) + z(Monetary). El cluster con mayor
    puntaje son los "Campeones"; el resto se nombra según si compró hace poco
    (Recency) y si su ticket promedio es alto (AvgTicket).
    """
    cols = RFM_COLS + ["AvgTicket"]
    prof = feats.groupby(label_col)[cols].mean()
    z = (prof - feats[cols].mean()) / feats[cols].std()
    score = -z["Recency"] + z["Frequency"] + z["Monetary"]
    order = score.sort_values(ascending=False).index.tolist()

    names = {}
    for rank, c in enumerate(order):
        recent = z.loc[c, "Recency"] < -0.25
        high_ticket = z.loc[c, "AvgTicket"] > 0
        if rank == 0:
            names[c] = "Campeones"
        elif high_ticket and not recent:
            names[c] = "Alto valor en riesgo"
        elif high_ticket:
            names[c] = "Leales recientes"
        elif recent:
            names[c] = "Nuevos"
        else:
            names[c] = "Bajo valor inactivos"
    # Evita nombres duplicados
    seen = {}
    for c in order:
        n = names[c]
        seen[n] = seen.get(n, 0) + 1
        if seen[n] > 1:
            names[c] = f"{n} {seen[n]}"
    return names


def run(k: int = 4) -> pd.DataFrame:
    feats = pd.read_parquet(PROCESSED / "customer_features.parquet")
    X, _ = prepare_matrix(feats)
    km = fit_kmeans(X, k)
    feats["Cluster"] = km.labels_
    feats["Segment"] = feats["Cluster"].map(name_segments(feats))
    coords, _ = pca_2d(X)
    feats["PC1"], feats["PC2"] = coords[:, 0], coords[:, 1]
    feats.to_parquet(PROCESSED / "customer_segments.parquet", index=False)
    return feats


if __name__ == "__main__":
    out = run()
    print(out.groupby("Segment")[RFM_COLS].agg(["mean", "count"]).round(1))
