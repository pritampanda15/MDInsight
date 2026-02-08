"""
Conformational Clusterer - ML-based clustering of binding modes and states.

Uses multiple algorithms (HDBSCAN, GMM, K-means, spectral) on interaction
fingerprints, PCA projections, or combined features to identify distinct
conformational states and binding modes in the trajectory.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple

import numpy as np

try:
    from sklearn.cluster import KMeans, SpectralClustering, DBSCAN
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score, calinski_harabasz_score
except ImportError:
    raise ImportError("scikit-learn is required: pip install scikit-learn")

try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False

logger = logging.getLogger(__name__)


@dataclass
class ClusterResult:
    """Clustering results container."""
    labels: np.ndarray = field(default_factory=lambda: np.array([]))
    n_clusters: int = 0
    algorithm: str = ""
    silhouette_score: float = 0.0
    calinski_harabasz: float = 0.0
    cluster_centers: Optional[np.ndarray] = None
    cluster_populations: Dict[int, int] = field(default_factory=dict)
    cluster_frame_indices: Dict[int, List[int]] = field(default_factory=dict)
    transition_matrix: Optional[np.ndarray] = None
    features_used: str = ""


class ConformationalClusterer:
    """
    Multi-algorithm conformational state clustering.

    Automatically selects the best algorithm and number of clusters
    based on internal validation metrics.

    Parameters
    ----------
    fingerprinter : InteractionFingerprinter, optional
        Interaction fingerprints for clustering.
    dynamics_result : DynamicsResult, optional
        PCA projections and dynamics data.
    feature_mode : str, default "combined"
        Which features to use: "fingerprints", "pca", "combined".
    """

    def __init__(
        self,
        fingerprinter=None,
        dynamics_result=None,
        feature_mode: str = "combined",
    ):
        self.fingerprinter = fingerprinter
        self.dynamics_result = dynamics_result
        self.feature_mode = feature_mode
        self._results: Dict[str, ClusterResult] = {}
        self._best_result: Optional[ClusterResult] = None

    @property
    def best(self) -> ClusterResult:
        if self._best_result is None:
            raise RuntimeError("Run cluster() first.")
        return self._best_result

    def _prepare_features(self) -> np.ndarray:
        """Prepare feature matrix based on mode."""
        features_parts = []

        if self.feature_mode in ("fingerprints", "combined") and self.fingerprinter:
            fp = self.fingerprinter.fingerprints.astype(float)
            features_parts.append(fp)

        if self.feature_mode in ("pca", "combined") and self.dynamics_result:
            pca = self.dynamics_result.pca
            if pca and pca.projections.size > 0:
                # Use first 5 PCs
                n_pc = min(5, pca.projections.shape[1])
                features_parts.append(pca.projections[:, :n_pc])

        if not features_parts:
            raise ValueError("No features available for clustering.")

        # Align frame counts (take minimum)
        min_frames = min(f.shape[0] for f in features_parts)
        features_parts = [f[:min_frames] for f in features_parts]

        features = np.hstack(features_parts)

        # Standardize
        scaler = StandardScaler()
        features = scaler.fit_transform(features)

        logger.info(f"Feature matrix: {features.shape}")
        return features

    def cluster(
        self,
        algorithms: Optional[List[str]] = None,
        k_range: Tuple[int, int] = (2, 8),
    ) -> ClusterResult:
        """
        Run clustering with multiple algorithms, select best.

        Parameters
        ----------
        algorithms : list of str, optional
            Algorithms to try. Default: ["kmeans", "gmm", "hdbscan"].
        k_range : tuple
            Range of cluster numbers to try for parametric methods.

        Returns
        -------
        ClusterResult
            Best clustering result by silhouette score.
        """
        if algorithms is None:
            algorithms = ["kmeans", "gmm"]
            if HAS_HDBSCAN:
                algorithms.append("hdbscan")

        features = self._prepare_features()
        best_score = -1
        best_result = None

        for algo in algorithms:
            logger.info(f"Trying {algo}...")
            try:
                if algo == "kmeans":
                    result = self._run_kmeans(features, k_range)
                elif algo == "gmm":
                    result = self._run_gmm(features, k_range)
                elif algo == "hdbscan":
                    result = self._run_hdbscan(features)
                elif algo == "spectral":
                    result = self._run_spectral(features, k_range)
                elif algo == "dbscan":
                    result = self._run_dbscan(features)
                else:
                    logger.warning(f"Unknown algorithm: {algo}")
                    continue

                self._results[algo] = result

                if result.silhouette_score > best_score:
                    best_score = result.silhouette_score
                    best_result = result

                logger.info(
                    f"  {algo}: {result.n_clusters} clusters, "
                    f"silhouette={result.silhouette_score:.3f}"
                )
            except Exception as e:
                logger.warning(f"  {algo} failed: {e}")

        if best_result is None:
            raise RuntimeError("All clustering algorithms failed.")

        # Compute transition matrix
        best_result.transition_matrix = self._compute_transitions(best_result.labels)
        best_result.features_used = self.feature_mode

        self._best_result = best_result
        logger.info(
            f"Best: {best_result.algorithm} with {best_result.n_clusters} clusters "
            f"(silhouette={best_result.silhouette_score:.3f})"
        )
        return best_result

    def _run_kmeans(self, X: np.ndarray, k_range: Tuple[int, int]) -> ClusterResult:
        """K-means with optimal k selection."""
        best_score = -1
        best_labels = None
        best_k = 2
        best_centers = None

        for k in range(k_range[0], k_range[1] + 1):
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = km.fit_predict(X)
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(X, labels)
            if score > best_score:
                best_score = score
                best_labels = labels
                best_k = k
                best_centers = km.cluster_centers_

        return self._build_result(
            best_labels, "kmeans", best_score, X, best_centers
        )

    def _run_gmm(self, X: np.ndarray, k_range: Tuple[int, int]) -> ClusterResult:
        """Gaussian Mixture Model with BIC-based selection."""
        best_bic = np.inf
        best_labels = None
        best_k = 2
        best_score = -1

        for k in range(k_range[0], k_range[1] + 1):
            gmm = GaussianMixture(n_components=k, n_init=5, random_state=42)
            gmm.fit(X)
            bic = gmm.bic(X)
            labels = gmm.predict(X)

            if len(set(labels)) < 2:
                continue

            score = silhouette_score(X, labels)

            if bic < best_bic:
                best_bic = bic
                best_labels = labels
                best_k = k
                best_score = score

        return self._build_result(best_labels, "gmm", best_score, X)

    def _run_hdbscan(self, X: np.ndarray) -> ClusterResult:
        """HDBSCAN density-based clustering."""
        if not HAS_HDBSCAN:
            raise ImportError("hdbscan not installed")

        clusterer = hdbscan.HDBSCAN(min_cluster_size=max(5, X.shape[0] // 50))
        labels = clusterer.fit_predict(X)

        # Handle noise points (-1 labels)
        valid_mask = labels >= 0
        if valid_mask.sum() < 2 or len(set(labels[valid_mask])) < 2:
            return ClusterResult(labels=labels, n_clusters=0, algorithm="hdbscan")

        score = silhouette_score(X[valid_mask], labels[valid_mask])
        return self._build_result(labels, "hdbscan", score, X)

    def _run_spectral(self, X: np.ndarray, k_range: Tuple[int, int]) -> ClusterResult:
        """Spectral clustering."""
        best_score = -1
        best_labels = None

        # Limit size for spectral (O(n²) memory)
        if X.shape[0] > 5000:
            idx = np.random.choice(X.shape[0], 5000, replace=False)
            X_sub = X[idx]
        else:
            X_sub = X
            idx = np.arange(X.shape[0])

        for k in range(k_range[0], min(k_range[1] + 1, 6)):
            sc = SpectralClustering(n_clusters=k, random_state=42, n_init=5)
            labels = sc.fit_predict(X_sub)
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(X_sub, labels)
            if score > best_score:
                best_score = score
                best_labels = labels

        if best_labels is None:
            return ClusterResult(algorithm="spectral")

        # Map back to full dataset if subsampled
        full_labels = np.full(X.shape[0], -1, dtype=int)
        full_labels[idx] = best_labels

        return self._build_result(full_labels, "spectral", best_score, X)

    def _run_dbscan(self, X: np.ndarray) -> ClusterResult:
        """DBSCAN clustering."""
        # Auto-tune eps using k-distance graph
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=5)
        nn.fit(X)
        dists, _ = nn.kneighbors(X)
        eps = np.percentile(dists[:, -1], 90)

        db = DBSCAN(eps=eps, min_samples=5)
        labels = db.fit_predict(X)

        valid = labels >= 0
        if valid.sum() < 2 or len(set(labels[valid])) < 2:
            return ClusterResult(labels=labels, algorithm="dbscan")

        score = silhouette_score(X[valid], labels[valid])
        return self._build_result(labels, "dbscan", score, X)

    def _build_result(
        self, labels, algorithm, silhouette, X, centers=None
    ) -> ClusterResult:
        """Build ClusterResult from labels."""
        if labels is None:
            return ClusterResult(algorithm=algorithm)

        unique_labels = set(labels)
        unique_labels.discard(-1)  # remove noise
        n_clusters = len(unique_labels)

        populations = {}
        frame_indices = {}
        for c in unique_labels:
            mask = labels == c
            populations[c] = int(mask.sum())
            frame_indices[c] = list(np.where(mask)[0])

        ch_score = 0.0
        if n_clusters >= 2:
            valid = labels >= 0
            if valid.sum() > n_clusters:
                try:
                    ch_score = calinski_harabasz_score(X[valid], labels[valid])
                except Exception:
                    pass

        return ClusterResult(
            labels=labels,
            n_clusters=n_clusters,
            algorithm=algorithm,
            silhouette_score=silhouette,
            calinski_harabasz=ch_score,
            cluster_centers=centers,
            cluster_populations=populations,
            cluster_frame_indices=frame_indices,
        )

    def _compute_transitions(self, labels: np.ndarray) -> np.ndarray:
        """Compute state transition probability matrix."""
        unique = sorted(set(labels))
        if -1 in unique:
            unique.remove(-1)
        n_states = len(unique)

        if n_states < 2:
            return np.array([[1.0]])

        state_map = {s: i for i, s in enumerate(unique)}
        trans = np.zeros((n_states, n_states))

        for i in range(len(labels) - 1):
            if labels[i] < 0 or labels[i + 1] < 0:
                continue
            s1 = state_map[labels[i]]
            s2 = state_map[labels[i + 1]]
            trans[s1, s2] += 1

        # Normalize rows
        row_sums = trans.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        trans /= row_sums

        return trans
