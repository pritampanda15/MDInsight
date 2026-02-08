"""
Binding Feature Analyzer - Feature importance for binding mode differences.

Uses Random Forest, mutual information, and permutation importance to
determine which protein-ligand interactions are most discriminative
between conformational clusters.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

import numpy as np

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.model_selection import cross_val_score
except ImportError:
    raise ImportError("scikit-learn is required.")

logger = logging.getLogger(__name__)


@dataclass
class FeatureImportanceResult:
    """Feature importance analysis results."""
    feature_names: List[str] = field(default_factory=list)
    rf_importance: np.ndarray = field(default_factory=lambda: np.array([]))
    mutual_info: np.ndarray = field(default_factory=lambda: np.array([]))
    permutation_importance_mean: np.ndarray = field(default_factory=lambda: np.array([]))
    permutation_importance_std: np.ndarray = field(default_factory=lambda: np.array([]))
    combined_rank: np.ndarray = field(default_factory=lambda: np.array([]))
    cv_accuracy: float = 0.0
    top_features: List[Tuple[str, float]] = field(default_factory=list)


class BindingFeatureAnalyzer:
    """
    Identify which interactions drive conformational state differences.

    Trains classifiers to predict cluster labels from interaction fingerprints,
    then extracts feature importance to rank interaction contributions.

    Parameters
    ----------
    fingerprinter : InteractionFingerprinter
        Computed interaction fingerprints.
    cluster_result : ClusterResult
        Cluster labels for each frame.
    """

    def __init__(self, fingerprinter, cluster_result):
        self.fingerprinter = fingerprinter
        self.cluster_result = cluster_result
        self._result: Optional[FeatureImportanceResult] = None

    @property
    def result(self) -> FeatureImportanceResult:
        if self._result is None:
            self.analyze()
        return self._result

    def analyze(self) -> FeatureImportanceResult:
        """Run feature importance analysis."""
        fps = self.fingerprinter.fingerprints.astype(float)
        labels = self.cluster_result.labels
        feature_names = self.fingerprinter.bit_labels

        # Align dimensions
        n = min(fps.shape[0], len(labels))
        X = fps[:n]
        y = labels[:n]

        # Remove noise points
        valid = y >= 0
        X = X[valid]
        y = y[valid]

        if len(set(y)) < 2:
            logger.warning("Fewer than 2 classes — skipping feature importance.")
            return FeatureImportanceResult(feature_names=feature_names)

        logger.info(f"Feature importance: {X.shape[0]} samples, {X.shape[1]} features")

        # 1. Random Forest importance
        rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        rf_imp = rf.feature_importances_

        # Cross-validated accuracy
        cv_scores = cross_val_score(rf, X, y, cv=min(5, len(set(y))), scoring="accuracy")
        cv_acc = cv_scores.mean()
        logger.info(f"  RF CV accuracy: {cv_acc:.3f} ± {cv_scores.std():.3f}")

        # 2. Mutual information
        mi = mutual_info_classif(X, y, random_state=42)

        # 3. Permutation importance
        perm_result = permutation_importance(
            rf, X, y, n_repeats=10, random_state=42, n_jobs=-1
        )
        perm_mean = perm_result.importances_mean
        perm_std = perm_result.importances_std

        # 4. Combined ranking (average of normalized ranks)
        def rank_normalize(arr):
            ranks = np.argsort(np.argsort(-arr)).astype(float)
            return ranks / max(1, len(ranks) - 1)

        combined = (rank_normalize(rf_imp) + rank_normalize(mi) + rank_normalize(perm_mean)) / 3

        # Top features
        top_idx = np.argsort(combined)[:20]  # lower combined rank = more important
        top_features = [(feature_names[i], float(combined[i])) for i in top_idx]

        self._result = FeatureImportanceResult(
            feature_names=feature_names,
            rf_importance=rf_imp,
            mutual_info=mi,
            permutation_importance_mean=perm_mean,
            permutation_importance_std=perm_std,
            combined_rank=combined,
            cv_accuracy=cv_acc,
            top_features=top_features,
        )

        logger.info(f"  Top 5 discriminative features:")
        for name, rank in top_features[:5]:
            logger.info(f"    {name}: rank={rank:.3f}")

        return self._result

    def get_state_specific_interactions(self) -> Dict[int, List[Tuple[str, float]]]:
        """
        For each cluster, find interactions that are enriched compared to others.

        Returns {cluster_id: [(feature_name, enrichment_score)]}
        """
        fps = self.fingerprinter.fingerprints.astype(float)
        labels = self.cluster_result.labels
        feature_names = self.fingerprinter.bit_labels

        n = min(fps.shape[0], len(labels))
        fps = fps[:n]
        labels = labels[:n]

        result = {}
        unique_states = sorted(set(labels))
        unique_states = [s for s in unique_states if s >= 0]

        global_freq = fps.mean(axis=0)

        for state in unique_states:
            mask = labels == state
            state_freq = fps[mask].mean(axis=0)

            # Enrichment = state_freq / global_freq (with smoothing)
            enrichment = (state_freq + 0.01) / (global_freq + 0.01)

            # Sort by enrichment
            sorted_idx = np.argsort(-enrichment)
            state_features = [
                (feature_names[i], float(enrichment[i]))
                for i in sorted_idx[:10]
                if enrichment[i] > 1.2  # at least 20% enriched
            ]
            result[state] = state_features

        return result
