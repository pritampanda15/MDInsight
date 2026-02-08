"""
Trajectory Anomaly Detector - Find unusual events in MD trajectories.

Detects: conformational outliers, rare transient interactions, unusual
binding geometries, and trajectory artifacts using Isolation Forest,
LOF, and statistical methods.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict

import numpy as np

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.preprocessing import StandardScaler
except ImportError:
    raise ImportError("scikit-learn is required.")

logger = logging.getLogger(__name__)


@dataclass
class AnomalyResult:
    """Anomaly detection results."""
    anomaly_scores: np.ndarray = field(default_factory=lambda: np.array([]))
    anomaly_mask: np.ndarray = field(default_factory=lambda: np.array([]))
    anomaly_frames: List[int] = field(default_factory=list)
    n_anomalies: int = 0
    anomaly_fraction: float = 0.0
    rare_interactions: List[Dict] = field(default_factory=list)
    method: str = ""


class TrajectoryAnomalyDetector:
    """
    Multi-method anomaly detection for MD trajectories.

    Parameters
    ----------
    fingerprinter : InteractionFingerprinter, optional
        Interaction fingerprints.
    dynamics_result : DynamicsResult, optional
        Dynamics analysis data.
    contamination : float, default 0.05
        Expected fraction of anomalous frames.
    """

    def __init__(
        self,
        fingerprinter=None,
        dynamics_result=None,
        contamination: float = 0.05,
    ):
        self.fingerprinter = fingerprinter
        self.dynamics_result = dynamics_result
        self.contamination = contamination
        self._result: Optional[AnomalyResult] = None

    @property
    def result(self) -> AnomalyResult:
        if self._result is None:
            self.detect()
        return self._result

    def detect(self) -> AnomalyResult:
        """Run anomaly detection."""
        features = self._prepare_features()
        if features.shape[0] == 0:
            return AnomalyResult()

        scaler = StandardScaler()
        X = scaler.fit_transform(features)

        # Isolation Forest
        iso = IsolationForest(
            contamination=self.contamination, random_state=42, n_jobs=-1
        )
        iso_labels = iso.fit_predict(X)
        iso_scores = iso.score_samples(X)

        # Local Outlier Factor
        lof = LocalOutlierFactor(
            contamination=self.contamination, n_neighbors=20
        )
        lof_labels = lof.fit_predict(X)
        lof_scores = lof.negative_outlier_factor_

        # Combine: frame is anomalous if both methods agree
        combined_anomaly = (iso_labels == -1) & (lof_labels == -1)

        # Normalize scores
        iso_norm = (iso_scores - iso_scores.mean()) / (iso_scores.std() + 1e-8)
        lof_norm = (lof_scores - lof_scores.mean()) / (lof_scores.std() + 1e-8)
        combined_scores = (iso_norm + lof_norm) / 2

        anomaly_frames = list(np.where(combined_anomaly)[0])

        # Detect rare interactions
        rare = self._find_rare_interactions()

        self._result = AnomalyResult(
            anomaly_scores=combined_scores,
            anomaly_mask=combined_anomaly,
            anomaly_frames=anomaly_frames,
            n_anomalies=int(combined_anomaly.sum()),
            anomaly_fraction=combined_anomaly.mean(),
            rare_interactions=rare,
            method="IsolationForest+LOF",
        )

        logger.info(
            f"Anomalies: {self._result.n_anomalies} frames "
            f"({self._result.anomaly_fraction:.1%}), "
            f"{len(rare)} rare interactions"
        )

        return self._result

    def _prepare_features(self) -> np.ndarray:
        """Combine available features."""
        parts = []

        if self.fingerprinter:
            parts.append(self.fingerprinter.fingerprints.astype(float))

        if self.dynamics_result:
            if self.dynamics_result.pca and self.dynamics_result.pca.projections.size > 0:
                n_pc = min(5, self.dynamics_result.pca.projections.shape[1])
                parts.append(self.dynamics_result.pca.projections[:, :n_pc])

        if not parts:
            return np.array([])

        min_frames = min(p.shape[0] for p in parts)
        parts = [p[:min_frames] for p in parts]
        return np.hstack(parts)

    def _find_rare_interactions(self) -> List[Dict]:
        """Find interactions that occur in very few frames."""
        if not self.fingerprinter:
            return []

        fps = self.fingerprinter.fingerprints
        labels = self.fingerprinter.bit_labels
        n_frames = fps.shape[0]

        rare = []
        for i, label in enumerate(labels):
            freq = fps[:, i].sum() / n_frames
            if 0 < freq < 0.02:  # present but in < 2% of frames
                frames_present = list(np.where(fps[:, i])[0])
                rare.append({
                    "interaction": label,
                    "frequency": freq,
                    "n_frames": len(frames_present),
                    "frames": frames_present[:10],  # sample
                })

        return sorted(rare, key=lambda x: x["frequency"])
