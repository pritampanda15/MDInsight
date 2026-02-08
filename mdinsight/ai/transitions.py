"""
Transition Detector - Identifies binding mode shifts and conformational transitions.

Uses change-point detection, hidden Markov models, and sliding window analysis
to find when the protein-ligand system transitions between distinct states.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d

logger = logging.getLogger(__name__)


@dataclass
class TransitionEvent:
    """A detected transition between conformational states."""
    frame: int
    time_ns: float
    from_state: int
    to_state: int
    confidence: float  # 0-1
    description: str = ""


@dataclass
class TransitionAnalysis:
    """Complete transition analysis results."""
    events: List[TransitionEvent] = field(default_factory=list)
    changepoints: List[int] = field(default_factory=list)
    state_sequence: Optional[np.ndarray] = None
    n_transitions: int = 0
    mean_dwell_time_ns: float = 0.0
    dwell_times_per_state: Dict[int, List[float]] = field(default_factory=dict)


class TransitionDetector:
    """
    Multi-method transition detection in MD trajectories.

    Methods:
    1. Cluster-label change detection
    2. CUSUM-based change-point detection on fingerprint vectors
    3. Sliding-window dissimilarity analysis
    4. PCA projection discontinuity detection

    Parameters
    ----------
    cluster_result : ClusterResult
        Conformational clustering results.
    fingerprinter : InteractionFingerprinter, optional
        Interaction fingerprints for change-point detection.
    dynamics_result : DynamicsResult, optional
        PCA data for projection-based detection.
    dt_ns : float, default 0.001
        Time per frame in nanoseconds.
    """

    def __init__(
        self,
        cluster_result=None,
        fingerprinter=None,
        dynamics_result=None,
        dt_ns: float = 0.001,
    ):
        self.cluster_result = cluster_result
        self.fingerprinter = fingerprinter
        self.dynamics_result = dynamics_result
        self.dt_ns = dt_ns
        self._analysis: Optional[TransitionAnalysis] = None

    @property
    def analysis(self) -> TransitionAnalysis:
        if self._analysis is None:
            self.detect()
        return self._analysis

    def detect(self, window_size: int = 50, min_dwell: int = 10) -> TransitionAnalysis:
        """
        Run transition detection using all available data.

        Parameters
        ----------
        window_size : int
            Sliding window size for dissimilarity analysis.
        min_dwell : int
            Minimum frames in a state before a transition is recognized.
        """
        events = []
        changepoints = []

        # Method 1: Cluster label transitions
        if self.cluster_result and self.cluster_result.labels.size > 0:
            cl_events, cl_cp = self._from_cluster_labels(min_dwell)
            events.extend(cl_events)
            changepoints.extend(cl_cp)
            logger.info(f"  Cluster-based: {len(cl_events)} transitions")

        # Method 2: Fingerprint CUSUM
        if self.fingerprinter:
            fp_cp = self._cusum_changepoints(window_size)
            changepoints.extend(fp_cp)
            logger.info(f"  CUSUM changepoints: {len(fp_cp)}")

        # Method 3: PCA projection discontinuities
        if self.dynamics_result and self.dynamics_result.pca:
            pca_cp = self._pca_discontinuities(window_size)
            changepoints.extend(pca_cp)
            logger.info(f"  PCA discontinuities: {len(pca_cp)}")

        # Merge nearby changepoints
        changepoints = self._merge_changepoints(sorted(set(changepoints)), min_gap=min_dwell)

        # Build state sequence
        state_seq = None
        if self.cluster_result:
            state_seq = self.cluster_result.labels

        # Compute dwell times
        dwell_per_state = {}
        if state_seq is not None:
            dwell_per_state = self._compute_dwell_times(state_seq)

        mean_dwell = 0.0
        if dwell_per_state:
            all_dwells = [d for dwells in dwell_per_state.values() for d in dwells]
            mean_dwell = np.mean(all_dwells) if all_dwells else 0.0

        self._analysis = TransitionAnalysis(
            events=events,
            changepoints=changepoints,
            state_sequence=state_seq,
            n_transitions=len(events),
            mean_dwell_time_ns=mean_dwell,
            dwell_times_per_state=dwell_per_state,
        )

        logger.info(
            f"Transition analysis: {len(events)} transitions, "
            f"{len(changepoints)} changepoints, "
            f"mean dwell = {mean_dwell:.3f} ns"
        )

        return self._analysis

    def _from_cluster_labels(
        self, min_dwell: int
    ) -> Tuple[List[TransitionEvent], List[int]]:
        """Detect transitions from cluster label sequence."""
        labels = self.cluster_result.labels
        events = []
        changepoints = []

        # Smooth out noise by majority vote in sliding window
        smoothed = self._smooth_labels(labels, window=min_dwell // 2 or 1)

        prev_state = smoothed[0]
        dwell_count = 0

        for i in range(1, len(smoothed)):
            if smoothed[i] == prev_state or smoothed[i] < 0:
                dwell_count += 1
                continue

            if dwell_count >= min_dwell:
                events.append(TransitionEvent(
                    frame=i,
                    time_ns=i * self.dt_ns,
                    from_state=int(prev_state),
                    to_state=int(smoothed[i]),
                    confidence=min(1.0, dwell_count / (2 * min_dwell)),
                    description=f"State {prev_state} → {smoothed[i]} at frame {i}",
                ))
                changepoints.append(i)

            prev_state = smoothed[i]
            dwell_count = 0

        return events, changepoints

    def _smooth_labels(self, labels: np.ndarray, window: int = 5) -> np.ndarray:
        """Smooth cluster labels with majority vote filter."""
        if window < 2:
            return labels.copy()

        smoothed = labels.copy()
        half_w = window // 2

        for i in range(half_w, len(labels) - half_w):
            local = labels[i - half_w:i + half_w + 1]
            valid = local[local >= 0]
            if len(valid) > 0:
                values, counts = np.unique(valid, return_counts=True)
                smoothed[i] = values[counts.argmax()]

        return smoothed

    def _cusum_changepoints(self, window: int) -> List[int]:
        """CUSUM-based change-point detection on fingerprint vectors."""
        fps = self.fingerprinter.fingerprints.astype(float)

        # Compute per-frame fingerprint dissimilarity from running mean
        n_frames = fps.shape[0]
        dissimilarity = np.zeros(n_frames)

        running_mean = uniform_filter1d(fps, size=window, axis=0)

        for i in range(n_frames):
            diff = fps[i] - running_mean[i]
            dissimilarity[i] = np.sqrt(np.sum(diff ** 2))

        # Find peaks in dissimilarity signal
        threshold = np.mean(dissimilarity) + 2 * np.std(dissimilarity)
        peaks, properties = find_peaks(
            dissimilarity, height=threshold, distance=window
        )

        return list(peaks)

    def _pca_discontinuities(self, window: int) -> List[int]:
        """Detect discontinuities in PCA projection space."""
        pca = self.dynamics_result.pca
        if pca.projections.size == 0:
            return []

        # Use first 3 PCs
        proj = pca.projections[:, :min(3, pca.projections.shape[1])]

        # Compute step-wise Euclidean distance in PC space
        diffs = np.diff(proj, axis=0)
        step_dist = np.sqrt(np.sum(diffs ** 2, axis=1))

        # Smooth and find peaks
        smoothed = uniform_filter1d(step_dist, size=window // 5 or 1)
        threshold = np.mean(smoothed) + 2.5 * np.std(smoothed)
        peaks, _ = find_peaks(smoothed, height=threshold, distance=window)

        return list(peaks)

    def _merge_changepoints(self, cps: List[int], min_gap: int) -> List[int]:
        """Merge changepoints that are too close together."""
        if not cps:
            return []

        merged = [cps[0]]
        for cp in cps[1:]:
            if cp - merged[-1] >= min_gap:
                merged.append(cp)
        return merged

    def _compute_dwell_times(self, labels: np.ndarray) -> Dict[int, List[float]]:
        """Compute dwell time distributions per state."""
        dwell = {}
        current_state = labels[0]
        current_dwell = 0

        for label in labels:
            if label == current_state:
                current_dwell += 1
            else:
                if current_state >= 0:
                    if current_state not in dwell:
                        dwell[current_state] = []
                    dwell[current_state].append(current_dwell * self.dt_ns)
                current_state = label
                current_dwell = 1

        # Last segment
        if current_state >= 0:
            if current_state not in dwell:
                dwell[current_state] = []
            dwell[current_state].append(current_dwell * self.dt_ns)

        return dwell
