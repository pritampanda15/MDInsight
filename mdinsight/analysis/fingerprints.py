"""
Interaction Fingerprinter - Binary interaction fingerprint vectors per frame.

Encodes the presence/absence of each interaction type for each binding site
residue as a fixed-length bit vector. This representation enables:
  - ML-based clustering of binding modes
  - Similarity analysis between frames
  - Temporal pattern detection
  - Comparison across different ligands/mutations
"""

import logging
from typing import Optional, Dict, List, Tuple
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)

INTERACTION_TYPES = [
    "hbond", "hydrophobic", "salt_bridge", "pi_stack", "water_bridge"
]


class InteractionFingerprinter:
    """
    Generate binary interaction fingerprint (IFP) vectors from trajectory data.

    Each fingerprint bit encodes: (residue_i, interaction_type_j) = 0 or 1.
    Vector length = n_binding_site_residues × n_interaction_types.

    Parameters
    ----------
    interaction_analyzer : InteractionAnalyzer
        Completed interaction analysis.
    system : MolecularSystem
        Molecular system for residue information.
    """

    def __init__(self, interaction_analyzer, system):
        self.analyzer = interaction_analyzer
        self.system = system
        self._fingerprints: Optional[np.ndarray] = None
        self._residue_labels: Optional[List[str]] = None
        self._bit_labels: Optional[List[str]] = None

    @property
    def fingerprints(self) -> np.ndarray:
        if self._fingerprints is None:
            self.compute()
        return self._fingerprints

    @property
    def bit_labels(self) -> List[str]:
        if self._bit_labels is None:
            self.compute()
        return self._bit_labels

    def compute(self) -> np.ndarray:
        """
        Compute interaction fingerprints for all analyzed frames.

        Returns
        -------
        np.ndarray
            Binary matrix of shape (n_frames, n_bits) where
            n_bits = n_residues × n_interaction_types.
        """
        profile = self.analyzer.profile
        events = profile.events

        # Determine all binding site residues
        all_residues = sorted(set(ev.protein_residue for ev in events))
        self._residue_labels = all_residues

        # Build bit labels
        self._bit_labels = []
        for res in all_residues:
            for itype in INTERACTION_TYPES:
                self._bit_labels.append(f"{res}:{itype}")

        n_bits = len(self._bit_labels)
        n_frames = profile.total_frames

        # Build residue-type → bit index mapping
        bit_index = {}
        for idx, label in enumerate(self._bit_labels):
            bit_index[label] = idx

        # Group events by frame
        frame_events = defaultdict(list)
        for ev in events:
            frame_events[ev.frame].append(ev)

        # Build fingerprint matrix
        fps = np.zeros((n_frames, n_bits), dtype=np.int8)

        frame_list = sorted(frame_events.keys())
        frame_to_idx = {f: i for i, f in enumerate(frame_list)}

        # Pad with all frame indices if some frames have no events
        for ev in events:
            key = f"{ev.protein_residue}:{ev.interaction_type}"
            if key in bit_index:
                if ev.frame in frame_to_idx:
                    fps[frame_to_idx[ev.frame], bit_index[key]] = 1

        self._fingerprints = fps

        logger.info(
            f"Fingerprints: {n_frames} frames × {n_bits} bits "
            f"({len(all_residues)} residues × {len(INTERACTION_TYPES)} types)"
        )

        return fps

    def tanimoto_similarity(self, fp1_idx: int, fp2_idx: int) -> float:
        """Tanimoto coefficient between two frames' fingerprints."""
        fps = self.fingerprints
        a = fps[fp1_idx]
        b = fps[fp2_idx]
        intersection = np.sum(a & b)
        union = np.sum(a | b)
        return intersection / union if union > 0 else 0.0

    def similarity_matrix(self) -> np.ndarray:
        """
        Compute pairwise Tanimoto similarity matrix between all frames.

        Returns (n_frames, n_frames) symmetric matrix.
        """
        fps = self.fingerprints
        n = fps.shape[0]

        # Vectorized Tanimoto
        # T(a,b) = sum(a&b) / sum(a|b)
        intersection = fps @ fps.T  # works for binary
        row_sums = fps.sum(axis=1)
        union = row_sums[:, None] + row_sums[None, :] - intersection
        union[union == 0] = 1

        sim = intersection / union
        return sim

    def temporal_autocorrelation(self, max_lag: Optional[int] = None) -> np.ndarray:
        """
        Compute fingerprint autocorrelation over time.

        Measures how quickly the interaction pattern decorrelates.
        """
        fps = self.fingerprints.astype(float)
        n_frames = fps.shape[0]
        max_lag = max_lag or min(n_frames // 2, 200)

        # Mean-centered
        mean_fp = fps.mean(axis=0)
        centered = fps - mean_fp

        # Variance
        var = np.sum(centered ** 2) / n_frames

        autocorr = np.zeros(max_lag)
        for lag in range(max_lag):
            corr = np.sum(centered[:n_frames - lag] * centered[lag:]) / (n_frames - lag)
            autocorr[lag] = corr / var if var > 0 else 0.0

        return autocorr

    def get_dominant_patterns(self, n_patterns: int = 5) -> List[Tuple[str, float]]:
        """
        Find the most frequently occurring complete fingerprint patterns.

        Returns list of (pattern_description, frequency) tuples.
        """
        fps = self.fingerprints
        # Convert each row to a tuple for hashing
        pattern_counts = defaultdict(int)
        for row in fps:
            key = tuple(row)
            pattern_counts[key] += 1

        total = fps.shape[0]
        sorted_patterns = sorted(pattern_counts.items(), key=lambda x: -x[1])

        results = []
        for pattern, count in sorted_patterns[:n_patterns]:
            freq = count / total
            # Describe which bits are on
            active_bits = [self._bit_labels[i] for i, b in enumerate(pattern) if b]
            desc = ", ".join(active_bits) if active_bits else "no interactions"
            results.append((desc, freq))

        return results

    def bit_importance(self) -> np.ndarray:
        """
        Compute the variance of each fingerprint bit across frames.

        High-variance bits indicate interactions that switch on/off frequently
        (potentially important for binding mode transitions).
        """
        return self.fingerprints.astype(float).var(axis=0)
