"""
Energy Decomposer - Per-residue interaction energy estimation.

Since true MM/PBSA or MM/GBSA requires force field parameters, this module
provides proxy energy estimates using empirical scoring functions based on
contact geometry and interaction type. Still highly informative for ranking
residue contributions to binding.
"""

import logging
from typing import Optional, Dict, List, Tuple
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)

# Empirical energy weights (kcal/mol approximate ranges from literature)
INTERACTION_ENERGIES = {
    "hbond": -3.0,           # -1 to -7 kcal/mol for H-bonds
    "salt_bridge": -4.0,     # -2 to -8 kcal/mol
    "hydrophobic": -0.7,     # -0.5 to -1.5 kcal/mol per contact
    "pi_stack": -2.5,        # -1 to -4 kcal/mol
    "water_bridge": -1.5,    # -0.5 to -3 kcal/mol
}

# Distance-dependent decay: E = E0 * exp(-α * (d - d0)²)
DECAY_PARAMS = {
    "hbond": {"d0": 2.8, "alpha": 0.5},
    "salt_bridge": {"d0": 3.0, "alpha": 0.3},
    "hydrophobic": {"d0": 3.8, "alpha": 0.2},
    "pi_stack": {"d0": 3.8, "alpha": 0.15},
    "water_bridge": {"d0": 3.0, "alpha": 0.4},
}


class EnergyDecomposer:
    """
    Proxy per-residue energy decomposition from interaction data.

    Uses empirical scoring functions weighted by interaction geometry
    to estimate each residue's energetic contribution to binding.

    Parameters
    ----------
    interaction_analyzer : InteractionAnalyzer
        Completed interaction analysis.
    custom_weights : dict, optional
        Override default interaction energy weights.
    """

    def __init__(self, interaction_analyzer, custom_weights: Optional[Dict] = None):
        self.analyzer = interaction_analyzer
        self.weights = {**INTERACTION_ENERGIES, **(custom_weights or {})}
        self._decomposition: Optional[Dict] = None

    @property
    def decomposition(self) -> Dict:
        if self._decomposition is None:
            self.compute()
        return self._decomposition

    def compute(self) -> Dict[str, Dict[str, float]]:
        """
        Compute per-residue energy decomposition.

        Returns
        -------
        dict
            {residue: {interaction_type: avg_energy, "total": total_avg_energy}}
        """
        profile = self.analyzer.profile
        n_frames = profile.total_frames

        # Accumulate energies per residue per type
        residue_energies = defaultdict(lambda: defaultdict(list))

        for ev in profile.events:
            itype = ev.interaction_type
            if itype not in self.weights:
                continue

            # Distance-dependent energy
            base_energy = self.weights[itype]
            params = DECAY_PARAMS.get(itype, {"d0": 3.0, "alpha": 0.3})
            d0, alpha = params["d0"], params["alpha"]

            energy = base_energy * np.exp(-alpha * (ev.distance - d0) ** 2)
            residue_energies[ev.protein_residue][itype].append(energy)

        # Average over frames
        decomposition = {}
        for residue, type_energies in residue_energies.items():
            res_result = {}
            total = 0.0
            for itype, energies in type_energies.items():
                avg = np.sum(energies) / n_frames  # average contribution per frame
                res_result[itype] = avg
                total += avg
            res_result["total"] = total
            decomposition[residue] = res_result

        self._decomposition = decomposition

        logger.info(f"Energy decomposition: {len(decomposition)} residues scored")
        return decomposition

    def get_hotspot_residues(self, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Get top contributing residues (most negative = strongest binding).

        Returns list of (residue, total_energy) sorted by energy.
        """
        decomp = self.decomposition
        ranked = sorted(
            [(res, data["total"]) for res, data in decomp.items()],
            key=lambda x: x[1],
        )
        return ranked[:top_n]

    def per_residue_breakdown(self) -> List[Dict]:
        """
        Structured breakdown for visualization/export.

        Returns list of dicts with residue, each interaction type energy, and total.
        """
        decomp = self.decomposition
        rows = []
        for residue, data in decomp.items():
            row = {"residue": residue}
            for itype in INTERACTION_ENERGIES:
                row[itype] = data.get(itype, 0.0)
            row["total"] = data.get("total", 0.0)
            rows.append(row)

        return sorted(rows, key=lambda x: x["total"])

    def energy_over_time(
        self, residue: str, window: int = 50
    ) -> Dict[str, np.ndarray]:
        """
        Compute rolling average energy contribution for a specific residue.

        Parameters
        ----------
        residue : str
            Residue identifier (e.g., "ARG-234").
        window : int
            Rolling average window size (in frames).

        Returns
        -------
        dict
            {interaction_type: np.ndarray of rolling energies}
        """
        profile = self.analyzer.profile
        n_frames = profile.total_frames

        # Bin energies per frame
        frame_energies = defaultdict(lambda: np.zeros(n_frames))

        for ev in profile.events:
            if ev.protein_residue != residue:
                continue
            itype = ev.interaction_type
            if itype not in self.weights:
                continue

            base_energy = self.weights[itype]
            params = DECAY_PARAMS.get(itype, {"d0": 3.0, "alpha": 0.3})
            energy = base_energy * np.exp(-params["alpha"] * (ev.distance - params["d0"]) ** 2)

            # Map frame to index (approximate)
            frame_idx = min(ev.frame, n_frames - 1)
            frame_energies[itype][frame_idx] += energy

        # Rolling average
        result = {}
        for itype, energies in frame_energies.items():
            if window > 1:
                kernel = np.ones(window) / window
                result[itype] = np.convolve(energies, kernel, mode="same")
            else:
                result[itype] = energies

        return result

    def compare_residues(self, residues: List[str]) -> Dict[str, Dict[str, float]]:
        """Compare energy contributions of specific residues."""
        decomp = self.decomposition
        return {res: decomp.get(res, {"total": 0.0}) for res in residues}
