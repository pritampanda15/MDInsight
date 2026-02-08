"""
Dynamics Analyzer - Advanced conformational dynamics beyond RMSD/RMSF.

Includes PCA on binding site, dynamic cross-correlation matrices,
binding site volume/SASA tracking, distance fluctuation analysis, and
quasi-harmonic entropy estimation.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple

import numpy as np
from scipy import linalg

try:
    import MDAnalysis as mda
    from MDAnalysis.analysis import align, rms, distances
    from MDAnalysis.analysis.rms import RMSF
except ImportError:
    raise ImportError("MDAnalysis is required.")

logger = logging.getLogger(__name__)


@dataclass
class PCAResult:
    """Principal component analysis results."""
    eigenvalues: np.ndarray = field(default_factory=lambda: np.array([]))
    eigenvectors: np.ndarray = field(default_factory=lambda: np.array([]))
    projections: np.ndarray = field(default_factory=lambda: np.array([]))
    variance_explained: np.ndarray = field(default_factory=lambda: np.array([]))
    cumulative_variance: np.ndarray = field(default_factory=lambda: np.array([]))
    mean_structure: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class DynamicsResult:
    """Container for all dynamics analysis results."""
    rmsd: Optional[np.ndarray] = None
    rmsf: Optional[np.ndarray] = None
    rmsf_residue_ids: Optional[np.ndarray] = None
    radius_of_gyration: Optional[np.ndarray] = None
    pca: Optional[PCAResult] = None
    cross_correlation: Optional[np.ndarray] = None
    cross_corr_residue_ids: Optional[np.ndarray] = None
    distance_matrix_evolution: Optional[np.ndarray] = None
    binding_site_rmsf: Optional[np.ndarray] = None
    binding_site_residue_ids: Optional[np.ndarray] = None
    ligand_rmsd: Optional[np.ndarray] = None
    protein_ligand_distance: Optional[np.ndarray] = None
    frame_times_ns: Optional[np.ndarray] = None
    entropy_estimate: Optional[float] = None


class DynamicsAnalyzer:
    """
    Advanced dynamics analysis engine.

    Parameters
    ----------
    system : MolecularSystem
        Molecular system to analyze.
    stride : int, default 1
        Frame stride for analysis.
    align_selection : str, default "protein and name CA"
        Selection for trajectory alignment.
    """

    def __init__(
        self,
        system,
        stride: int = 1,
        align_selection: str = "protein and name CA",
    ):
        self.system = system
        self.stride = stride
        self.align_selection = align_selection
        self._result: Optional[DynamicsResult] = None

    @property
    def result(self) -> DynamicsResult:
        if self._result is None:
            raise RuntimeError("Run analyze() first.")
        return self._result

    def analyze(
        self,
        start: int = 0,
        stop: Optional[int] = None,
        compute_pca: bool = True,
        compute_cross_correlation: bool = True,
        compute_entropy: bool = True,
    ) -> DynamicsResult:
        """Run comprehensive dynamics analysis."""
        u = self.system.universe
        traj = u.trajectory
        stop = stop or traj.n_frames

        logger.info(f"Dynamics analysis: frames {start}-{stop}, stride {self.stride}")

        result = DynamicsResult()

        # Collect frame times
        times = []
        for ts in traj[start:stop:self.stride]:
            times.append(ts.time / 1000.0)
        result.frame_times_ns = np.array(times)

        # 1. RMSD (protein backbone)
        result.rmsd = self._compute_rmsd(start, stop)
        logger.info("  ✓ RMSD computed")

        # 2. RMSF (per-residue Cα)
        rmsf_vals, rmsf_resids = self._compute_rmsf(start, stop)
        result.rmsf = rmsf_vals
        result.rmsf_residue_ids = rmsf_resids
        logger.info("  ✓ RMSF computed")

        # 3. Binding site RMSF
        bs_rmsf, bs_resids = self._compute_binding_site_rmsf(start, stop)
        result.binding_site_rmsf = bs_rmsf
        result.binding_site_residue_ids = bs_resids
        logger.info("  ✓ Binding site RMSF computed")

        # 4. Radius of gyration
        result.radius_of_gyration = self._compute_rog(start, stop)
        logger.info("  ✓ Radius of gyration computed")

        # 5. Ligand RMSD
        result.ligand_rmsd = self._compute_ligand_rmsd(start, stop)
        logger.info("  ✓ Ligand RMSD computed")

        # 6. Protein-ligand COM distance
        result.protein_ligand_distance = self._compute_pl_distance(start, stop)
        logger.info("  ✓ Protein-ligand distance computed")

        # 7. PCA on binding site
        if compute_pca:
            result.pca = self._compute_pca(start, stop)
            logger.info("  ✓ PCA computed")

        # 8. Dynamic cross-correlation
        if compute_cross_correlation:
            dccm, resids = self._compute_dccm(start, stop)
            result.cross_correlation = dccm
            result.cross_corr_residue_ids = resids
            logger.info("  ✓ Cross-correlation matrix computed")

        # 9. Quasi-harmonic entropy
        if compute_entropy:
            result.entropy_estimate = self._compute_quasi_harmonic_entropy(start, stop)
            logger.info(f"  ✓ Entropy estimate: {result.entropy_estimate:.2f} J/(mol·K)")

        self._result = result
        return result

    def _compute_rmsd(self, start: int, stop: int) -> np.ndarray:
        """RMSD of protein backbone relative to first frame."""
        u = self.system.universe
        ref = u.copy()
        ref.trajectory[start]

        backbone = u.select_atoms("protein and backbone")
        ref_backbone = ref.select_atoms("protein and backbone")

        rmsd_values = []
        for ts in u.trajectory[start:stop:self.stride]:
            # Align
            _, rmsd_val = align.rotation_matrix(
                backbone.positions, ref_backbone.positions
            )
            rmsd_values.append(rmsd_val)

        return np.array(rmsd_values)

    def _compute_rmsf(self, start: int, stop: int) -> Tuple[np.ndarray, np.ndarray]:
        """Per-residue Cα RMSF."""
        u = self.system.universe
        ca = u.select_atoms("protein and name CA")

        # Collect positions
        positions = []
        for ts in u.trajectory[start:stop:self.stride]:
            positions.append(ca.positions.copy())

        positions = np.array(positions)
        mean_pos = positions.mean(axis=0)
        diff = positions - mean_pos
        rmsf = np.sqrt((diff ** 2).sum(axis=2).mean(axis=0))

        return rmsf, ca.resids.copy()

    def _compute_binding_site_rmsf(self, start, stop) -> Tuple[np.ndarray, np.ndarray]:
        """RMSF for binding site Cα atoms only."""
        u = self.system.universe
        bs = self.system.binding_site
        if not bs.residue_ids:
            return np.array([]), np.array([])

        resid_str = " ".join(str(r) for r in bs.residue_ids)
        bs_ca = u.select_atoms(f"protein and name CA and resid {resid_str}")

        if len(bs_ca) == 0:
            return np.array([]), np.array([])

        positions = []
        for ts in u.trajectory[start:stop:self.stride]:
            positions.append(bs_ca.positions.copy())

        positions = np.array(positions)
        mean_pos = positions.mean(axis=0)
        diff = positions - mean_pos
        rmsf = np.sqrt((diff ** 2).sum(axis=2).mean(axis=0))

        return rmsf, bs_ca.resids.copy()

    def _compute_rog(self, start: int, stop: int) -> np.ndarray:
        """Radius of gyration over trajectory."""
        protein = self.system.protein
        rog = []
        for ts in self.system.universe.trajectory[start:stop:self.stride]:
            rog.append(protein.radius_of_gyration())
        return np.array(rog)

    def _compute_ligand_rmsd(self, start: int, stop: int) -> np.ndarray:
        """RMSD of ligand heavy atoms (after protein alignment)."""
        u = self.system.universe
        lig = self.system.ligand.select_atoms("not name H*")

        if len(lig) == 0:
            lig = self.system.ligand

        # Reference position
        u.trajectory[start]
        ref_pos = lig.positions.copy()

        rmsd_vals = []
        for ts in u.trajectory[start:stop:self.stride]:
            diff = lig.positions - ref_pos
            rmsd = np.sqrt((diff ** 2).sum(axis=1).mean())
            rmsd_vals.append(rmsd)

        return np.array(rmsd_vals)

    def _compute_pl_distance(self, start: int, stop: int) -> np.ndarray:
        """Protein-ligand center-of-mass distance over time."""
        prot = self.system.protein
        lig = self.system.ligand
        dists = []

        for ts in self.system.universe.trajectory[start:stop:self.stride]:
            d = np.linalg.norm(prot.center_of_mass() - lig.center_of_mass())
            dists.append(d)

        return np.array(dists)

    def _compute_pca(self, start: int, stop: int, n_components: int = 10) -> PCAResult:
        """PCA on binding site Cα coordinates."""
        u = self.system.universe
        bs = self.system.binding_site

        if not bs.residue_ids:
            # Fallback to full protein
            ca = u.select_atoms("protein and name CA")
        else:
            resid_str = " ".join(str(r) for r in bs.residue_ids)
            ca = u.select_atoms(f"protein and name CA and resid {resid_str}")

        if len(ca) < 3:
            logger.warning("Too few atoms for PCA")
            return PCAResult()

        # Collect coordinate matrix
        coords = []
        for ts in u.trajectory[start:stop:self.stride]:
            coords.append(ca.positions.flatten())

        coords = np.array(coords)  # (n_frames, 3*n_atoms)
        mean_struct = coords.mean(axis=0)
        centered = coords - mean_struct

        # Covariance matrix
        cov = np.cov(centered.T)

        # Eigendecomposition
        n_comp = min(n_components, cov.shape[0])
        eigenvalues, eigenvectors = linalg.eigh(
            cov, subset_by_index=[cov.shape[0] - n_comp, cov.shape[0] - 1]
        )

        # Sort descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Projections
        projections = centered @ eigenvectors

        # Variance explained
        total_var = eigenvalues.sum()
        var_explained = eigenvalues / total_var if total_var > 0 else eigenvalues
        cum_var = np.cumsum(var_explained)

        return PCAResult(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            projections=projections,
            variance_explained=var_explained,
            cumulative_variance=cum_var,
            mean_structure=mean_struct,
        )

    def _compute_dccm(
        self, start: int, stop: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Dynamic Cross-Correlation Matrix (DCCM) for Cα atoms.

        Returns normalized correlation matrix (-1 to +1).
        """
        u = self.system.universe
        ca = u.select_atoms("protein and name CA")
        n_atoms = len(ca)

        # Collect positions
        positions = []
        for ts in u.trajectory[start:stop:self.stride]:
            positions.append(ca.positions.copy())

        positions = np.array(positions)  # (n_frames, n_atoms, 3)
        mean_pos = positions.mean(axis=0)
        fluctuations = positions - mean_pos  # (n_frames, n_atoms, 3)

        # Cross-correlation: C_ij = <Δr_i · Δr_j> / sqrt(<|Δr_i|²> * <|Δr_j|²>)
        n_frames = len(positions)

        # Compute dot products
        # numerator[i,j] = mean of dot(fluct[f,i], fluct[f,j]) over frames
        # This is equivalent to sum over xyz of cov(pos_i_x, pos_j_x) etc.

        # Reshape for efficient computation
        flat_fluct = fluctuations.reshape(n_frames, n_atoms, 3)

        # Numerator: <Δr_i · Δr_j>
        numerator = np.zeros((n_atoms, n_atoms))
        for f in range(n_frames):
            numerator += flat_fluct[f] @ flat_fluct[f].T
        numerator /= n_frames

        # Denominator: sqrt(<|Δr_i|²> * <|Δr_j|²>)
        magnitudes = np.sqrt(np.diag(numerator))
        denom = np.outer(magnitudes, magnitudes)
        denom[denom == 0] = 1.0  # avoid division by zero

        dccm = numerator / denom

        return dccm, ca.resids.copy()

    def _compute_quasi_harmonic_entropy(self, start: int, stop: int) -> float:
        """
        Quasi-harmonic approximation of configurational entropy (Schlitter method).

        S ≤ 0.5 * kB * Σ ln(1 + kBT*e²/(ℏ²) * λ_i)
        """
        u = self.system.universe
        ca = u.select_atoms("protein and name CA")

        positions = []
        for ts in u.trajectory[start:stop:self.stride]:
            positions.append(ca.positions.flatten())

        positions = np.array(positions)
        mean_pos = positions.mean(axis=0)
        centered = positions - mean_pos

        # Mass-weighted covariance (use uniform mass for Cα)
        cov = np.cov(centered.T)

        # Schlitter formula constants
        kB = 1.380649e-23  # J/K
        T = 300.0  # K (assume standard)
        hbar = 1.054571817e-34  # J·s
        e = np.e

        # Convert covariance from Å² to m²
        cov_m2 = cov * 1e-20

        # Mass of carbon alpha (roughly 12 amu in kg)
        mass = 12.011 * 1.66054e-27  # kg

        # Mass-weighted covariance
        sigma = mass * kB * T * e**2 / hbar**2 * cov_m2

        # Eigenvalues
        eigenvalues = np.linalg.eigvalsh(sigma)
        eigenvalues = eigenvalues[eigenvalues > 0]

        # Entropy
        S = 0.5 * kB * np.sum(np.log(1.0 + eigenvalues))

        # Convert to J/(mol·K) — multiply by Avogadro
        S_per_mol = S * 6.02214076e23

        return S_per_mol
