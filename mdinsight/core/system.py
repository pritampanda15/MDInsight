"""
Molecular System - High-level representation of a protein-ligand MD system.

Provides convenient abstractions for accessing protein, ligand, solvent,
binding site, and other subsystem selections.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Set

import numpy as np

try:
    import MDAnalysis as mda
    from MDAnalysis.analysis import distances
except ImportError:
    raise ImportError("MDAnalysis is required: pip install MDAnalysis")

logger = logging.getLogger(__name__)


@dataclass
class BindingSite:
    """Represents a protein binding site around a ligand."""
    residue_ids: List[int] = field(default_factory=list)
    residue_names: List[str] = field(default_factory=list)
    selection_string: str = ""
    centroid: Optional[np.ndarray] = None
    radius: float = 0.0


class MolecularSystem:
    """
    High-level molecular system with auto-detection of components.

    Parameters
    ----------
    universe : MDAnalysis.Universe
        Loaded MD universe.
    ligand_selection : str, optional
        MDAnalysis selection for the ligand. If None, auto-detection is attempted.
    binding_site_cutoff : float, default 5.0
        Distance cutoff (Å) for defining binding site residues.
    """

    def __init__(
        self,
        universe: mda.Universe,
        ligand_selection: Optional[str] = None,
        binding_site_cutoff: float = 5.0,
    ):
        self.universe = universe
        self._ligand_sel_str = ligand_selection
        self.binding_site_cutoff = binding_site_cutoff

        # Ensure elements are available (GRO files lack element info)
        self._ensure_elements()

        # Cached selections
        self._protein: Optional[mda.AtomGroup] = None
        self._ligand: Optional[mda.AtomGroup] = None
        self._solvent: Optional[mda.AtomGroup] = None
        self._binding_site: Optional[BindingSite] = None
        self._binding_site_atoms: Optional[mda.AtomGroup] = None

    @property
    def protein(self) -> mda.AtomGroup:
        if self._protein is None:
            self._protein = self.universe.select_atoms("protein")
            if len(self._protein) == 0:
                # Fallback: try backbone selection
                self._protein = self.universe.select_atoms("backbone")
            logger.info(f"Protein: {self._protein.n_atoms} atoms, "
                        f"{self._protein.residues.n_residues} residues")
        return self._protein

    @property
    def ligand(self) -> mda.AtomGroup:
        if self._ligand is None:
            if self._ligand_sel_str:
                self._ligand = self.universe.select_atoms(self._ligand_sel_str)
            else:
                self._ligand = self._auto_detect_ligand()
            if len(self._ligand) == 0:
                raise ValueError("No ligand atoms found. Specify ligand_selection explicitly.")
            logger.info(f"Ligand: {self._ligand.n_atoms} atoms, "
                        f"resname={set(self._ligand.resnames)}")
        return self._ligand

    @property
    def solvent(self) -> mda.AtomGroup:
        if self._solvent is None:
            solvent_sels = ["resname SOL WAT HOH TIP3 SPC T3P"]
            for sel in solvent_sels:
                self._solvent = self.universe.select_atoms(sel)
                if len(self._solvent) > 0:
                    break
            logger.info(f"Solvent: {self._solvent.n_atoms} atoms")
        return self._solvent

    @property
    def binding_site(self) -> BindingSite:
        """Dynamic binding site based on average ligand position across trajectory."""
        if self._binding_site is None:
            self._binding_site = self._compute_binding_site()
        return self._binding_site

    @property
    def binding_site_atoms(self) -> mda.AtomGroup:
        """Protein atoms within binding site cutoff of ligand."""
        if self._binding_site_atoms is None:
            bs = self.binding_site
            self._binding_site_atoms = self.universe.select_atoms(bs.selection_string)
        return self._binding_site_atoms

    def _auto_detect_ligand(self) -> mda.AtomGroup:
        """Heuristic ligand detection."""
        known_exclude = {
            "SOL", "WAT", "HOH", "TIP3", "SPC", "T3P", "T4P", "T5P",
            "NA", "CL", "K", "MG", "CA", "ZN", "FE", "NA+", "CL-",
            "DPPC", "POPC", "POPE", "DOPC", "CHOL",
        }
        protein_resnames = set(self.protein.residues.resnames)
        all_resnames = set(self.universe.residues.resnames)
        candidates = all_resnames - protein_resnames - known_exclude

        if not candidates:
            return self.universe.select_atoms("none")

        # Pick largest non-protein, non-solvent residue
        best_resname = max(
            candidates,
            key=lambda r: self.universe.select_atoms(
                f"resname {r} and not name H*"
            ).n_atoms,
        )
        logger.info(f"Auto-detected ligand: {best_resname}")
        return self.universe.select_atoms(f"resname {best_resname}")

    def _compute_binding_site(self) -> BindingSite:
        """
        Compute binding site by finding all protein residues that come within
        cutoff of the ligand across the trajectory.
        """
        lig = self.ligand
        prot = self.protein
        cutoff = self.binding_site_cutoff

        # Sample frames for binding site definition
        traj = self.universe.trajectory
        n_sample = min(50, traj.n_frames)
        sample_frames = np.linspace(0, traj.n_frames - 1, n_sample, dtype=int)

        contact_residues: Set[int] = set()

        for frame_idx in sample_frames:
            traj[frame_idx]
            # Distance-based contact detection
            dist_array = distances.distance_array(
                lig.positions, prot.positions, box=self.universe.dimensions
            )
            contact_mask = np.any(dist_array < cutoff, axis=0)
            contact_atoms = prot[contact_mask]
            contact_residues.update(contact_atoms.resids)

        # Build binding site object
        bs_resids = sorted(contact_residues)
        if not bs_resids:
            logger.warning("No binding site residues found!")
            return BindingSite()

        resid_str = " ".join(str(r) for r in bs_resids)
        sel_string = f"protein and resid {resid_str}"
        bs_atoms = self.universe.select_atoms(sel_string)

        # Compute centroid from first frame
        traj[0]
        centroid = lig.center_of_mass()

        bs = BindingSite(
            residue_ids=bs_resids,
            residue_names=list(set(bs_atoms.resnames)),
            selection_string=sel_string,
            centroid=centroid,
            radius=cutoff,
        )

        logger.info(
            f"Binding site: {len(bs_resids)} residues within {cutoff} Å of ligand"
        )
        return bs

    def get_protein_ligand_complex(self) -> mda.AtomGroup:
        """Return combined protein + ligand atom group."""
        return self.protein | self.ligand

    def get_binding_site_with_ligand(self) -> mda.AtomGroup:
        """Return binding site residues + ligand."""
        return self.binding_site_atoms | self.ligand

    def _ensure_elements(self):
        """Guess element attributes from atom names if not present (e.g. GRO files)."""
        if hasattr(self.universe.atoms, 'elements'):
            try:
                _ = self.universe.atoms.elements[0]
                return  # elements already available
            except Exception:
                pass
        logger.info("Guessing element types from atom names (topology lacks element info)")
        from MDAnalysis.topology.guessers import guess_types
        guessed = guess_types(self.universe.atoms.names)
        self.universe.add_TopologyAttr('elements', guessed)

    def system_summary(self) -> Dict:
        """Return system composition summary."""
        return {
            "n_protein_atoms": self.protein.n_atoms,
            "n_protein_residues": self.protein.residues.n_residues,
            "n_ligand_atoms": self.ligand.n_atoms,
            "ligand_resnames": list(set(self.ligand.resnames)),
            "n_solvent_atoms": self.solvent.n_atoms,
            "n_binding_site_residues": len(self.binding_site.residue_ids),
            "binding_site_cutoff": self.binding_site_cutoff,
            "total_atoms": self.universe.atoms.n_atoms,
        }
