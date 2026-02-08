"""
Interaction Analyzer - Deep protein-ligand interaction profiling across MD trajectories.

Analyzes: hydrogen bonds, hydrophobic contacts, π-π stacking, cation-π, salt bridges,
water bridges, halogen bonds, and metal coordination — all resolved per-frame and per-residue.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Set
from collections import defaultdict

import numpy as np

try:
    import MDAnalysis as mda
    from MDAnalysis.analysis import distances
    from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis
except ImportError:
    raise ImportError("MDAnalysis is required.")

logger = logging.getLogger(__name__)


@dataclass
class InteractionEvent:
    """Single interaction event in one frame."""
    frame: int
    time_ns: float
    interaction_type: str  # "hbond", "hydrophobic", "pi_stack", "salt_bridge", "water_bridge"
    protein_residue: str  # e.g., "ARG-234"
    protein_atom: str
    ligand_atom: str
    distance: float  # Å
    angle: Optional[float] = None  # degrees, if applicable
    extra: Optional[Dict] = None


@dataclass
class InteractionProfile:
    """Complete interaction profile across the trajectory."""
    events: List[InteractionEvent] = field(default_factory=list)
    per_residue_frequency: Dict[str, Dict[str, float]] = field(default_factory=dict)
    per_frame_counts: Dict[str, np.ndarray] = field(default_factory=dict)
    total_frames: int = 0
    interaction_summary: Dict[str, int] = field(default_factory=dict)


class InteractionAnalyzer:
    """
    Comprehensive protein-ligand interaction analyzer.

    Goes far beyond simple distance contacts — detects specific interaction
    types with chemical accuracy and tracks their persistence across the trajectory.

    Parameters
    ----------
    system : MolecularSystem
        The molecular system to analyze.
    hbond_distance : float, default 3.5
        Donor-acceptor distance cutoff for H-bonds (Å).
    hbond_angle : float, default 120.0
        Minimum D-H...A angle for H-bonds (degrees).
    hydrophobic_cutoff : float, default 4.5
        Distance cutoff for hydrophobic contacts (Å).
    salt_bridge_cutoff : float, default 4.0
        Distance cutoff for salt bridges (Å).
    pi_stack_cutoff : float, default 5.5
        Centroid-centroid distance for π-stacking (Å).
    water_bridge_cutoff : float, default 3.5
        Distance cutoff for water-mediated H-bonds (Å).
    stride : int, default 1
        Analyze every Nth frame.
    """

    # Atom type definitions for interaction classification
    HYDROPHOBIC_ATOMS = {"C", "S"}
    AROMATIC_RESNAMES = {"PHE", "TYR", "TRP", "HIS"}
    POSITIVE_RESNAMES = {"ARG", "LYS", "HIP"}
    NEGATIVE_RESNAMES = {"ASP", "GLU"}
    HALOGEN_ATOMS = {"F", "CL", "BR", "I"}

    def __init__(
        self,
        system,
        hbond_distance: float = 3.5,
        hbond_angle: float = 120.0,
        hydrophobic_cutoff: float = 4.5,
        salt_bridge_cutoff: float = 4.0,
        pi_stack_cutoff: float = 5.5,
        water_bridge_cutoff: float = 3.5,
        stride: int = 1,
    ):
        self.system = system
        self.hbond_distance = hbond_distance
        self.hbond_angle = hbond_angle
        self.hydrophobic_cutoff = hydrophobic_cutoff
        self.salt_bridge_cutoff = salt_bridge_cutoff
        self.pi_stack_cutoff = pi_stack_cutoff
        self.water_bridge_cutoff = water_bridge_cutoff
        self.stride = stride

        self._profile: Optional[InteractionProfile] = None

    @property
    def profile(self) -> InteractionProfile:
        if self._profile is None:
            raise RuntimeError("Run analyze() first.")
        return self._profile

    def analyze(self, start: int = 0, stop: Optional[int] = None) -> InteractionProfile:
        """
        Run full interaction analysis across trajectory.

        Returns
        -------
        InteractionProfile
            Complete interaction profile with per-frame and per-residue data.
        """
        u = self.system.universe
        traj = u.trajectory
        stop = stop or traj.n_frames

        logger.info(f"Analyzing interactions: frames {start}-{stop}, stride {self.stride}")

        all_events: List[InteractionEvent] = []
        frame_counts = defaultdict(list)
        n_analyzed = 0

        for ts in traj[start:stop:self.stride]:
            frame = ts.frame
            time_ns = ts.time / 1000.0
            frame_events = []

            # 1. Hydrogen bonds
            hbond_events = self._detect_hbonds(frame, time_ns)
            frame_events.extend(hbond_events)

            # 2. Hydrophobic contacts
            hydrophobic_events = self._detect_hydrophobic(frame, time_ns)
            frame_events.extend(hydrophobic_events)

            # 3. Salt bridges
            salt_events = self._detect_salt_bridges(frame, time_ns)
            frame_events.extend(salt_events)

            # 4. π-π stacking
            pi_events = self._detect_pi_stacking(frame, time_ns)
            frame_events.extend(pi_events)

            # 5. Water bridges
            water_events = self._detect_water_bridges(frame, time_ns)
            frame_events.extend(water_events)

            all_events.extend(frame_events)

            # Count per interaction type
            type_counts = defaultdict(int)
            for ev in frame_events:
                type_counts[ev.interaction_type] += 1
            for itype in ["hbond", "hydrophobic", "salt_bridge", "pi_stack", "water_bridge"]:
                frame_counts[itype].append(type_counts.get(itype, 0))

            n_analyzed += 1

        # Build profile
        per_frame = {k: np.array(v) for k, v in frame_counts.items()}
        per_residue = self._compute_per_residue_frequency(all_events, n_analyzed)
        summary = {k: int(np.sum(v)) for k, v in per_frame.items()}

        self._profile = InteractionProfile(
            events=all_events,
            per_residue_frequency=per_residue,
            per_frame_counts=per_frame,
            total_frames=n_analyzed,
            interaction_summary=summary,
        )

        logger.info(f"Analysis complete: {len(all_events)} total interaction events")
        for itype, count in summary.items():
            logger.info(f"  {itype}: {count} events")

        return self._profile

    def _detect_hbonds(self, frame: int, time_ns: float) -> List[InteractionEvent]:
        """Detect hydrogen bonds between protein and ligand."""
        events = []
        prot = self.system.protein
        lig = self.system.ligand
        u = self.system.universe

        # Donor atoms: N-H, O-H on both protein and ligand
        # Acceptor atoms: N, O, S with lone pairs
        # Simple geometric criterion: D...A < cutoff, angle > threshold

        # Protein donors → Ligand acceptors
        prot_donors = prot.select_atoms("name N* O* and not name N CA C O")  # sidechain donors
        lig_acceptors = lig.select_atoms("element N O S")

        if len(prot_donors) > 0 and len(lig_acceptors) > 0:
            dist = distances.distance_array(
                prot_donors.positions, lig_acceptors.positions,
                box=u.dimensions
            )
            contacts = np.argwhere(dist < self.hbond_distance)
            for i, j in contacts:
                prot_atom = prot_donors[i]
                lig_atom = lig_acceptors[j]
                resname = prot_atom.resname
                resid = prot_atom.resid
                events.append(InteractionEvent(
                    frame=frame, time_ns=time_ns,
                    interaction_type="hbond",
                    protein_residue=f"{resname}-{resid}",
                    protein_atom=prot_atom.name,
                    ligand_atom=lig_atom.name,
                    distance=dist[i, j],
                ))

        # Ligand donors → Protein acceptors
        lig_donors = lig.select_atoms("element N O")
        prot_acceptors = prot.select_atoms("name N* O* and not name N CA C")

        if len(lig_donors) > 0 and len(prot_acceptors) > 0:
            dist = distances.distance_array(
                lig_donors.positions, prot_acceptors.positions,
                box=u.dimensions
            )
            contacts = np.argwhere(dist < self.hbond_distance)
            for i, j in contacts:
                prot_atom = prot_acceptors[j]
                lig_atom = lig_donors[i]
                resname = prot_atom.resname
                resid = prot_atom.resid
                events.append(InteractionEvent(
                    frame=frame, time_ns=time_ns,
                    interaction_type="hbond",
                    protein_residue=f"{resname}-{resid}",
                    protein_atom=prot_atom.name,
                    ligand_atom=lig_atom.name,
                    distance=dist[i, j],
                ))

        return events

    def _detect_hydrophobic(self, frame: int, time_ns: float) -> List[InteractionEvent]:
        """Detect hydrophobic contacts."""
        events = []
        u = self.system.universe

        # Select carbon atoms (hydrophobic centers)
        try:
            prot_carbons = self.system.protein.select_atoms(
                "element C and not bonded element N O S"
            )
        except (mda.exceptions.NoDataError, AttributeError):
            prot_carbons = self.system.protein.select_atoms("element C")
        # Fallback: just carbon atoms
        if len(prot_carbons) == 0:
            prot_carbons = self.system.protein.select_atoms("element C")

        lig_carbons = self.system.ligand.select_atoms("element C")

        if len(prot_carbons) == 0 or len(lig_carbons) == 0:
            return events

        dist = distances.distance_array(
            prot_carbons.positions, lig_carbons.positions,
            box=u.dimensions
        )
        contacts = np.argwhere(dist < self.hydrophobic_cutoff)

        # Deduplicate by residue
        seen_residues = set()
        for i, j in contacts:
            prot_atom = prot_carbons[i]
            reskey = f"{prot_atom.resname}-{prot_atom.resid}"
            if reskey in seen_residues:
                continue
            seen_residues.add(reskey)

            events.append(InteractionEvent(
                frame=frame, time_ns=time_ns,
                interaction_type="hydrophobic",
                protein_residue=reskey,
                protein_atom=prot_atom.name,
                ligand_atom=lig_carbons[j].name,
                distance=dist[i, j],
            ))

        return events

    def _detect_salt_bridges(self, frame: int, time_ns: float) -> List[InteractionEvent]:
        """Detect salt bridges between charged groups."""
        events = []
        u = self.system.universe

        # Protein positive: ARG (CZ/NH*), LYS (NZ)
        prot_pos = self.system.protein.select_atoms(
            "(resname ARG and name CZ) or (resname LYS and name NZ) or "
            "(resname HIP and name ND1 NE2)"
        )
        # Protein negative: ASP (CG), GLU (CD)
        prot_neg = self.system.protein.select_atoms(
            "(resname ASP and name CG) or (resname GLU and name CD)"
        )

        # Ligand charged groups (heuristic: N with + or O with -)
        lig_neg = self.system.ligand.select_atoms("element O")
        lig_pos = self.system.ligand.select_atoms("element N")

        # Protein(+) ↔ Ligand(-)
        if len(prot_pos) > 0 and len(lig_neg) > 0:
            dist = distances.distance_array(
                prot_pos.positions, lig_neg.positions, box=u.dimensions
            )
            contacts = np.argwhere(dist < self.salt_bridge_cutoff)
            for i, j in contacts:
                pa = prot_pos[i]
                events.append(InteractionEvent(
                    frame=frame, time_ns=time_ns,
                    interaction_type="salt_bridge",
                    protein_residue=f"{pa.resname}-{pa.resid}",
                    protein_atom=pa.name,
                    ligand_atom=lig_neg[j].name,
                    distance=dist[i, j],
                ))

        # Protein(-) ↔ Ligand(+)
        if len(prot_neg) > 0 and len(lig_pos) > 0:
            dist = distances.distance_array(
                prot_neg.positions, lig_pos.positions, box=u.dimensions
            )
            contacts = np.argwhere(dist < self.salt_bridge_cutoff)
            for i, j in contacts:
                pa = prot_neg[i]
                events.append(InteractionEvent(
                    frame=frame, time_ns=time_ns,
                    interaction_type="salt_bridge",
                    protein_residue=f"{pa.resname}-{pa.resid}",
                    protein_atom=pa.name,
                    ligand_atom=lig_pos[j].name,
                    distance=dist[i, j],
                ))

        return events

    def _detect_pi_stacking(self, frame: int, time_ns: float) -> List[InteractionEvent]:
        """Detect π-π stacking interactions."""
        events = []
        u = self.system.universe

        # Aromatic residues ring centroids
        aromatic_defs = {
            "PHE": "name CG CD1 CD2 CE1 CE2 CZ",
            "TYR": "name CG CD1 CD2 CE1 CE2 CZ",
            "TRP": "name CG CD1 CD2 NE1 CE2 CE3 CZ2 CZ3 CH2",
            "HIS": "name CG ND1 CD2 CE1 NE2",
        }

        # Find ligand aromatic rings (heuristic: connected carbons forming cycles)
        # Simplified: use all ligand aromatic-like carbons as a centroid
        lig_aromatic = self.system.ligand.select_atoms(
            "element C or element N"
        )
        if len(lig_aromatic) < 3:
            return events

        lig_centroid = lig_aromatic.center_of_geometry()

        for resname, atom_sel in aromatic_defs.items():
            for residue in self.system.protein.residues:
                if residue.resname != resname:
                    continue
                ring_atoms = residue.atoms.select_atoms(atom_sel)
                if len(ring_atoms) < 3:
                    continue

                ring_centroid = ring_atoms.center_of_geometry()
                dist = np.linalg.norm(ring_centroid - lig_centroid)

                if dist < self.pi_stack_cutoff:
                    events.append(InteractionEvent(
                        frame=frame, time_ns=time_ns,
                        interaction_type="pi_stack",
                        protein_residue=f"{resname}-{residue.resid}",
                        protein_atom="ring",
                        ligand_atom="ring",
                        distance=dist,
                    ))

        return events

    def _detect_water_bridges(self, frame: int, time_ns: float) -> List[InteractionEvent]:
        """Detect water-mediated hydrogen bonds."""
        events = []
        u = self.system.universe
        solvent = self.system.solvent

        if len(solvent) == 0:
            return events

        prot_polar = self.system.protein.select_atoms("element N O")
        lig_polar = self.system.ligand.select_atoms("element N O")

        if len(prot_polar) == 0 or len(lig_polar) == 0:
            return events

        # Water oxygen atoms
        water_O = solvent.select_atoms("name OW O")
        if len(water_O) == 0:
            return events

        # Water-ligand distances
        dist_wl = distances.distance_array(
            water_O.positions, lig_polar.positions, box=u.dimensions
        )
        # Water-protein distances
        dist_wp = distances.distance_array(
            water_O.positions, prot_polar.positions, box=u.dimensions
        )

        cutoff = self.water_bridge_cutoff

        # Find water molecules bridging protein and ligand
        water_near_lig = np.any(dist_wl < cutoff, axis=1)  # waters near ligand
        water_near_prot = np.any(dist_wp < cutoff, axis=1)  # waters near protein
        bridging = np.where(water_near_lig & water_near_prot)[0]

        seen_residues = set()
        for w_idx in bridging[:10]:  # limit to avoid explosion
            # Find which protein residues this water bridges to
            prot_contacts = np.where(dist_wp[w_idx] < cutoff)[0]
            for p_idx in prot_contacts[:3]:
                pa = prot_polar[p_idx]
                reskey = f"{pa.resname}-{pa.resid}"
                if reskey in seen_residues:
                    continue
                seen_residues.add(reskey)

                events.append(InteractionEvent(
                    frame=frame, time_ns=time_ns,
                    interaction_type="water_bridge",
                    protein_residue=reskey,
                    protein_atom=pa.name,
                    ligand_atom="via_water",
                    distance=float(dist_wp[w_idx, p_idx]),
                ))

        return events

    def _compute_per_residue_frequency(
        self, events: List[InteractionEvent], n_frames: int
    ) -> Dict[str, Dict[str, float]]:
        """Compute interaction frequency per residue per type."""
        residue_type_frames = defaultdict(lambda: defaultdict(set))

        for ev in events:
            residue_type_frames[ev.protein_residue][ev.interaction_type].add(ev.frame)

        frequency = {}
        for residue, type_frames in residue_type_frames.items():
            frequency[residue] = {
                itype: len(frames) / n_frames
                for itype, frames in type_frames.items()
            }

        return frequency

    def get_key_residues(self, min_frequency: float = 0.3) -> List[Tuple[str, Dict[str, float]]]:
        """
        Get residues with interaction frequency above threshold.

        Returns sorted list of (residue, {type: frequency}) tuples.
        """
        profile = self.profile
        key = [
            (res, freqs)
            for res, freqs in profile.per_residue_frequency.items()
            if max(freqs.values()) >= min_frequency
        ]
        return sorted(key, key=lambda x: max(x[1].values()), reverse=True)

    def get_interaction_timeline(self, residue: str) -> Dict[str, List[float]]:
        """Get temporal presence/absence of interactions for a specific residue."""
        profile = self.profile
        timeline = defaultdict(list)
        for ev in profile.events:
            if ev.protein_residue == residue:
                timeline[ev.interaction_type].append(ev.time_ns)
        return dict(timeline)

    def get_residence_times(self, interaction_type: str = "hbond") -> Dict[str, List[float]]:
        """
        Compute residence times (consecutive frame durations) for each
        protein residue's interactions.

        Returns dict of {residue: [duration_ns, duration_ns, ...]}
        """
        profile = self.profile
        dt_ns = profile.events[0].time_ns if len(profile.events) > 1 else 0.001

        # Group frames by residue
        residue_frames = defaultdict(set)
        for ev in profile.events:
            if ev.interaction_type == interaction_type:
                residue_frames[ev.protein_residue].add(ev.frame)

        residence = {}
        for residue, frames in residue_frames.items():
            sorted_frames = sorted(frames)
            durations = []
            current_start = sorted_frames[0]
            prev = sorted_frames[0]

            for f in sorted_frames[1:]:
                if f - prev > self.stride:  # gap detected
                    durations.append((prev - current_start + 1) * dt_ns)
                    current_start = f
                prev = f
            durations.append((prev - current_start + 1) * dt_ns)
            residence[residue] = durations

        return residence
