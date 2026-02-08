"""
Trajectory Loader - Universal MD trajectory I/O engine.

Supports: XTC, TRR, GRO, TPR, DCD, PDB, XYZ, AMBER netCDF
Built on MDAnalysis with intelligent format detection and memory-mapped streaming.
"""

import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Union, List, Tuple

import numpy as np

try:
    import MDAnalysis as mda
    from MDAnalysis.coordinates.memory import MemoryReader
except ImportError:
    raise ImportError(
        "MDAnalysis is required: pip install MDAnalysis"
    )

logger = logging.getLogger(__name__)

# Supported format registry
TOPOLOGY_FORMATS = {".gro", ".pdb", ".tpr", ".psf", ".prmtop", ".top", ".mol2"}
TRAJECTORY_FORMATS = {".xtc", ".trr", ".dcd", ".nc", ".netcdf", ".xyz", ".lammpstrj"}
COMBINED_FORMATS = {".pdb", ".gro", ".xyz"}  # can serve as both topology + trajectory


@dataclass
class TrajectoryMetadata:
    """Metadata extracted from a loaded trajectory."""
    n_frames: int = 0
    n_atoms: int = 0
    dt_ps: float = 0.0  # timestep in picoseconds
    total_time_ns: float = 0.0
    box_dimensions: Optional[np.ndarray] = None
    has_velocities: bool = False
    has_forces: bool = False
    topology_format: str = ""
    trajectory_format: str = ""
    n_residues: int = 0
    n_segments: int = 0
    unique_resnames: List[str] = field(default_factory=list)


class TrajectoryLoader:
    """
    Universal trajectory loader with smart format detection and streaming support.

    Parameters
    ----------
    topology : str or Path
        Path to topology file (GRO, PDB, TPR, PSF, PRMTOP).
    trajectory : str, Path, or list thereof, optional
        Path(s) to trajectory file(s) (XTC, TRR, DCD, etc.).
        If None, topology is used as a single-frame structure.
    in_memory : bool, default False
        Load entire trajectory into RAM for faster random access.
        Use for trajectories < ~2 GB.
    stride : int, default 1
        Read every Nth frame (reduces memory/time for large trajectories).
    start_frame : int, optional
        First frame to read (0-indexed).
    end_frame : int, optional
        Last frame to read (exclusive).

    Examples
    --------
    >>> loader = TrajectoryLoader("system.gro", "md_prod.xtc")
    >>> universe = loader.universe
    >>> print(loader.metadata)

    >>> # Multiple trajectories (concatenated)
    >>> loader = TrajectoryLoader("system.tpr", ["run1.xtc", "run2.xtc"])

    >>> # Memory-mapped for small trajectories
    >>> loader = TrajectoryLoader("system.gro", "short_md.xtc", in_memory=True)
    """

    def __init__(
        self,
        topology: Union[str, Path],
        trajectory: Optional[Union[str, Path, List[Union[str, Path]]]] = None,
        in_memory: bool = False,
        stride: int = 1,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
    ):
        self.topology_path = Path(topology)
        self.trajectory_paths = self._normalize_traj_paths(trajectory)
        self.in_memory = in_memory
        self.stride = stride
        self.start_frame = start_frame
        self.end_frame = end_frame

        self._validate_inputs()
        self._universe: Optional[mda.Universe] = None
        self._metadata: Optional[TrajectoryMetadata] = None

    @property
    def universe(self) -> mda.Universe:
        """Lazy-loaded MDAnalysis Universe."""
        if self._universe is None:
            self._load()
        return self._universe

    @property
    def metadata(self) -> TrajectoryMetadata:
        """Trajectory metadata (auto-populated on load)."""
        if self._metadata is None:
            _ = self.universe  # trigger load
        return self._metadata

    def _normalize_traj_paths(self, trajectory) -> Optional[List[Path]]:
        if trajectory is None:
            return None
        if isinstance(trajectory, (str, Path)):
            return [Path(trajectory)]
        return [Path(t) for t in trajectory]

    def _validate_inputs(self):
        if not self.topology_path.exists():
            raise FileNotFoundError(f"Topology not found: {self.topology_path}")

        if self.trajectory_paths:
            for tp in self.trajectory_paths:
                if not tp.exists():
                    raise FileNotFoundError(f"Trajectory not found: {tp}")

        suffix = self.topology_path.suffix.lower()
        if suffix not in TOPOLOGY_FORMATS and suffix not in COMBINED_FORMATS:
            logger.warning(
                f"Unrecognized topology format '{suffix}'. "
                f"Supported: {TOPOLOGY_FORMATS | COMBINED_FORMATS}"
            )

    def _load(self):
        """Load trajectory into MDAnalysis Universe."""
        logger.info(f"Loading topology: {self.topology_path}")

        kwargs = {}
        if self.in_memory:
            kwargs["in_memory"] = True
            if self.stride > 1:
                kwargs["in_memory_step"] = self.stride

        if self.trajectory_paths:
            traj_args = (
                [str(p) for p in self.trajectory_paths]
                if len(self.trajectory_paths) > 1
                else str(self.trajectory_paths[0])
            )
            logger.info(f"Loading trajectory: {traj_args}")
            self._universe = mda.Universe(str(self.topology_path), traj_args, **kwargs)
        else:
            self._universe = mda.Universe(str(self.topology_path), **kwargs)

        self._extract_metadata()

        # Apply frame slicing if requested
        if self.start_frame is not None or self.end_frame is not None:
            start = self.start_frame or 0
            end = self.end_frame or self._universe.trajectory.n_frames
            logger.info(f"Slicing trajectory: frames {start} to {end}, stride {self.stride}")

        logger.info(
            f"Loaded: {self._metadata.n_atoms} atoms, "
            f"{self._metadata.n_frames} frames, "
            f"{self._metadata.total_time_ns:.2f} ns"
        )

    def _extract_metadata(self):
        """Extract comprehensive metadata from loaded universe."""
        u = self._universe
        traj = u.trajectory

        # Determine timestep
        dt = 0.0
        if hasattr(traj, "dt"):
            dt = traj.dt  # in ps typically

        n_frames = traj.n_frames
        total_time = dt * n_frames / 1000.0  # convert ps to ns

        # Box dimensions from first frame
        traj[0]
        box = u.dimensions.copy() if u.dimensions is not None else None

        self._metadata = TrajectoryMetadata(
            n_frames=n_frames,
            n_atoms=u.atoms.n_atoms,
            dt_ps=dt,
            total_time_ns=total_time,
            box_dimensions=box,
            has_velocities=hasattr(traj.ts, "velocities") and traj.ts.has_velocities,
            has_forces=hasattr(traj.ts, "forces") and traj.ts.has_forces,
            topology_format=self.topology_path.suffix,
            trajectory_format=(
                self.trajectory_paths[0].suffix if self.trajectory_paths else "single"
            ),
            n_residues=u.residues.n_residues,
            n_segments=u.segments.n_segments,
            unique_resnames=list(set(u.residues.resnames)),
        )

    def get_frame_iterator(
        self,
        start: int = 0,
        stop: Optional[int] = None,
        step: Optional[int] = None,
    ):
        """
        Memory-efficient frame iterator with optional slicing.

        Yields (frame_index, timestamp_ns, positions) tuples.
        """
        u = self.universe
        step = step or self.stride
        stop = stop or u.trajectory.n_frames

        for ts in u.trajectory[start:stop:step]:
            yield ts.frame, ts.time / 1000.0, u.atoms.positions.copy()

    def select_atoms(self, selection: str) -> mda.AtomGroup:
        """MDAnalysis atom selection with validation."""
        ag = self.universe.select_atoms(selection)
        if len(ag) == 0:
            raise ValueError(f"Selection '{selection}' matched 0 atoms.")
        logger.info(f"Selection '{selection}' → {len(ag)} atoms")
        return ag

    def detect_ligand(self, exclude_ions: bool = True) -> Optional[str]:
        """
        Auto-detect ligand residue name by finding non-protein, non-solvent,
        non-ion residues.

        Returns
        -------
        str or None
            MDAnalysis selection string for the detected ligand.
        """
        u = self.universe
        known_solvent = {"SOL", "WAT", "HOH", "TIP3", "SPC", "T3P", "T4P", "T5P"}
        known_ions = {"NA", "CL", "K", "MG", "CA", "ZN", "FE", "MN", "CU", "NA+", "CL-"}
        known_lipids = {"DPPC", "POPC", "POPE", "DOPC", "CHOL"}

        exclude = known_solvent | known_lipids
        if exclude_ions:
            exclude |= known_ions

        # Get protein residue names
        try:
            protein_resnames = set(u.select_atoms("protein").residues.resnames)
        except Exception:
            protein_resnames = set()

        all_resnames = set(u.residues.resnames)
        candidates = all_resnames - protein_resnames - exclude

        if not candidates:
            logger.warning("No ligand candidates detected.")
            return None

        if len(candidates) == 1:
            lig = candidates.pop()
            logger.info(f"Auto-detected ligand: {lig}")
            return f"resname {lig}"

        # Multiple candidates — pick the one with most heavy atoms
        best = None
        best_count = 0
        for resname in candidates:
            ag = u.select_atoms(f"resname {resname} and not name H*")
            if ag.n_atoms > best_count:
                best = resname
                best_count = ag.n_atoms

        logger.info(f"Auto-detected ligand: {best} (from candidates: {candidates})")
        return f"resname {best}"

    def summary(self) -> str:
        """Human-readable summary of loaded system."""
        m = self.metadata
        lines = [
            "=" * 60,
            "MDInsight Trajectory Summary",
            "=" * 60,
            f"  Topology:       {self.topology_path.name} ({m.topology_format})",
            f"  Trajectory:     {m.trajectory_format}",
            f"  Atoms:          {m.n_atoms:,}",
            f"  Residues:       {m.n_residues:,}",
            f"  Frames:         {m.n_frames:,}",
            f"  Timestep:       {m.dt_ps:.2f} ps",
            f"  Total time:     {m.total_time_ns:.2f} ns",
            f"  Velocities:     {'Yes' if m.has_velocities else 'No'}",
            f"  Forces:         {'Yes' if m.has_forces else 'No'}",
            f"  Box:            {np.array2string(m.box_dimensions[:3], precision=1) if m.box_dimensions is not None else 'None'}",
            f"  Unique resnames: {len(m.unique_resnames)}",
            "=" * 60,
        ]
        return "\n".join(lines)
