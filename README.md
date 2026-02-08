# MDInsight — AI-Powered Deep Dive Analysis for Molecular Dynamics

**Go beyond RMSD/RMSF.** MDInsight reads your MD trajectory and instantly runs a comprehensive, AI-powered analysis of protein-ligand interactions — generating an interactive HTML dashboard with insights that would normally take weeks of manual scripting.

---

## The Problem

Current MD analysis is fragmented and shallow:

| Tool | What it does | What it misses |
|------|-------------|---------------|
| **PyMOL/Chimera** | Beautiful static visualization | No trajectory analysis, no quantitative interaction profiling |
| **MDAnalysis/MDTraj** | Trajectory parsing, RMSD/RMSF | No built-in interaction detection, no ML, no reports |
| **PLIP** | Single-frame interaction detection | No temporal evolution, no dynamics |
| **gmx_MMPBSA** | Binding energy decomposition | Slow, requires extensive setup, no visualization |
| **ProLIF** | Interaction fingerprints | Limited ML integration, basic visualization |

**MDInsight unifies all of this** into a single `run_deep_dive()` call with AI-powered analysis and beautiful interactive reports.

---

## What You Get

### 8-Phase Deep Dive Pipeline

```
Phase 1: Deep Interaction Profiling
  → H-bonds, hydrophobic, salt bridges, π-stacking, water bridges
  → Per-residue frequency maps, residence times

Phase 2: Advanced Dynamics
  → RMSD, RMSF, radius of gyration (standard)
  → Binding site PCA, DCCM, quasi-harmonic entropy (advanced)
  → Ligand RMSD, protein-ligand COM distance

Phase 3: Interaction Fingerprints
  → Binary encoding of interaction state per frame
  → Tanimoto similarity, temporal autocorrelation
  → Dominant interaction patterns

Phase 4: Energy Decomposition
  → Per-residue proxy energy scoring
  → Distance-dependent empirical potentials
  → Hotspot residue ranking

Phase 5: AI Conformational Clustering
  → K-means, GMM, HDBSCAN, spectral, DBSCAN
  → Automatic algorithm & k selection via silhouette score
  → State transition probability matrices

Phase 6: Binding Mode Transitions
  → Cluster-label change detection
  → CUSUM changepoint detection on fingerprints
  → PCA projection discontinuity detection
  → Dwell time distributions per state

Phase 7: Feature Importance
  → Random Forest + mutual information + permutation importance
  → Identifies which interactions drive binding mode differences
  → State-specific interaction enrichment

Phase 8: Anomaly Detection
  → Isolation Forest + Local Outlier Factor
  → Flags unusual conformations and trajectory artifacts
  → Detects rare transient interactions (<2% occurrence)
```

### Interactive HTML Report

The generated report includes 12+ interactive Plotly panels:
- Dynamics overview (6-panel: RMSD, RMSF, RoG, ligand RMSD, P-L distance, BS RMSF)
- Interaction frequency heatmap
- Interaction timeline (stacked area)
- Protein-ligand interaction network graph
- Per-residue energy decomposition (stacked bars)
- Interaction fingerprint evolution heatmap
- Interaction persistence Gantt chart
- 3D PCA scatter (colored by cluster)
- Dynamic cross-correlation matrix
- Cluster populations + state timeline
- Transition probability matrix
- Anomaly score timeline
- Feature importance ranking
- Fingerprint autocorrelation

---

## Installation

```bash
# Core installation
pip install mdinsight

# With all optional dependencies
pip install mdinsight[full]

# From source
git clone https://github.com/mdinsight/mdinsight.git
cd mdinsight
pip install -e ".[dev]"
```

### Dependencies
- **Core:** MDAnalysis, NumPy, SciPy, scikit-learn, Plotly, NetworkX
- **Optional:** HDBSCAN, UMAP, NGLView

---

## Quick Start

### Python API (3 lines)

```python
from mdinsight import MDInsight

engine = MDInsight("system.gro", "trajectory.xtc", ligand_selection="resname LIG")
engine.run_deep_dive()
engine.generate_report("deep_dive_report.html")
```

### Command Line

```bash
# Basic usage
mdinsight run -t system.gro -x trajectory.xtc -l "resname LIG" -o report.html

# Multiple trajectories with stride
mdinsight run -t system.tpr -x run1.xtc run2.xtc --stride 5 --in-memory

# Just trajectory info
mdinsight info -t system.gro -x trajectory.xtc
```

### Check the sample reports here: [sample_report](https://github.com/pritampanda15/MDInsight/tree/main/examples/sample_report.html)

### Advanced Usage

```python
from mdinsight import MDInsight

engine = MDInsight(
    "system.tpr",
    ["equilibration.xtc", "production.xtc"],
    ligand_selection="resname PRF",  # propofol
    binding_site_cutoff=6.0,
    stride=2,
    in_memory=True,
)

# Customized analysis
engine.run_deep_dive(
    start_frame=500,          # skip equilibration
    compute_pca=True,
    compute_entropy=True,
    clustering_algorithms=["kmeans", "gmm", "hdbscan"],
    interaction_stride=5,      # faster interaction analysis
)

# Access individual results programmatically
key_residues = engine.interaction_analyzer.get_key_residues(min_frequency=0.5)
hotspots = engine.energy_decomposer.get_hotspot_residues(top_n=10)
transitions = engine.transition_detector.analysis.events
summary = engine.get_executive_summary()

# Specific residue deep-dive
timeline = engine.interaction_analyzer.get_interaction_timeline("ASN-265")
residence = engine.interaction_analyzer.get_residence_times("hbond")
energy_time = engine.energy_decomposer.energy_over_time("ASN-265", window=100)

# Cluster-specific analysis
state_interactions = engine.feature_analyzer.get_state_specific_interactions()

# Generate report
engine.generate_report("propofol_gabaa_deepdive.html")
```

---

## Supported File Formats

| Format | Type | Engine |
|--------|------|--------|
| `.xtc` | Trajectory | GROMACS compressed |
| `.trr` | Trajectory | GROMACS full precision |
| `.gro` | Topology/Structure | GROMACS |
| `.tpr` | Topology | GROMACS run input |
| `.pdb` | Structure | Universal |
| `.dcd` | Trajectory | CHARMM/NAMD |
| `.psf` | Topology | CHARMM |
| `.nc` | Trajectory | AMBER NetCDF |
| `.prmtop` | Topology | AMBER |
| `.xyz` | Structure | Generic |

---

## Comparison

| Feature | MDInsight | ProLIF | PLIP | gmx_MMPBSA | Manual Scripts |
|---------|-----------|--------|------|------------|----------------|
| Trajectory I/O | ✅ All formats | ✅ | ❌ Single frame | ✅ | ✅ |
| H-bonds | ✅ Temporal | ✅ | ✅ | ❌ | Custom |
| Hydrophobic | ✅ Temporal | ✅ | ✅ | ❌ | Custom |
| π-stacking | ✅ Temporal | ✅ | ✅ | ❌ | Hard |
| Water bridges | ✅ Temporal | ❌ | ✅ | ❌ | Very hard |
| Salt bridges | ✅ Temporal | ✅ | ✅ | ❌ | Custom |
| Fingerprints | ✅ + similarity | ✅ | ❌ | ❌ | Custom |
| PCA | ✅ Binding site | ❌ | ❌ | ❌ | Custom |
| DCCM | ✅ | ❌ | ❌ | ❌ | Custom |
| Energy decomp | ✅ Proxy | ❌ | ❌ | ✅ Full | Custom |
| ML clustering | ✅ 5 algorithms | ❌ | ❌ | ❌ | Custom |
| Transitions | ✅ 3 methods | ❌ | ❌ | ❌ | Custom |
| Feature importance | ✅ RF+MI+Perm | ❌ | ❌ | ❌ | Custom |
| Anomaly detection | ✅ IF+LOF | ❌ | ❌ | ❌ | ❌ |
| Interactive report | ✅ 12+ panels | Basic | ❌ | ❌ | ❌ |
| One command | ✅ | ❌ | ❌ | ❌ | ❌ |

---

## Roadmap

- [ ] **v0.2**: tICA (time-lagged independent component analysis)
- [ ] **v0.2**: MM/GBSA integration via `gmx_MMPBSA`
- [ ] **v0.3**: NGL-based 3D trajectory viewer in report
- [ ] **v0.3**: Markov State Models (MSM) via PyEMMA
- [ ] **v0.4**: LLM-powered natural language summary of findings
- [ ] **v0.4**: Multi-ligand comparison mode
- [ ] **v0.5**: Real-time streaming analysis for running simulations
- [ ] **v0.5**: GPU-accelerated interaction detection via RAPIDS
- [ ] **v1.0**: Full web UI with drag-and-drop trajectory upload

---

## Citation

If you use MDInsight in your research, please cite:

```bibtex
@software{MDInsight2026,
  title={MDInsight: AI-Powered Deep Dive Analysis for Molecular Dynamics},
  year={2026},
  author={Pritam Kumar Panda}
  url={https://github.com/pritampanda15/MDInsight}
}
```

---

## License

MIT License — use freely in academic and commercial projects.
