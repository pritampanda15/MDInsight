"""
MDInsight Engine - One-command deep dive analysis orchestrator.

This is the main entry point. Load trajectory → run_deep_dive() → generate_report().
That's it.

Example:
    >>> from mdinsight import MDInsight
    >>> engine = MDInsight("system.gro", "trajectory.xtc", ligand_selection="resname LIG")
    >>> engine.run_deep_dive()
    >>> engine.generate_report("deep_dive_report.html")
"""

import logging
import time
from pathlib import Path
from typing import Dict, Optional, Union, List

import numpy as np

from mdinsight.core.trajectory import TrajectoryLoader
from mdinsight.core.system import MolecularSystem
from mdinsight.analysis.interactions import InteractionAnalyzer
from mdinsight.analysis.dynamics import DynamicsAnalyzer
from mdinsight.analysis.fingerprints import InteractionFingerprinter
from mdinsight.analysis.energy import EnergyDecomposer
from mdinsight.ai.clustering import ConformationalClusterer
from mdinsight.ai.transitions import TransitionDetector
from mdinsight.ai.feature_importance import BindingFeatureAnalyzer
from mdinsight.ai.anomaly import TrajectoryAnomalyDetector
from mdinsight.visualization.dashboard import DashboardGenerator
from mdinsight.visualization.networks import InteractionNetworkViz
from mdinsight.visualization.temporal import TemporalViz
from mdinsight.reports.html_report import ReportGenerator

logger = logging.getLogger(__name__)


def _group_consecutive_residues(residues):
    """Group consecutive residue IDs into sublists."""
    if len(residues) == 0:
        return []
    groups = [[residues[0]]]
    for r in residues[1:]:
        if r - groups[-1][-1] <= 2:
            groups[-1].append(r)
        else:
            groups.append([r])
    return groups


class MDInsight:
    """
    AI-Powered Deep Dive Analysis Engine for Molecular Dynamics.

    One object to rule them all — loads your trajectory and runs every analysis
    in the toolkit, then generates a comprehensive interactive HTML report.

    Parameters
    ----------
    topology : str or Path
        Topology file (GRO, PDB, TPR, PSF).
    trajectory : str, Path, or list, optional
        Trajectory file(s) (XTC, TRR, DCD, etc.).
    ligand_selection : str, optional
        MDAnalysis selection for ligand. Auto-detected if None.
    binding_site_cutoff : float, default 5.0
        Distance cutoff for binding site definition (Å).
    stride : int, default 1
        Frame stride for analysis.
    in_memory : bool, default False
        Load trajectory into RAM.
    log_level : str, default "INFO"
        Logging level.

    Examples
    --------
    >>> # Basic usage
    >>> engine = MDInsight("system.gro", "md.xtc", ligand_selection="resname LIG")
    >>> engine.run_deep_dive()
    >>> engine.generate_report("analysis.html")

    >>> # Customized analysis
    >>> engine = MDInsight("system.tpr", ["run1.xtc", "run2.xtc"],
    ...                    ligand_selection="resname DRG", stride=5)
    >>> engine.run_deep_dive(
    ...     start_frame=100,
    ...     compute_pca=True,
    ...     clustering_algorithms=["kmeans", "gmm", "hdbscan"],
    ... )
    >>> engine.generate_report("custom_analysis.html")

    >>> # Access individual results
    >>> key_residues = engine.interaction_analyzer.get_key_residues(min_frequency=0.5)
    >>> hotspots = engine.energy_decomposer.get_hotspot_residues(top_n=5)
    >>> transitions = engine.transition_detector.analysis.events
    """

    def __init__(
        self,
        topology: Union[str, Path],
        trajectory: Optional[Union[str, Path, List]] = None,
        ligand_selection: Optional[str] = None,
        binding_site_cutoff: float = 5.0,
        stride: int = 1,
        in_memory: bool = False,
        log_level: str = "INFO",
    ):
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, log_level),
            format="%(asctime)s [MDInsight] %(message)s",
            datefmt="%H:%M:%S",
        )

        # Load trajectory
        logger.info("=" * 60)
        logger.info("MDInsight v0.1.0 — AI-Powered MD Deep Dive")
        logger.info("=" * 60)

        self.loader = TrajectoryLoader(
            topology, trajectory, in_memory=in_memory, stride=stride,
        )
        logger.info(self.loader.summary())

        # Build molecular system
        self.system = MolecularSystem(
            self.loader.universe,
            ligand_selection=ligand_selection,
            binding_site_cutoff=binding_site_cutoff,
        )

        self.stride = stride

        # Analysis modules (populated after run_deep_dive)
        self.interaction_analyzer: Optional[InteractionAnalyzer] = None
        self.dynamics_analyzer: Optional[DynamicsAnalyzer] = None
        self.fingerprinter: Optional[InteractionFingerprinter] = None
        self.energy_decomposer: Optional[EnergyDecomposer] = None
        self.clusterer: Optional[ConformationalClusterer] = None
        self.transition_detector: Optional[TransitionDetector] = None
        self.feature_analyzer: Optional[BindingFeatureAnalyzer] = None
        self.anomaly_detector: Optional[TrajectoryAnomalyDetector] = None

        # Visualization
        self.dashboard = DashboardGenerator()

        self._analysis_complete = False

    def run_deep_dive(
        self,
        start_frame: int = 0,
        stop_frame: Optional[int] = None,
        compute_pca: bool = True,
        compute_cross_correlation: bool = True,
        compute_entropy: bool = True,
        clustering_algorithms: Optional[List[str]] = None,
        interaction_stride: Optional[int] = None,
    ):
        """
        Run the complete deep dive analysis pipeline.

        Parameters
        ----------
        start_frame : int
            First frame to analyze.
        stop_frame : int, optional
            Last frame (exclusive).
        compute_pca : bool
            Run PCA on binding site dynamics.
        compute_cross_correlation : bool
            Compute DCCM.
        compute_entropy : bool
            Estimate quasi-harmonic entropy.
        clustering_algorithms : list of str, optional
            Algorithms for conformational clustering.
        interaction_stride : int, optional
            Separate stride for interaction analysis (can be larger for speed).
        """
        t0 = time.time()
        int_stride = interaction_stride or self.stride

        # Phase 1: Interaction Analysis
        logger.info("\n" + "─" * 50)
        logger.info("Phase 1: Deep Interaction Profiling")
        logger.info("─" * 50)
        self.interaction_analyzer = InteractionAnalyzer(
            self.system, stride=int_stride,
        )
        self.interaction_analyzer.analyze(start_frame, stop_frame)

        # Phase 2: Dynamics Analysis
        logger.info("\n" + "─" * 50)
        logger.info("Phase 2: Advanced Dynamics Analysis")
        logger.info("─" * 50)
        self.dynamics_analyzer = DynamicsAnalyzer(
            self.system, stride=self.stride,
        )
        self.dynamics_analyzer.analyze(
            start_frame, stop_frame,
            compute_pca=compute_pca,
            compute_cross_correlation=compute_cross_correlation,
            compute_entropy=compute_entropy,
        )

        # Phase 3: Interaction Fingerprints
        logger.info("\n" + "─" * 50)
        logger.info("Phase 3: Interaction Fingerprint Generation")
        logger.info("─" * 50)
        self.fingerprinter = InteractionFingerprinter(
            self.interaction_analyzer, self.system,
        )
        self.fingerprinter.compute()

        # Phase 4: Energy Decomposition
        logger.info("\n" + "─" * 50)
        logger.info("Phase 4: Per-Residue Energy Decomposition")
        logger.info("─" * 50)
        self.energy_decomposer = EnergyDecomposer(self.interaction_analyzer)
        self.energy_decomposer.compute()

        # Phase 5: AI — Conformational Clustering
        logger.info("\n" + "─" * 50)
        logger.info("Phase 5: AI Conformational Clustering")
        logger.info("─" * 50)
        self.clusterer = ConformationalClusterer(
            fingerprinter=self.fingerprinter,
            dynamics_result=self.dynamics_analyzer.result,
        )
        try:
            self.clusterer.cluster(algorithms=clustering_algorithms)
        except Exception as e:
            logger.warning(f"Clustering failed: {e}")

        # Phase 6: Transition Detection
        logger.info("\n" + "─" * 50)
        logger.info("Phase 6: Binding Mode Transition Detection")
        logger.info("─" * 50)
        dt_ns = self.loader.metadata.dt_ps / 1000.0 * self.stride
        self.transition_detector = TransitionDetector(
            cluster_result=self.clusterer.best if self.clusterer._best_result else None,
            fingerprinter=self.fingerprinter,
            dynamics_result=self.dynamics_analyzer.result,
            dt_ns=dt_ns or 0.001,
        )
        try:
            self.transition_detector.detect()
        except Exception as e:
            logger.warning(f"Transition detection failed: {e}")

        # Phase 7: Feature Importance
        logger.info("\n" + "─" * 50)
        logger.info("Phase 7: Binding Feature Importance Analysis")
        logger.info("─" * 50)
        if self.clusterer._best_result and self.clusterer.best.n_clusters >= 2:
            self.feature_analyzer = BindingFeatureAnalyzer(
                self.fingerprinter, self.clusterer.best,
            )
            try:
                self.feature_analyzer.analyze()
            except Exception as e:
                logger.warning(f"Feature importance failed: {e}")

        # Phase 8: Anomaly Detection
        logger.info("\n" + "─" * 50)
        logger.info("Phase 8: Trajectory Anomaly Detection")
        logger.info("─" * 50)
        self.anomaly_detector = TrajectoryAnomalyDetector(
            fingerprinter=self.fingerprinter,
            dynamics_result=self.dynamics_analyzer.result,
        )
        try:
            self.anomaly_detector.detect()
        except Exception as e:
            logger.warning(f"Anomaly detection failed: {e}")

        self._analysis_complete = True

        elapsed = time.time() - t0
        logger.info("\n" + "=" * 60)
        logger.info(f"Deep dive complete in {elapsed:.1f}s")
        logger.info("=" * 60)

        self._print_executive_summary()

    def _print_executive_summary(self):
        """Print executive summary to log."""
        logger.info("\n📊 EXECUTIVE SUMMARY")
        logger.info("─" * 40)

        # System
        ss = self.system.system_summary()
        logger.info(f"  System: {ss['n_protein_residues']} residues, "
                     f"ligand={ss['ligand_resnames']}")
        logger.info(f"  Binding site: {ss['n_binding_site_residues']} residues")

        # Interactions
        profile = self.interaction_analyzer.profile
        logger.info(f"  Total interactions: {sum(profile.interaction_summary.values())}")
        for itype, count in profile.interaction_summary.items():
            logger.info(f"    {itype}: {count}")

        # Key residues
        key_res = self.interaction_analyzer.get_key_residues(min_frequency=0.3)
        if key_res:
            logger.info(f"  Key residues (>30% freq): {', '.join(r[0] for r in key_res[:10])}")

        # Hotspots
        hotspots = self.energy_decomposer.get_hotspot_residues(top_n=5)
        if hotspots:
            logger.info(f"  Energy hotspots: " +
                         ", ".join(f"{r} ({e:.1f} kcal/mol)" for r, e in hotspots))

        # Clustering
        if self.clusterer._best_result:
            cl = self.clusterer.best
            logger.info(f"  Conformational states: {cl.n_clusters} "
                         f"({cl.algorithm}, silhouette={cl.silhouette_score:.3f})")

        # Entropy
        dr = self.dynamics_analyzer.result
        if dr.entropy_estimate:
            logger.info(f"  Configurational entropy: {dr.entropy_estimate:.1f} J/(mol·K)")

        # Anomalies
        if self.anomaly_detector._result:
            ar = self.anomaly_detector.result
            logger.info(f"  Anomalies: {ar.n_anomalies} frames ({ar.anomaly_fraction:.1%})")

    def generate_report(
        self,
        output_path: str = "mdinsight_report.html",
        title: Optional[str] = None,
    ) -> Path:
        """
        Generate comprehensive interactive HTML report.

        Parameters
        ----------
        output_path : str
            Output file path.
        title : str, optional
            Report title.

        Returns
        -------
        Path
            Path to generated report.
        """
        if not self._analysis_complete:
            raise RuntimeError("Run run_deep_dive() first.")

        title = title or f"MDInsight Deep Dive — {self.loader.topology_path.stem}"
        report = ReportGenerator(title=title, output_path=output_path)

        ss = self.system.system_summary()
        meta = self.loader.metadata
        profile = self.interaction_analyzer.profile
        dr = self.dynamics_analyzer.result

        # ── 1. System Summary ───────────────────────────────────────
        report.add_summary_table("System Overview", {
            "Topology": self.loader.topology_path.name,
            "Frames": f"{meta.n_frames:,}",
            "Total Time": f"{meta.total_time_ns:.2f} ns",
            "Timestep": f"{meta.dt_ps:.1f} ps",
            "Protein Residues": ss["n_protein_residues"],
            "Ligand": ", ".join(ss["ligand_resnames"]),
            "Ligand Atoms": ss["n_ligand_atoms"],
            "Binding Site Residues": ss["n_binding_site_residues"],
            "Total Atoms": f"{ss['total_atoms']:,}",
        })

        # ── 2. Dynamics Overview ────────────────────────────────────
        dynamics_cards = []
        if dr.rmsd is not None:
            dynamics_cards.append({"value": f"{np.mean(dr.rmsd):.1f} Ang", "label": "Mean RMSD (backbone)"})
        if dr.radius_of_gyration is not None:
            dynamics_cards.append({"value": f"{np.mean(dr.radius_of_gyration):.1f} Ang", "label": "Mean Rg"})
        if dr.protein_ligand_distance is not None:
            dynamics_cards.append({"value": f"{np.mean(dr.protein_ligand_distance):.1f} Ang", "label": "Mean Ligand-Protein Distance"})

        if dynamics_cards:
            report.add_stat_cards("Dynamics Overview", dynamics_cards,
                                  description="RMSD, RMSF, radius of gyration, and protein-ligand distance over the trajectory.")

        # Individual dynamics plots
        dynamics_figs = []
        for fig_fn in [
            lambda: self.dashboard.rmsd_plot(dr),
            lambda: self.dashboard.rmsf_plot(dr),
            lambda: self.dashboard.binding_site_rmsf_plot(dr),
            lambda: self.dashboard.rog_plot(dr),
        ]:
            try:
                fig = fig_fn()
                if fig.data:
                    dynamics_figs.append(fig)
            except Exception:
                pass

        if dynamics_figs:
            report.add_figures_to_last_section(dynamics_figs)

        # Dynamics insight
        insight_parts = []
        if dr.rmsd is not None and len(dr.rmsd) > 20:
            rmsd_mean = np.mean(dr.rmsd)
            rmsd_std = np.std(dr.rmsd)
            window = min(50, len(dr.rmsd) // 4)
            if window > 1:
                running = np.convolve(dr.rmsd, np.ones(window) / window, mode='valid')
                final = running[-1]
                eq_idx = np.argmax(np.abs(running - final) < rmsd_std)
                eq_time = dr.frame_times_ns[eq_idx] if eq_idx > 0 else dr.frame_times_ns[0]
                insight_parts.append(f"RMSD stabilizes at ~{eq_time:.0f} ns (mean = {rmsd_mean:.1f} +/- {rmsd_std:.1f} Ang).")
        if dr.rmsf is not None and len(dr.rmsf) > 0 and dr.rmsf_residue_ids is not None:
            rmsf_thresh = np.mean(dr.rmsf) + 2 * np.std(dr.rmsf)
            flexible = dr.rmsf_residue_ids[dr.rmsf > rmsf_thresh]
            if len(flexible) > 0:
                regions = _group_consecutive_residues(flexible.tolist())
                strs = [f"res {r[0]}-{r[-1]}" if len(r) > 1 else f"res {r[0]}" for r in regions[:4]]
                insight_parts.append(f"RMSF peaks at flexible regions: {', '.join(strs)}.")
        if insight_parts:
            report.add_insight_box(" ".join(insight_parts), label="Dynamics Insight")

        # ── 3. Interaction Frequency Map ────────────────────────────
        report.add_section(
            "Interaction Frequency Map",
            "Per-residue interaction frequency across the trajectory.",
            figures=[
                self.dashboard.interaction_frequency_heatmap(profile),
                self.dashboard.interaction_timeline(profile),
            ],
        )

        # Interaction insight with badges
        key_res = self.interaction_analyzer.get_key_residues(min_frequency=0.3)
        if key_res:
            badge_map = {
                "hbond": "badge-hbond", "hydrophobic": "badge-hydrophobic",
                "salt_bridge": "badge-ionic", "pi_stack": "badge-pi",
                "water_bridge": "badge-water",
            }
            parts = []
            for res, freqs in key_res[:8]:
                dom = max(freqs, key=freqs.get)
                badge = badge_map.get(dom, "badge-hbond")
                pct = max(freqs.values()) * 100
                parts.append(f'<span class="badge {badge}">{dom.replace("_"," ").title()}</span> {res} ({pct:.0f}%)')
            report.add_insight_box(
                f"{len(key_res)} residues with >30% contact frequency: " + ", ".join(parts),
                label="Key Residues"
            )

        # ── 4. Interaction Network ──────────────────────────────────
        network_viz = InteractionNetworkViz(profile, min_frequency=0.15)
        report.add_section(
            "Interaction Network",
            "Graph-based view of protein-ligand contacts weighted by frequency.",
            figures=[network_viz.plot()],
        )
        report.add_insight_box(
            "Node size reflects interaction count, edge thickness reflects occupancy. "
            "Residues with high betweenness centrality serve as critical hub contacts.",
            label="Network Analysis"
        )

        # ── 5. Energy Decomposition ─────────────────────────────────
        report.add_section(
            "Per-Residue Energy Decomposition",
            "Empirical scoring of each residue's energetic contribution to binding.",
            figures=[self.dashboard.energy_decomposition_bar(self.energy_decomposer)],
        )

        hotspots = self.energy_decomposer.get_hotspot_residues(top_n=10)
        if hotspots:
            rows_data = self.energy_decomposer.per_residue_breakdown()
            rows_sorted = sorted(rows_data, key=lambda x: x["total"])[:10]
            itypes = ["hbond", "hydrophobic", "salt_bridge", "pi_stack", "water_bridge"]
            table_rows = []
            for rank, row in enumerate(rows_sorted, 1):
                dom = min(itypes, key=lambda t: row.get(t, 0.0))
                table_rows.append([str(rank), row["residue"], f"{row['total']:.1f}", dom.replace("_", " ").title()])
            report.add_data_table(["Rank", "Residue", "Energy (kcal/mol)", "Dominant Interaction"], table_rows)

            if len(hotspots) >= 2:
                top5_e = sum(e for _, e in hotspots[:5])
                total_e = sum(e for _, e in hotspots)
                pct = (top5_e / total_e * 100) if total_e != 0 else 0
                report.add_insight_box(
                    f"Top 5 residues account for {pct:.0f}% of total binding energy. "
                    f"Primary anchor: {hotspots[0][0]} ({hotspots[0][1]:.1f} kcal/mol).",
                    label="Energy Hotspots"
                )

        # ── 6. Interaction Fingerprints ─────────────────────────────
        report.add_section(
            "Interaction Fingerprint Evolution",
            "Binary interaction patterns over the trajectory timeline.",
            figures=[
                TemporalViz.fingerprint_heatmap(self.fingerprinter),
                TemporalViz.interaction_persistence_gantt(self.interaction_analyzer),
            ],
        )
        report.add_insight_box(
            "Dark bands show persistent contacts. Vertical transitions in the heatmap "
            "indicate binding mode changes. The Gantt chart below shows temporal presence of key residue contacts.",
            label="Temporal Pattern"
        )

        # ── 7. PCA ──────────────────────────────────────────────────
        cluster_labels = self.clusterer.best.labels if self.clusterer._best_result else None
        if dr.pca and dr.pca.projections.size > 0:
            report.add_section(
                "Binding Site PCA",
                "Principal component analysis of binding site dynamics.",
                figures=[self.dashboard.pca_scatter(dr, cluster_labels)],
            )
            ve = dr.pca.variance_explained
            cum = dr.pca.cumulative_variance
            n_show = min(3, len(ve))
            pca_table = "<table style='max-width:500px;'>"
            for pc_i in range(n_show):
                pca_table += f"<tr><td>PC{pc_i+1} Variance Explained</td><td>{ve[pc_i]:.1%}</td></tr>"
            pca_table += f"<tr><td>Cumulative ({n_show} PCs)</td><td>{cum[n_show-1]:.1%}</td></tr></table>"
            report.add_html_content(pca_table)
            report.add_insight_box(
                "Points colored by cluster show whether conformational states "
                "separate in the reduced PCA space.",
                label="PCA Insight"
            )

        # ── 8. Cross-Correlation ────────────────────────────────────
        if dr.cross_correlation is not None:
            report.add_section(
                "Dynamic Cross-Correlation Matrix",
                "Correlated motions between residues reveal allosteric pathways.",
                figures=[self.dashboard.cross_correlation_matrix(dr)],
            )
            report.add_insight_box(
                "Red = correlated motions (residues move together), blue = anti-correlated. "
                "Strong off-diagonal signals may indicate allosteric communication.",
                label="Allosteric Signal"
            )

        # ── 9. Conformational Clustering ────────────────────────────
        if self.clusterer._best_result:
            cl = self.clusterer.best
            cluster_cards = [
                {"value": str(cl.n_clusters), "label": "Conformational States"},
                {"value": cl.algorithm.upper(), "label": "Best Algorithm"},
                {"value": f"{cl.silhouette_score:.3f}", "label": "Silhouette Score"},
            ]
            report.add_stat_cards(
                "Conformational State Clustering", cluster_cards,
                description=f"AI-driven identification of {cl.n_clusters} distinct binding modes via {cl.algorithm}.",
            )

            cluster_figs = [self.dashboard.cluster_pie_chart(cl)]
            timeline_fig = self.dashboard.state_timeline(cl)
            if timeline_fig.data:
                cluster_figs.append(timeline_fig)
            if cl.transition_matrix is not None:
                cluster_figs.append(self.dashboard.transition_matrix_heatmap(cl.transition_matrix))
            report.add_figures_to_last_section(cluster_figs)

            # State population table
            total_frames = sum(cl.cluster_populations.values())
            ta = self.transition_detector._analysis if self.transition_detector else None
            table_rows = []
            for state in sorted(cl.cluster_populations.keys()):
                pop = cl.cluster_populations[state]
                pop_pct = pop / total_frames * 100 if total_frames > 0 else 0
                dwell_str = "N/A"
                if ta and ta.dwell_times_per_state and state in ta.dwell_times_per_state:
                    dwells = ta.dwell_times_per_state[state]
                    if dwells:
                        dwell_str = f"{np.mean(dwells):.1f} ns"
                table_rows.append([f"State {state}", f"{pop_pct:.1f}%", dwell_str])
            report.add_data_table(["State", "Population", "Mean Dwell Time"], table_rows)

            if ta and ta.n_transitions > 0:
                report.add_insight_box(
                    f"{ta.n_transitions} state transitions detected. "
                    f"Mean dwell time: {ta.mean_dwell_time_ns:.1f} ns.",
                    label="Transition Analysis"
                )

        # ── 10. Feature Importance ──────────────────────────────────
        if self.feature_analyzer and self.feature_analyzer._result:
            fr = self.feature_analyzer.result
            report.add_section(
                "Binding Mode Discriminative Features",
                "Which interactions distinguish between conformational states.",
                figures=[self.dashboard.feature_importance_bar(fr)],
            )
            if fr.top_features:
                table_rows = [[str(i+1), f, f"{s:.3f}"] for i, (f, s) in enumerate(fr.top_features[:10])]
                report.add_data_table(["Rank", "Feature", "Importance Score"], table_rows)
                report.add_insight_box(
                    f"RF cross-validated accuracy = {fr.cv_accuracy:.2f}. "
                    f"Top discriminative feature: '{fr.top_features[0][0]}'.",
                    label="Discriminative Insight"
                )

        # ── 11. Anomalies ───────────────────────────────────────────
        if self.anomaly_detector._result:
            ar = self.anomaly_detector.result
            anomaly_cards = [
                {"value": str(ar.n_anomalies), "label": "Anomalous Frames", "color": "#e67e22"},
                {"value": f"{ar.anomaly_fraction:.1%}", "label": "Anomaly Fraction", "color": "#e67e22"},
            ]
            if ar.rare_interactions:
                anomaly_cards.append({"value": str(len(ar.rare_interactions)), "label": "Rare Interactions", "color": "#e67e22"})
            report.add_stat_cards("Trajectory Anomalies", anomaly_cards,
                                  description="Unusual frames detected by Isolation Forest + LOF consensus.")
            report.add_figures_to_last_section([self.dashboard.anomaly_timeline(ar)])

            for rare in ar.rare_interactions[:5]:
                frames = ", ".join(str(f) for f in rare["frames"][:5])
                report.add_insight_box(
                    f"'{rare['interaction']}' &mdash; {rare['n_frames']} frames "
                    f"({rare['frequency']:.1%} occupancy). Frames: {frames}.",
                    box_type="anomaly", label="Rare Interaction"
                )

        # ── 12. Autocorrelation ─────────────────────────────────────
        try:
            report.add_section(
                "Interaction Decorrelation",
                "How quickly do interaction patterns change.",
                figures=[TemporalViz.autocorrelation_plot(self.fingerprinter)],
            )
            autocorr = self.fingerprinter.temporal_autocorrelation()
            half_idx = int(np.argmax(autocorr < 0.5)) if np.any(autocorr < 0.5) else len(autocorr)
            total_time = meta.total_time_ns
            dt = total_time / max(1, meta.n_frames)
            decorr_time = half_idx * dt
            eff_samples = int(total_time / decorr_time) if decorr_time > 0 else meta.n_frames

            decorr_table = "<table style='max-width:500px;'>"
            decorr_table += f"<tr><td>Decorrelation Time</td><td>{decorr_time:.1f} ns</td></tr>"
            decorr_table += f"<tr><td>Effective Independent Samples</td><td>~{eff_samples}</td></tr></table>"
            report.add_html_content(decorr_table)
            report.add_insight_box(
                f"With a decorrelation time of {decorr_time:.1f} ns and {total_time:.0f} ns total, "
                f"there are ~{eff_samples} effectively independent samples. "
                + ("Sufficient for reliable population estimates." if eff_samples >= 20
                   else "Consider extending the simulation for better statistics."),
                label="Sampling Assessment"
            )
        except Exception:
            pass

        # ── 13. Executive Summary ───────────────────────────────────
        report.add_section("Executive Summary", description="")

        # System + Metrics grid
        sys_items = [
            f"{ss['n_protein_residues']} protein residues, ligand = {', '.join(ss['ligand_resnames'])}",
            f"{ss['n_binding_site_residues']} binding site residues ({ss['binding_site_cutoff']} Ang cutoff)",
            f"{meta.n_frames:,} frames over {meta.total_time_ns:.0f} ns",
        ]
        met_items = []
        if dr.rmsd is not None:
            met_items.append(f"Mean RMSD: {np.mean(dr.rmsd):.1f} Ang")
        if dr.entropy_estimate:
            met_items.append(f"Configurational entropy: {dr.entropy_estimate:.1f} J/(mol*K)")
        if self.anomaly_detector._result:
            ar = self.anomaly_detector.result
            met_items.append(f"Anomalies: {ar.n_anomalies} frames ({ar.anomaly_fraction:.1%})")

        grid = '<div class="grid-2" style="margin-bottom:1.5em;">'
        grid += '<div><h3 style="color:#2c3e50;margin-bottom:0.5em;">System</h3><ul style="list-style:none;padding:0;">'
        for i in sys_items:
            grid += f"<li>{i}</li>"
        grid += '</ul></div><div><h3 style="color:#2c3e50;margin-bottom:0.5em;">Key Metrics</h3><ul style="list-style:none;padding:0;">'
        for i in met_items:
            grid += f"<li>{i}</li>"
        grid += '</ul></div></div>'
        report.add_html_content(grid)

        # Key Findings
        findings = []
        if self.clusterer._best_result:
            cl = self.clusterer.best
            findings.append(
                f"<strong>{cl.n_clusters} distinct binding modes</strong> identified by "
                f"{cl.algorithm} (silhouette = {cl.silhouette_score:.3f})."
            )
        if hotspots:
            findings.append(
                f"<strong>{hotspots[0][0]} is the energetic anchor</strong> "
                f"({hotspots[0][1]:.1f} kcal/mol)."
            )
        if self.feature_analyzer and self.feature_analyzer._result and self.feature_analyzer.result.top_features:
            tf = self.feature_analyzer.result.top_features[0]
            findings.append(f"<strong>{tf[0]}</strong> is the top discriminative interaction between states.")
        if findings:
            html = '<h3 style="color:#2c3e50;margin:1em 0 0.5em;">Key Findings</h3><ol style="padding-left:1.2em;">'
            for f in findings:
                html += f"<li>{f}</li>"
            html += "</ol>"
            report.add_html_content(html)

        # Interaction Summary with badges
        badge_map = {
            "hbond": ("badge-hbond", "Hydrogen Bonds"),
            "hydrophobic": ("badge-hydrophobic", "Hydrophobic"),
            "salt_bridge": ("badge-ionic", "Salt Bridges"),
            "pi_stack": ("badge-pi", "Pi-Stacking"),
            "water_bridge": ("badge-water", "Water Bridges"),
        }
        int_rows_html = ""
        for itype, count in profile.interaction_summary.items():
            bcls, label = badge_map.get(itype, ("badge-hbond", itype.replace("_", " ").title()))
            int_rows_html += (
                f'<tr><td><span class="badge {bcls}">{label}</span></td>'
                f'<td>{count:,} total events</td></tr>'
            )
        if int_rows_html:
            report.add_html_content(
                '<h3 style="color:#2c3e50;margin:1em 0 0.5em;">Interaction Summary</h3>'
                f'<table style="max-width:600px;">{int_rows_html}</table>'
            )

        path = report.generate()
        logger.info(f"\n📄 Report generated: {path}")
        return path

    def get_executive_summary(self) -> Dict:
        """Return structured executive summary as a dictionary."""
        if not self._analysis_complete:
            raise RuntimeError("Run run_deep_dive() first.")

        ss = self.system.system_summary()
        profile = self.interaction_analyzer.profile
        dr = self.dynamics_analyzer.result

        summary = {
            "system": ss,
            "trajectory": {
                "n_frames": self.loader.metadata.n_frames,
                "total_time_ns": self.loader.metadata.total_time_ns,
            },
            "interactions": profile.interaction_summary,
            "key_residues": [
                {"residue": r, "frequencies": f}
                for r, f in self.interaction_analyzer.get_key_residues(0.3)
            ],
            "energy_hotspots": [
                {"residue": r, "energy_kcal": e}
                for r, e in self.energy_decomposer.get_hotspot_residues(10)
            ],
            "conformational_states": (
                self.clusterer.best.n_clusters if self.clusterer._best_result else 0
            ),
            "entropy_J_per_mol_K": dr.entropy_estimate,
            "n_anomalies": (
                self.anomaly_detector.result.n_anomalies
                if self.anomaly_detector._result else 0
            ),
        }

        return summary
