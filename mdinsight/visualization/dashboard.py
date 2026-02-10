"""
Dashboard Generator - Interactive Plotly visualizations for MDInsight analyses.

Creates comprehensive multi-panel HTML dashboards with linked interactivity.
"""

import logging
from typing import Optional, Dict, List

import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
except ImportError:
    raise ImportError("plotly is required: pip install plotly")

logger = logging.getLogger(__name__)

# Color palette
COLORS = {
    "hbond": "#e74c3c",
    "hydrophobic": "#2ecc71",
    "salt_bridge": "#3498db",
    "pi_stack": "#9b59b6",
    "water_bridge": "#1abc9c",
    "primary": "#2c3e50",
    "secondary": "#7f8c8d",
    "bg": "#fafafa",
}


class DashboardGenerator:
    """
    Generate interactive Plotly dashboard panels.

    Each method returns a plotly Figure that can be displayed or saved.
    """

    def __init__(self):
        self.template = "plotly_white"

    def dynamics_overview(self, dynamics_result) -> go.Figure:
        """Multi-panel dynamics overview: RMSD, RMSF, RoG, ligand RMSD."""
        dr = dynamics_result
        t = dr.frame_times_ns

        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                "Protein Backbone RMSD", "Ligand RMSD",
                "Per-Residue RMSF", "Radius of Gyration",
                "Protein-Ligand COM Distance", "Binding Site RMSF"
            ),
            vertical_spacing=0.08,
        )

        # RMSD
        if dr.rmsd is not None:
            fig.add_trace(
                go.Scatter(x=t, y=dr.rmsd, mode="lines", name="Backbone RMSD",
                           line=dict(color=COLORS["primary"], width=1)),
                row=1, col=1,
            )

        # Ligand RMSD
        if dr.ligand_rmsd is not None:
            fig.add_trace(
                go.Scatter(x=t, y=dr.ligand_rmsd, mode="lines", name="Ligand RMSD",
                           line=dict(color=COLORS["hbond"], width=1)),
                row=1, col=2,
            )

        # RMSF
        if dr.rmsf is not None:
            fig.add_trace(
                go.Bar(x=dr.rmsf_residue_ids, y=dr.rmsf, name="RMSF",
                       marker_color=COLORS["hydrophobic"]),
                row=2, col=1,
            )

        # Radius of Gyration
        if dr.radius_of_gyration is not None:
            fig.add_trace(
                go.Scatter(x=t, y=dr.radius_of_gyration, mode="lines", name="RoG",
                           line=dict(color=COLORS["salt_bridge"], width=1)),
                row=2, col=2,
            )

        # P-L distance
        if dr.protein_ligand_distance is not None:
            fig.add_trace(
                go.Scatter(x=t, y=dr.protein_ligand_distance, mode="lines",
                           name="P-L Distance",
                           line=dict(color=COLORS["pi_stack"], width=1)),
                row=3, col=1,
            )

        # Binding site RMSF
        if dr.binding_site_rmsf is not None and len(dr.binding_site_rmsf) > 0:
            fig.add_trace(
                go.Bar(x=dr.binding_site_residue_ids, y=dr.binding_site_rmsf,
                       name="BS RMSF", marker_color=COLORS["water_bridge"]),
                row=3, col=2,
            )

        fig.update_layout(
            height=900, title_text="Dynamics Overview",
            showlegend=False, template=self.template,
        )
        fig.update_xaxes(title_text="Time (ns)", row=1, col=1)
        fig.update_xaxes(title_text="Time (ns)", row=1, col=2)
        fig.update_xaxes(title_text="Residue ID", row=2, col=1)
        fig.update_xaxes(title_text="Time (ns)", row=2, col=2)
        fig.update_yaxes(title_text="RMSD (Å)", row=1, col=1)
        fig.update_yaxes(title_text="RMSF (Å)", row=2, col=1)

        return fig

    # ── Individual dynamics plots ──────────────────────────────────────

    def rmsd_plot(self, dynamics_result) -> go.Figure:
        """Standalone RMSD plot with ligand distance overlay."""
        dr = dynamics_result
        t = dr.frame_times_ns
        fig = go.Figure()
        if dr.rmsd is not None:
            fig.add_trace(go.Scatter(
                x=t, y=dr.rmsd, mode="lines", name="Backbone RMSD",
                line=dict(color=COLORS["primary"], width=1.5),
            ))
        if dr.protein_ligand_distance is not None:
            fig.add_trace(go.Scatter(
                x=t, y=dr.protein_ligand_distance, mode="lines",
                name="Ligand-Protein Distance",
                line=dict(color=COLORS["hbond"], width=1.5), yaxis="y2",
            ))
        fig.update_layout(
            title="RMSD & Ligand-Protein Distance",
            xaxis_title="Time (ns)",
            yaxis=dict(title="RMSD (Ang)", side="left"),
            yaxis2=dict(title="Distance (Ang)", overlaying="y", side="right", showgrid=False),
            height=380, template=self.template,
            legend=dict(x=0.02, y=0.98),
        )
        return fig

    def rmsf_plot(self, dynamics_result) -> go.Figure:
        """Standalone per-residue RMSF plot with fill."""
        dr = dynamics_result
        if dr.rmsf is None:
            return go.Figure()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dr.rmsf_residue_ids, y=dr.rmsf, mode="lines", name="RMSF",
            line=dict(color=COLORS["primary"], width=1.5),
            fill="tozeroy", fillcolor="rgba(52,152,219,0.15)",
        ))
        fig.update_layout(
            title="Per-Residue RMSF",
            xaxis_title="Residue Number", yaxis_title="RMSF (Ang)",
            height=380, template=self.template,
        )
        return fig

    def binding_site_rmsf_plot(self, dynamics_result) -> go.Figure:
        """Standalone binding site RMSF bar chart."""
        dr = dynamics_result
        if dr.binding_site_rmsf is None or len(dr.binding_site_rmsf) == 0:
            return go.Figure()
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=dr.binding_site_residue_ids, y=dr.binding_site_rmsf,
            name="Binding Site RMSF", marker_color=COLORS["water_bridge"],
        ))
        fig.update_layout(
            title="Binding Site Residue RMSF",
            xaxis_title="Residue ID", yaxis_title="RMSF (Ang)",
            height=380, template=self.template,
        )
        return fig

    def rog_plot(self, dynamics_result) -> go.Figure:
        """Standalone radius of gyration plot."""
        dr = dynamics_result
        if dr.radius_of_gyration is None:
            return go.Figure()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dr.frame_times_ns, y=dr.radius_of_gyration, mode="lines",
            name="Radius of Gyration",
            line=dict(color=COLORS["salt_bridge"], width=1.5),
        ))
        fig.update_layout(
            title="Radius of Gyration",
            xaxis_title="Time (ns)", yaxis_title="Rg (Ang)",
            height=350, template=self.template,
        )
        return fig

    # ── Individual clustering plots ─────────────────────────────────

    def state_timeline(self, cluster_result) -> go.Figure:
        """Standalone state timeline scatter plot."""
        labels = cluster_result.labels
        pops = cluster_result.cluster_populations
        if len(labels) == 0:
            return go.Figure()

        state_colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12",
                        "#1abc9c", "#e67e22", "#34495e"]
        fig = go.Figure()
        for state in sorted(set(labels)):
            if state < 0:
                continue
            mask = labels == state
            frames = np.where(mask)[0]
            color = state_colors[state % len(state_colors)]
            pop_pct = pops.get(state, 0) / len(labels) * 100
            fig.add_trace(go.Scatter(
                x=frames, y=labels[mask], mode="markers",
                marker=dict(size=4, color=color),
                name=f"State {state} ({pop_pct:.1f}%)",
            ))
        fig.update_layout(
            title="Conformational State Timeline",
            xaxis_title="Frame", yaxis_title="State",
            yaxis=dict(dtick=1), height=320, template=self.template,
        )
        return fig

    def cluster_pie_chart(self, cluster_result) -> go.Figure:
        """Standalone cluster population donut chart."""
        pops = cluster_result.cluster_populations
        if not pops:
            return go.Figure()
        state_colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12",
                        "#1abc9c", "#e67e22", "#34495e"]
        labels_list = [f"State {k}" for k in sorted(pops.keys())]
        values = [pops[k] for k in sorted(pops.keys())]
        colors = [state_colors[k % len(state_colors)] for k in sorted(pops.keys())]
        fig = go.Figure(data=go.Pie(
            labels=labels_list, values=values, hole=0.4,
            marker=dict(colors=colors),
            textinfo="label+percent", textposition="inside",
        ))
        fig.update_layout(
            title="Conformational State Populations",
            height=400, template=self.template,
        )
        return fig

    # ── Interaction persistence ─────────────────────────────────────

    def interaction_persistence_heatmap(self, interaction_analyzer, top_n: int = 20) -> go.Figure:
        """Heatmap of per-residue interaction presence across time bins."""
        from collections import defaultdict

        profile = interaction_analyzer.profile
        freq = profile.per_residue_frequency
        if not freq:
            return go.Figure()

        ranked = sorted(freq.items(), key=lambda x: max(x[1].values()), reverse=True)
        top_residues = [r[0] for r in ranked[:top_n]]

        # Collect time points per residue
        residue_times = defaultdict(list)
        for ev in profile.events:
            if ev.protein_residue in top_residues:
                residue_times[ev.protein_residue].append(ev.time_ns)

        if not residue_times:
            return go.Figure()

        # Determine time bins
        all_times = sorted(set(t for times in residue_times.values() for t in times))
        t_min, t_max = all_times[0], all_times[-1]
        n_bins = min(100, len(all_times))
        bin_edges = np.linspace(t_min, t_max + 1e-9, n_bins + 1)

        # Build occupancy matrix
        z = np.zeros((len(top_residues), n_bins))
        for i, residue in enumerate(top_residues):
            times = np.array(residue_times.get(residue, []))
            if len(times) == 0:
                continue
            hist, _ = np.histogram(times, bins=bin_edges)
            z[i] = (hist > 0).astype(float)

        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

        fig = go.Figure(data=go.Heatmap(
            z=z, x=np.round(bin_centers, 1), y=top_residues,
            colorscale=[[0, "white"], [1, "#2c3e50"]],
            showscale=False,
            hovertemplate="Residue: %{y}<br>Time: %{x:.1f} ns<br>Present: %{z}<extra></extra>",
        ))

        fig.update_layout(
            title="Interaction Persistence Heatmap",
            xaxis_title="Time (ns)", yaxis_title="Residue",
            height=max(400, 25 * len(top_residues)),
            template=self.template,
            margin=dict(l=130),
        )

        return fig

    # ── Original methods ────────────────────────────────────────────

    def interaction_frequency_heatmap(self, interaction_profile) -> go.Figure:
        """Per-residue interaction frequency heatmap."""
        freq = interaction_profile.per_residue_frequency
        if not freq:
            return go.Figure()

        # Sort residues by total frequency
        residues = sorted(
            freq.keys(),
            key=lambda r: sum(freq[r].values()),
            reverse=True,
        )[:30]  # Top 30

        itypes = ["hbond", "hydrophobic", "salt_bridge", "pi_stack", "water_bridge"]

        z = []
        for itype in itypes:
            row = [freq.get(res, {}).get(itype, 0.0) for res in residues]
            z.append(row)

        fig = go.Figure(data=go.Heatmap(
            z=z,
            x=residues,
            y=[t.replace("_", " ").title() for t in itypes],
            colorscale="RdBu_r",
            zmin=0, zmax=1,
            text=[[f"{v:.2f}" for v in row] for row in z],
            texttemplate="%{text}",
            hovertemplate="Residue: %{x}<br>Type: %{y}<br>Frequency: %{z:.2f}<extra></extra>",
        ))

        fig.update_layout(
            title="Per-Residue Interaction Frequency",
            xaxis_title="Residue",
            yaxis_title="Interaction Type",
            height=400,
            template=self.template,
        )

        return fig

    def interaction_timeline(self, interaction_profile) -> go.Figure:
        """Temporal interaction count plot."""
        counts = interaction_profile.per_frame_counts
        if not counts:
            return go.Figure()

        fig = go.Figure()

        for itype, values in counts.items():
            x = np.arange(len(values))
            # Smooth for visibility
            if len(values) > 100:
                kernel = np.ones(20) / 20
                smoothed = np.convolve(values, kernel, mode="same")
            else:
                smoothed = values

            fig.add_trace(go.Scatter(
                x=x, y=smoothed, mode="lines",
                name=itype.replace("_", " ").title(),
                line=dict(color=COLORS.get(itype, "#888"), width=2),
                fill="tozeroy",
                opacity=0.7,
            ))

        fig.update_layout(
            title="Interaction Counts Over Trajectory",
            xaxis_title="Frame",
            yaxis_title="Count",
            height=400,
            template=self.template,
        )

        return fig

    def pca_scatter(self, dynamics_result, cluster_labels=None) -> go.Figure:
        """PCA projection scatter plot, optionally colored by cluster."""
        pca = dynamics_result.pca
        if pca is None or pca.projections.size == 0:
            return go.Figure()

        proj = pca.projections
        n_pc = min(3, proj.shape[1])

        if cluster_labels is not None and len(cluster_labels) >= proj.shape[0]:
            colors = cluster_labels[:proj.shape[0]]
        else:
            colors = np.arange(proj.shape[0])

        if n_pc >= 3:
            fig = go.Figure(data=go.Scatter3d(
                x=proj[:, 0], y=proj[:, 1], z=proj[:, 2],
                mode="markers",
                marker=dict(
                    size=2, color=colors,
                    colorscale="Viridis", opacity=0.6,
                    colorbar=dict(title="Cluster" if cluster_labels is not None else "Frame"),
                ),
                text=[f"Frame {i}" for i in range(proj.shape[0])],
            ))
            fig.update_layout(
                title=f"PCA of Binding Site Dynamics "
                      f"({pca.variance_explained[0]:.1%}, "
                      f"{pca.variance_explained[1]:.1%}, "
                      f"{pca.variance_explained[2]:.1%})",
                scene=dict(
                    xaxis_title=f"PC1 ({pca.variance_explained[0]:.1%})",
                    yaxis_title=f"PC2 ({pca.variance_explained[1]:.1%})",
                    zaxis_title=f"PC3 ({pca.variance_explained[2]:.1%})",
                ),
                height=600,
            )
        else:
            fig = go.Figure(data=go.Scatter(
                x=proj[:, 0], y=proj[:, 1] if n_pc > 1 else np.zeros(proj.shape[0]),
                mode="markers",
                marker=dict(
                    size=3, color=colors,
                    colorscale="Viridis", opacity=0.6,
                ),
            ))
            fig.update_layout(
                title="PCA of Binding Site",
                xaxis_title="PC1", yaxis_title="PC2",
                height=500, template=self.template,
            )

        return fig

    def cross_correlation_matrix(self, dynamics_result) -> go.Figure:
        """Dynamic cross-correlation matrix (DCCM) heatmap."""
        dccm = dynamics_result.cross_correlation
        resids = dynamics_result.cross_corr_residue_ids

        if dccm is None:
            return go.Figure()

        fig = go.Figure(data=go.Heatmap(
            z=dccm,
            x=resids,
            y=resids,
            colorscale="RdBu_r",
            zmid=0, zmin=-1, zmax=1,
            colorbar=dict(title="Correlation"),
        ))

        fig.update_layout(
            title="Dynamic Cross-Correlation Matrix (DCCM)",
            xaxis_title="Residue ID",
            yaxis_title="Residue ID",
            height=600, width=700,
            template=self.template,
        )

        return fig

    def energy_decomposition_bar(self, energy_decomposer) -> go.Figure:
        """Per-residue energy decomposition stacked bar chart."""
        rows = energy_decomposer.per_residue_breakdown()
        if not rows:
            return go.Figure()

        # Take top 20 by absolute total
        rows = sorted(rows, key=lambda x: x["total"])[:20]

        residues = [r["residue"] for r in rows]
        itypes = ["hbond", "hydrophobic", "salt_bridge", "pi_stack", "water_bridge"]

        fig = go.Figure()
        for itype in itypes:
            values = [r.get(itype, 0) for r in rows]
            fig.add_trace(go.Bar(
                name=itype.replace("_", " ").title(),
                x=residues, y=values,
                marker_color=COLORS.get(itype, "#888"),
            ))

        fig.update_layout(
            barmode="stack",
            title="Per-Residue Energy Decomposition (Proxy)",
            xaxis_title="Residue",
            yaxis_title="Energy (kcal/mol)",
            height=500,
            template=self.template,
        )

        return fig

    def cluster_populations(self, cluster_result) -> go.Figure:
        """Cluster population pie chart and state timeline."""
        pops = cluster_result.cluster_populations
        if not pops:
            return go.Figure()

        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "pie"}, {"type": "xy"}]],
            subplot_titles=("Cluster Populations", "State Timeline"),
        )

        labels_list = [f"State {k}" for k in sorted(pops.keys())]
        values = [pops[k] for k in sorted(pops.keys())]

        fig.add_trace(
            go.Pie(labels=labels_list, values=values, hole=0.4),
            row=1, col=1,
        )

        # Timeline
        labels = cluster_result.labels
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(labels)), y=labels,
                mode="markers", marker=dict(size=1, color=labels, colorscale="Viridis"),
                name="State",
            ),
            row=1, col=2,
        )

        fig.update_layout(
            height=400,
            title_text="Conformational State Analysis",
            template=self.template,
        )

        return fig

    def transition_matrix_heatmap(self, transition_matrix: np.ndarray) -> go.Figure:
        """State transition probability heatmap."""
        n = transition_matrix.shape[0]
        labels = [f"State {i}" for i in range(n)]

        fig = go.Figure(data=go.Heatmap(
            z=transition_matrix,
            x=labels, y=labels,
            colorscale="Blues",
            text=[[f"{v:.2f}" for v in row] for row in transition_matrix],
            texttemplate="%{text}",
            zmin=0, zmax=1,
        ))

        fig.update_layout(
            title="State Transition Probability Matrix",
            xaxis_title="To State",
            yaxis_title="From State",
            height=400, width=500,
            template=self.template,
        )

        return fig

    def anomaly_timeline(self, anomaly_result) -> go.Figure:
        """Anomaly score timeline with flagged frames."""
        scores = anomaly_result.anomaly_scores
        mask = anomaly_result.anomaly_mask

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=np.arange(len(scores)), y=scores,
            mode="lines", name="Anomaly Score",
            line=dict(color=COLORS["secondary"], width=1),
        ))

        anomaly_idx = np.where(mask)[0]
        if len(anomaly_idx) > 0:
            fig.add_trace(go.Scatter(
                x=anomaly_idx, y=scores[anomaly_idx],
                mode="markers", name="Anomalies",
                marker=dict(color=COLORS["hbond"], size=5, symbol="x"),
            ))

        fig.update_layout(
            title=f"Trajectory Anomalies ({anomaly_result.n_anomalies} detected)",
            xaxis_title="Frame",
            yaxis_title="Anomaly Score",
            height=350,
            template=self.template,
        )

        return fig

    def feature_importance_bar(self, feature_result) -> go.Figure:
        """Top discriminative interaction features."""
        if not feature_result.top_features:
            return go.Figure()

        names = [f[0] for f in feature_result.top_features[:15]]
        rf_vals = [
            feature_result.rf_importance[feature_result.feature_names.index(n)]
            for n in names
        ]

        fig = go.Figure(data=go.Bar(
            y=names, x=rf_vals,
            orientation="h",
            marker_color=COLORS["salt_bridge"],
        ))

        fig.update_layout(
            title=f"Top Discriminative Interactions (CV acc={feature_result.cv_accuracy:.2f})",
            xaxis_title="Feature Importance",
            height=500,
            template=self.template,
            yaxis=dict(autorange="reversed"),
        )

        return fig
