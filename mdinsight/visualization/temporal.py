"""
Temporal Visualization - Time-resolved interaction pattern views.
"""

import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    raise ImportError("plotly is required.")


class TemporalViz:
    """Temporal visualization utilities for interaction data."""

    @staticmethod
    def interaction_persistence_gantt(interaction_analyzer, top_n: int = 15) -> go.Figure:
        """
        Gantt-style chart showing when each key residue is interacting
        with the ligand over the trajectory timeline.
        """
        profile = interaction_analyzer.profile
        freq = profile.per_residue_frequency

        # Top residues by frequency
        ranked = sorted(freq.items(), key=lambda x: max(x[1].values()), reverse=True)
        top_residues = [r[0] for r in ranked[:top_n]]

        # Build frame presence per residue
        from collections import defaultdict
        residue_frames = defaultdict(set)
        for ev in profile.events:
            if ev.protein_residue in top_residues:
                residue_frames[ev.protein_residue].add(ev.frame)

        fig = go.Figure()

        for i, residue in enumerate(top_residues):
            frames = sorted(residue_frames.get(residue, []))
            if not frames:
                continue

            # Convert to segments for plotting
            segments = []
            start = frames[0]
            prev = frames[0]
            for f in frames[1:]:
                if f - prev > 2:  # gap
                    segments.append((start, prev))
                    start = f
                prev = f
            segments.append((start, prev))

            for s, e in segments:
                fig.add_trace(go.Scatter(
                    x=[s, e], y=[i, i],
                    mode="lines",
                    line=dict(width=8, color=f"hsl({i * 24 % 360}, 70%, 50%)"),
                    showlegend=False,
                    hovertext=f"{residue}: frames {s}-{e}",
                    hoverinfo="text",
                ))

        fig.update_layout(
            title="Interaction Persistence Timeline",
            xaxis_title="Frame",
            yaxis=dict(
                tickvals=list(range(len(top_residues))),
                ticktext=top_residues,
            ),
            height=max(300, 30 * len(top_residues)),
            template="plotly_white",
        )

        return fig

    @staticmethod
    def fingerprint_heatmap(fingerprinter, max_frames: int = 500) -> go.Figure:
        """
        Heatmap of interaction fingerprints over time.

        Rows = fingerprint bits (residue:type), Columns = frames.
        """
        fps = fingerprinter.fingerprints
        labels = fingerprinter.bit_labels

        # Subsample if too many frames
        if fps.shape[0] > max_frames:
            step = fps.shape[0] // max_frames
            fps = fps[::step]

        # Only show active bits
        active = fps.sum(axis=0) > 0
        fps_active = fps[:, active]
        labels_active = [l for l, a in zip(labels, active) if a]

        fig = go.Figure(data=go.Heatmap(
            z=fps_active.T,
            x=list(range(fps_active.shape[0])),
            y=labels_active,
            colorscale=[[0, "white"], [1, "#2c3e50"]],
            showscale=False,
        ))

        fig.update_layout(
            title="Interaction Fingerprint Evolution",
            xaxis_title="Frame",
            yaxis_title="Interaction",
            height=max(400, 20 * len(labels_active)),
            template="plotly_white",
        )

        return fig

    @staticmethod
    def autocorrelation_plot(fingerprinter) -> go.Figure:
        """Plot fingerprint temporal autocorrelation."""
        autocorr = fingerprinter.temporal_autocorrelation()

        fig = go.Figure(data=go.Scatter(
            x=np.arange(len(autocorr)),
            y=autocorr,
            mode="lines",
            line=dict(color="#2c3e50", width=2),
        ))

        # Add decorrelation time marker
        half_idx = np.argmax(autocorr < 0.5)
        if half_idx > 0:
            fig.add_vline(
                x=half_idx, line_dash="dash",
                annotation_text=f"τ½ = {half_idx} frames",
            )

        fig.update_layout(
            title="Interaction Pattern Autocorrelation",
            xaxis_title="Lag (frames)",
            yaxis_title="Autocorrelation",
            height=350,
            template="plotly_white",
        )

        return fig
