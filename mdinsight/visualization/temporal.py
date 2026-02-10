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

        Uses time (ns) on the x-axis with filled rectangles so
        segments are visible even with large stride values.
        """
        from collections import defaultdict

        profile = interaction_analyzer.profile
        freq = profile.per_residue_frequency

        if not freq:
            fig = go.Figure()
            fig.update_layout(
                title="Interaction Persistence Timeline",
                annotations=[dict(text="No interaction data", x=0.5, y=0.5,
                                  xref="paper", yref="paper", showarrow=False)],
            )
            return fig

        # Top residues by frequency
        ranked = sorted(freq.items(), key=lambda x: max(x[1].values()), reverse=True)
        top_residues = [r[0] for r in ranked[:top_n]]

        # Build time presence per residue and track dominant interaction type
        residue_times = defaultdict(list)
        residue_types = defaultdict(lambda: defaultdict(int))
        for ev in profile.events:
            if ev.protein_residue in top_residues:
                residue_times[ev.protein_residue].append(ev.time_ns)
                residue_types[ev.protein_residue][ev.interaction_type] += 1

        # Auto-detect time step from all event times
        all_times = sorted(set(t for times in residue_times.values() for t in times))
        if len(all_times) > 1:
            diffs = np.diff(all_times)
            dt = float(np.median(diffs[diffs > 0])) if np.any(diffs > 0) else float(diffs[0])
        else:
            dt = 1.0
        gap_threshold = dt * 2.5

        # Color mapping
        type_colors = {
            "hbond": "#27ae60", "hydrophobic": "#e67e22",
            "salt_bridge": "#2980b9", "pi_stack": "#8e44ad",
            "water_bridge": "#1abc9c",
        }

        fig = go.Figure()

        for i, residue in enumerate(top_residues):
            times = sorted(set(residue_times.get(residue, [])))
            if not times:
                continue

            # Dominant interaction type for color
            types = residue_types[residue]
            dom_type = max(types, key=types.get) if types else "hbond"
            color = type_colors.get(dom_type, "#3498db")

            # Build contiguous segments
            segments = []
            seg_start = times[0]
            prev = times[0]
            for t in times[1:]:
                if t - prev > gap_threshold:
                    segments.append((seg_start, prev + dt))
                    seg_start = t
                prev = t
            segments.append((seg_start, prev + dt))

            # Draw filled rectangles
            for s, e in segments:
                fig.add_trace(go.Scatter(
                    x=[s, e, e, s, s],
                    y=[i - 0.35, i - 0.35, i + 0.35, i + 0.35, i - 0.35],
                    fill="toself", fillcolor=color,
                    line=dict(width=0), mode="lines",
                    showlegend=False,
                    hovertext=f"{residue} ({dom_type}): {s:.1f}–{e:.1f} ns",
                    hoverinfo="text",
                ))

        fig.update_layout(
            title="Interaction Persistence Timeline",
            xaxis_title="Time (ns)",
            yaxis=dict(
                tickvals=list(range(len(top_residues))),
                ticktext=top_residues,
            ),
            height=max(400, 40 * len(top_residues)),
            template="plotly_white",
            margin=dict(l=130),
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
