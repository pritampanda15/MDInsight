"""
Interaction Network Visualization - Graph-based views of protein-ligand contacts.

Creates network graphs where nodes are protein residues and the ligand,
edges represent interactions weighted by frequency, and community detection
reveals allosteric communication pathways.
"""

import logging
from typing import Optional, Dict, List, Tuple

import numpy as np

try:
    import networkx as nx
except ImportError:
    nx = None

try:
    import plotly.graph_objects as go
except ImportError:
    raise ImportError("plotly is required.")

logger = logging.getLogger(__name__)

INTERACTION_COLORS = {
    "hbond": "#e74c3c",
    "hydrophobic": "#2ecc71",
    "salt_bridge": "#3498db",
    "pi_stack": "#9b59b6",
    "water_bridge": "#1abc9c",
}


class InteractionNetworkViz:
    """
    Build and visualize protein-ligand interaction networks.

    Parameters
    ----------
    interaction_profile : InteractionProfile
        Completed interaction analysis.
    min_frequency : float, default 0.1
        Minimum interaction frequency to include an edge.
    """

    def __init__(self, interaction_profile, min_frequency: float = 0.1):
        self.profile = interaction_profile
        self.min_frequency = min_frequency
        self._graph: Optional[object] = None

    def build_graph(self):
        """Build interaction network graph."""
        if nx is None:
            logger.warning("networkx not installed — skipping network analysis.")
            return None

        G = nx.Graph()
        G.add_node("LIGAND", node_type="ligand")

        freq = self.profile.per_residue_frequency

        for residue, type_freqs in freq.items():
            max_freq = max(type_freqs.values())
            if max_freq < self.min_frequency:
                continue

            G.add_node(residue, node_type="residue")

            for itype, f in type_freqs.items():
                if f >= self.min_frequency:
                    # Add or update edge
                    if G.has_edge(residue, "LIGAND"):
                        edge_data = G[residue]["LIGAND"]
                        edge_data["weight"] += f
                        edge_data["types"].append(itype)
                    else:
                        G.add_edge(
                            residue, "LIGAND",
                            weight=f,
                            types=[itype],
                            primary_type=itype,
                        )

        self._graph = G
        logger.info(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G

    def plot(self) -> go.Figure:
        """Create interactive Plotly network visualization."""
        G = self._graph or self.build_graph()
        if G is None or G.number_of_nodes() == 0:
            return go.Figure()

        # Spring layout
        pos = nx.spring_layout(G, k=2, seed=42)

        # Edge traces
        edge_traces = []
        for u, v, data in G.edges(data=True):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            color = INTERACTION_COLORS.get(data.get("primary_type", ""), "#888")
            width = max(1, data["weight"] * 5)

            edge_traces.append(go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                mode="lines",
                line=dict(width=width, color=color),
                hoverinfo="text",
                text=f"{u} ↔ {v}<br>Weight: {data['weight']:.2f}<br>Types: {', '.join(data['types'])}",
                showlegend=False,
            ))

        # Node trace
        node_x, node_y, node_text, node_size, node_color = [], [], [], [], []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

            if node == "LIGAND":
                node_text.append("LIGAND")
                node_size.append(30)
                node_color.append("#e74c3c")
            else:
                degree = G.degree(node, weight="weight")
                node_text.append(f"{node}<br>Degree: {degree:.2f}")
                node_size.append(max(10, degree * 15))
                node_color.append("#3498db")

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode="markers+text",
            marker=dict(size=node_size, color=node_color, line=dict(width=1, color="white")),
            text=[n for n in G.nodes()],
            textposition="top center",
            textfont=dict(size=8),
            hovertext=node_text,
            hoverinfo="text",
        )

        fig = go.Figure(data=[*edge_traces, node_trace])
        fig.update_layout(
            title="Protein-Ligand Interaction Network",
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600,
            template="plotly_white",
        )

        return fig

    def get_centrality_ranking(self) -> List[Tuple[str, float]]:
        """Rank residues by betweenness centrality in interaction network."""
        G = self._graph or self.build_graph()
        if G is None:
            return []

        centrality = nx.betweenness_centrality(G, weight="weight")
        ranked = sorted(centrality.items(), key=lambda x: -x[1])
        return [(node, cent) for node, cent in ranked if node != "LIGAND"]
