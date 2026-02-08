"""
MDInsight - AI-Powered Deep Dive Analysis for Molecular Dynamics Simulations
=============================================================================

A unified Python toolkit that reads MD trajectory files (XTC, GRO, TPR, TRR, DCD)
and performs comprehensive protein-ligand interaction analysis powered by machine
learning — going far beyond traditional RMSD/RMSF.

Example:
    >>> from mdinsight import MDInsight
    >>> engine = MDInsight("topology.gro", "trajectory.xtc", ligand_selection="resname LIG")
    >>> engine.run_deep_dive()
    >>> engine.generate_report("my_analysis.html")
"""

__version__ = "0.1.0"
__author__ = "MDInsight Contributors"
