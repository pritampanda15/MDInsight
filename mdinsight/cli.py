"""
MDInsight CLI - Command-line interface for deep dive analysis.

Usage:
    mdinsight run -t system.gro -x trajectory.xtc -l "resname LIG" -o report.html
    mdinsight run -t system.tpr -x run1.xtc run2.xtc --stride 5
"""

import argparse
import sys
import logging


def main():
    parser = argparse.ArgumentParser(
        prog="mdinsight",
        description="MDInsight — AI-Powered Deep Dive Analysis for MD Simulations",
    )
    subparsers = parser.add_subparsers(dest="command")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run deep dive analysis")
    run_parser.add_argument("-t", "--topology", required=True, help="Topology file (GRO/PDB/TPR)")
    run_parser.add_argument("-x", "--trajectory", nargs="+", help="Trajectory file(s) (XTC/TRR/DCD)")
    run_parser.add_argument("-l", "--ligand", default=None, help='Ligand selection (e.g., "resname LIG")')
    run_parser.add_argument("-o", "--output", default="mdinsight_report.html", help="Output report path")
    run_parser.add_argument("--stride", type=int, default=1, help="Frame stride")
    run_parser.add_argument("--cutoff", type=float, default=5.0, help="Binding site cutoff (Å)")
    run_parser.add_argument("--start", type=int, default=0, help="Start frame")
    run_parser.add_argument("--stop", type=int, default=None, help="Stop frame")
    run_parser.add_argument("--in-memory", action="store_true", help="Load trajectory into RAM")
    run_parser.add_argument("--no-pca", action="store_true", help="Skip PCA")
    run_parser.add_argument("--no-dccm", action="store_true", help="Skip cross-correlation")
    run_parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")

    # Info command
    info_parser = subparsers.add_parser("info", help="Show trajectory information")
    info_parser.add_argument("-t", "--topology", required=True)
    info_parser.add_argument("-x", "--trajectory", nargs="+", default=None)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "info":
        from mdinsight.core.trajectory import TrajectoryLoader
        traj_files = args.trajectory[0] if args.trajectory and len(args.trajectory) == 1 else args.trajectory
        loader = TrajectoryLoader(args.topology, traj_files)
        print(loader.summary())

    elif args.command == "run":
        from mdinsight.engine import MDInsight

        traj = args.trajectory
        if traj and len(traj) == 1:
            traj = traj[0]

        engine = MDInsight(
            topology=args.topology,
            trajectory=traj,
            ligand_selection=args.ligand,
            binding_site_cutoff=args.cutoff,
            stride=args.stride,
            in_memory=args.in_memory,
            log_level="DEBUG" if args.verbose else "INFO",
        )

        engine.run_deep_dive(
            start_frame=args.start,
            stop_frame=args.stop,
            compute_pca=not args.no_pca,
            compute_cross_correlation=not args.no_dccm,
        )

        engine.generate_report(args.output)
        print(f"\n✅ Report saved to: {args.output}")


if __name__ == "__main__":
    main()
