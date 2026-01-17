"""
Visualization module for thesis figures.

Usage:
    python -m visualization.thesis_figures --outdir thesis/figures/chapter6
"""

from .thesis_figures import main as generate_chapter6_figures

__all__ = ["generate_chapter6_figures"]
