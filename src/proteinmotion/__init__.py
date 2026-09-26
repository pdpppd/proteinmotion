"""ProteinMotion: programmatic molecular films, rendered on native GPUs."""

from . import rates
from .animation import (
    Conceal,
    Deform,
    FadeIn,
    FadeOut,
    Focus,
    FocusPull,
    Morph,
    PlayTrajectory,
    Representation,
    Reveal,
    Rotate,
)
from .annotations import Callout, ResidueLabel, ResidueLabels, Text, Unwrite, Write
from .backbone import BackboneMorph
from .density import DensityMap, DensitySlice, DensitySurface
from .distances import Distance
from .eevee import EEVEEOptions
from .interactions import Electrostatics, HydrogenBonds, Interaction, InteractionHighlight, charges_from_pqr
from .matching import ContactMatch, match_backbones
from .nucleic import BaseStyle
from .plots import ColorLegend, ContactMap, SequenceTrack, TimeSeriesPlot
from .properties import ColorByProperty, ColorScale, ResidueValues
from .protein import NucleicAcid, Protein
from .rates import ease_in_out_sine, linear, smooth, there_and_back
from .regions import Region, RegionHighlight
from .scene import ProteinScene, Scene
from .styling import Colorize, HideAtoms, HideSideChains, SetOpacity, ShowAtoms, ShowSideChains
from .thread import Thread, Unthread
from .trajectory import Trajectory

__version__ = "0.12.0"
__all__ = [
    "NucleicAcid",
    "BaseStyle",
    "DensityMap",
    "DensitySurface",
    "DensitySlice",
    "ColorScale",
    "ResidueValues",
    "ColorByProperty",
    "ColorLegend",
    "TimeSeriesPlot",
    "SequenceTrack",
    "ContactMap",
    "Electrostatics",
    "HydrogenBonds",
    "Interaction",
    "InteractionHighlight",
    "charges_from_pqr",
    "Distance",
    "Colorize",
    "SetOpacity",
    "ShowAtoms",
    "HideAtoms",
    "ShowSideChains",
    "HideSideChains",
    "Text",
    "Callout",
    "ResidueLabel",
    "ResidueLabels",
    "Write",
    "Unwrite",
    "BackboneMorph",
    "ContactMatch",
    "match_backbones",
    "ProteinScene",
    "Scene",
    "Protein",
    "Trajectory",
    "Rotate",
    "Morph",
    "Deform",
    "PlayTrajectory",
    "FadeIn",
    "FadeOut",
    "Focus",
    "FocusPull",
    "EEVEEOptions",
    "Region",
    "RegionHighlight",
    "Representation",
    "Reveal",
    "Conceal",
    "Thread",
    "Unthread",
    "rates",
    "smooth",
    "linear",
    "ease_in_out_sine",
    "there_and_back",
]
