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
    SecondaryStructure,
)
from .annotations import Callout, ResidueLabel, ResidueLabels, Text, TextGroup, Unwrite, Write
from .backbone import BackboneMorph
from .conformations import DomainMotion, StructureMatch, StructureMorph, domain_motion, match_structures
from .density import DensityMap, DensitySlice, DensitySurface
from .distances import Distance
from .eevee import EEVEEOptions
from .interactions import Electrostatics, HydrogenBonds, Interaction, InteractionHighlight, charges_from_pqr
from .looks import StudioLook
from .matching import ContactMatch, match_backbones
from .nucleic import BaseStyle
from .plots import ColorLegend, ContactMap, Heatmap, SequenceTrack, TimeSeriesPlot
from .properties import ColorByProperty, ColorScale, ResidueValues
from .protein import NucleicAcid, Protein
from .ramachandran import RamachandranPlot
from .rates import ease_in_out_sine, linear, smooth, there_and_back
from .regions import Region, RegionHighlight
from .scene import ProteinScene, Scene
from .styles import MolecularStyle, TextStyle
from .styling import Colorize, HideAtoms, HideSideChains, SetOpacity, ShowAtoms, ShowSideChains
from .thread import Thread, Unthread
from .timeline import AnimationGroup
from .torsions import RotateTorsions, SetTorsions, TorsionAngles, TorsionMarker
from .trajectory import Trajectory

__version__ = "0.13.0"
__all__ = [
    "AnimationGroup",
    "MolecularStyle",
    "TextStyle",
    "TextGroup",
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
    "Heatmap",
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
    "StructureMorph",
    "StructureMatch",
    "match_structures",
    "domain_motion",
    "DomainMotion",
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
    "StudioLook",
    "Region",
    "RegionHighlight",
    "Representation",
    "Reveal",
    "Conceal",
    "Thread",
    "Unthread",
    "SetTorsions",
    "RotateTorsions",
    "TorsionAngles",
    "TorsionMarker",
    "RamachandranPlot",
    "SecondaryStructure",
    "rates",
    "smooth",
    "linear",
    "ease_in_out_sine",
    "there_and_back",
]
