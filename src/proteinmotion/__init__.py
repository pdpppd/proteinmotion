"""ProteinMotion: programmatic molecular films, rendered through native Metal."""

from . import rates
from .animation import Deform, FadeIn, FadeOut, Focus, Morph, PlayTrajectory, Representation, Rotate
from .annotations import Callout, ResidueLabel, ResidueLabels, Text, Unwrite, Write
from .backbone import BackboneMorph
from .matching import ContactMatch, match_backbones
from .protein import Protein
from .rates import ease_in_out_sine, linear, smooth, there_and_back
from .regions import Region, RegionHighlight
from .scene import ProteinScene, Scene
from .trajectory import Trajectory

__version__ = "0.5.0"
__all__ = [
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
    "Region",
    "RegionHighlight",
    "Representation",
    "rates",
    "smooth",
    "linear",
    "ease_in_out_sine",
    "there_and_back",
]
