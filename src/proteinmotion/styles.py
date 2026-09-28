"""Reusable molecular representation settings, independent of activation."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MolecularStyle:
    """A named representation and its options, usable for setup or animation.

    ``MolecularStyle('surface', {'kind': 'vdw', 'resolution': 0.5})`` can be
    passed to ``protein.set_style(style)`` or ``Representation(protein, style)``.
    """

    representation: str = "cartoon"
    options: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.representation not in ("cartoon", "ribbon", "ball_and_stick", "surface"):
            raise ValueError("Choose cartoon, ribbon, ball_and_stick, or surface")
        object.__setattr__(self, "options", dict(self.options))

    def apply(self, protein):
        return getattr(protein, self.representation)(**self.options)


@dataclass(frozen=True)
class TextStyle:
    """Reusable typography, with dimensions in 1080p design pixels."""

    font_size: float = 36
    font: str = "regular"
    color: str | tuple = "#edf3fc"
    line_spacing: float = 1.3
