"""Reusable molecular representation settings, independent of activation.

A MolecularStyle names a representation and its options; ``protein.set_style(style)``
applies it at once and ``Representation(protein, style)`` animates to it. TextStyle holds
typography that ``Text(style=...)`` reads for any argument left unset.
"""

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
        # Keep a private copy, so later edits to the caller's dict cannot change a frozen style.
        object.__setattr__(self, "options", dict(self.options))

    def apply(self, protein):
        """Call the protein's representation method with these options, as set_style() does."""
        return getattr(protein, self.representation)(**self.options)


@dataclass(frozen=True)
class TextStyle:
    """Reusable typography, with dimensions in 1080p design pixels."""

    font_size: float = 36
    font: str = "regular"  # "regular", "semibold" (bundled Source Sans 3) or a font file path
    color: str | tuple = "#edf3fc"
    line_spacing: float = 1.3  # Distance between successive lines, as a multiple of font_size.
