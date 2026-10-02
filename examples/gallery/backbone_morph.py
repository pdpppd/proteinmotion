"""Morph one protein into a different one through matched contacts.

Inputs: calmodulin (PDB 1CLL) and troponin C (PDB 1NCX). A contact-map
search pairs 114 Cα residues; matched residues move N to C with a 25 ms
delay, while unmatched residues fade out or in. The path is illustrative.
"""

from pathlib import Path

from proteinmotion import BackboneMorph, ContactMatch, Protein, ProteinScene, StudioLook, Text

DATA = Path(__file__).resolve().parent.parent / "data"


class ContactMorph(ProteinScene):
    renderer = "studio"
    look = StudioLook(lighting="studio", material="medium", effects="subtle")

    def construct(self):
        source = Protein.from_file(DATA / "1cll.cif", chains="A").cartoon(color="rainbow").center()
        target = Protein.from_file(DATA / "1ncx.cif", chains="A").cartoon(color="rainbow").center()
        match = ContactMatch.load(DATA / "calmodulin-troponin-match.json")
        self.camera.frame(source, target, margin=1.0, screen_position=(0.6, 0.52))
        self.add(source, Text("Contact-guided morphs", position=(0.05, 0.08), font_size=46, font="semibold"))
        caption = Text("Calmodulin · PDB 1CLL", position=(0.05, 0.145), font_size=28, color="#a3b3c7")
        self.add(caption)
        self.play(self.camera.animate.orbit(0.3), run_time=1.5)
        self.play(
            BackboneMorph(source, target, match=match, residue_delay=0.025),
            caption.animate.set_text("Troponin C · PDB 1NCX · 114 matched Cα"),
            run_time=6,
        )
        self.play(self.camera.animate.orbit(0.6), run_time=2.5)
