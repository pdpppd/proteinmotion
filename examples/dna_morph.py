"""C1′ contact-guided morph between chain A of PDB 1BNA and 2DCG.

Matched nucleotides are gold. Unmatched source residues fade out and unmatched
destination residues fade in. This is a visual interpolation, not simulated dynamics.
Render: proteinmotion render examples/dna_morph.py DNAMorph --fps 60 -o dna-morph.mp4
"""

from pathlib import Path

from proteinmotion import (
    BackboneMorph,
    NucleicAcid,
    ProteinScene,
    Representation,
    Text,
    Write,
    match_backbones,
)

DATA = Path(__file__).resolve().parent / "data"


class DNAMorph(ProteinScene):
    def construct(self):
        source = (
            NucleicAcid.from_file(DATA / "1bna.cif", chains="A")
            .cartoon(bases="rings", color="#75b8d9")
            .center()
        )
        target = (
            NucleicAcid.from_file(DATA / "2dcg.cif", chains="A")
            .cartoon(bases="rings", color="#df92a0")
            .center()
        )
        source.rotate(1.35, axis=(1, 0, 0)).rotate(-0.28, axis=(0, 0, 1))
        target.rotate(1.35, axis=(1, 0, 0)).rotate(-0.28, axis=(0, 0, 1))
        match = match_backbones(source, target, max_contact_error=0.20, max_candidates=None, search_seconds=2)
        self.match = match
        for p, ids in ((source, match.source_indices), (target, match.target_indices)):
            p.color_residues("#ffd47b", chain="A", residues=[p.topology.residues[i].resid for i in ids])
            extras = [r.resid for r in p.topology.residues if not r.is_nucleic]
            if extras:
                p.set_residue_opacity(0, chain="A", residues=extras)
        self.add(source)
        self.camera.frame(source, aspect=self.width / self.height).zoom(1.1)
        self.camera.depth_cue = 0.25
        self.add(Text("1BNA A → 2DCG A · matched nucleotides in gold", position=(0.055, 0.89), font_size=27))
        self.play(Write(Text("DNA morph · C1′ anchors", position=(0.055, 0.10), font_size=40)), run_time=1)
        self.wait(1)
        self.play(
            BackboneMorph(source, target, match=match, residue_delay=0.25, align=True),
            run_time=5,
        )
        self.focus(target, margin=1.4, run_time=1.5)
        self.play(Representation(target, "ball_and_stick"), run_time=1.2)
        self.play(self.camera.animate.orbit(theta=0.35), run_time=2)
        self.wait(0.5)
