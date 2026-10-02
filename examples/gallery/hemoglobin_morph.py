"""Hemoglobin T to R: one αβ dimer turns against the other.

Inputs: human deoxyhemoglobin (T state, PDB 2DN2) and oxyhemoglobin (R
state, PDB 2DN1, expanded to the tetramer with assembly 1). Superposed on
α1β1, so the change appears as the rotation of α2β2, viewed down its axis.
"""

from pathlib import Path

from proteinmotion import Protein, ProteinScene, StructureMorph, StudioLook, Text, domain_motion

DATA = Path(__file__).resolve().parent.parent / "data"


class HemoglobinMorph(ProteinScene):
    renderer = "studio"
    look = StudioLook(lighting="soft", material="medium", effects="subtle")

    def construct(self):
        deoxy = Protein.from_file(DATA / "2dn2.cif").cartoon()
        oxy = Protein.from_file(DATA / "2dn1.cif", assembly="1").cartoon()
        oxy.select(resname="MBN").hide_atoms()  # toluene from crystallization
        for chain, color in (("A", "#7f93b3"), ("B", "#9fb0c9"), ("C", "#50e0d0"), ("D", "#f5d477")):
            deoxy.select(chain=chain, polymer=True).set_color(color)
        dimer = deoxy.select(chain=["A", "B"])
        motion = domain_motion(deoxy, oxy, moving=deoxy.select(chain=["C", "D"]), fixed=dimer)
        deoxy.center()
        self.camera.look_along(motion.axis, deoxy)
        self.camera.frame(deoxy, margin=1.0, screen_position=(0.6, 0.55))
        self.add(deoxy, Text("Quaternary change", position=(0.05, 0.08), font_size=46, font="semibold"))
        caption = Text(
            "Deoxyhemoglobin, T state · PDB 2DN2", position=(0.05, 0.145), font_size=28, color="#a3b3c7"
        )
        self.add(caption)
        self.add(Text("α1β1", position=(0.05, 0.83), font_size=30, color="#9fb0c9"))
        self.add(Text("α2β2", position=(0.05, 0.88), font_size=30, color="#f5d477"))
        self.wait(1)
        self.play(
            StructureMorph(deoxy, oxy, align=dimer),
            caption.animate.set_text(f"Oxyhemoglobin, R state · α2β2 turns {motion.angle:.0f}°"),
            run_time=5,
        )
        self.wait(0.5)
        self.play(self.camera.animate.orbit(0.6), run_time=3)
