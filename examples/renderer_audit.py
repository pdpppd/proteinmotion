"""Controlled native/studio comparisons. Run scripts/render_renderer_audit.py.

Each factory constructs ONE timeline. Both renderers evaluate the very same objects,
camera, and time. These are visual diagnostics, not physical simulations or assertions
that the two lighting/surface algorithms should produce identical pixels.
"""

import tempfile
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import numpy as np

from proteinmotion import (
    ColorByProperty,
    ColorLegend,
    ColorScale,
    Conceal,
    Deform,
    DensityMap,
    Electrostatics,
    FadeIn,
    FadeOut,
    FocusPull,
    HideAtoms,
    HideSideChains,
    NucleicAcid,
    PlayTrajectory,
    Protein,
    ProteinScene,
    ResidueValues,
    Reveal,
    Rotate,
    ShowAtoms,
    ShowSideChains,
    Text,
    Thread,
    TimeSeriesPlot,
    Trajectory,
    Unthread,
    Unwrite,
    Write,
    charges_from_pqr,
    ease_in_out_sine,
    linear,
    smooth,
    there_and_back,
)

DATA = Path(__file__).parent / "data"


@dataclass
class Case:
    id: str
    group: str
    title: str
    factory: object
    api: list[str]
    notes: str = ""
    times: tuple = ()
    video: bool = False
    look: dict = field(default_factory=dict)
    scene_options: dict = field(default_factory=dict)


class Diagnostic(ProteinScene):
    def __init__(self, setup, **kwargs):
        self.setup = setup
        super().__init__(**kwargs)

    def construct(self):
        self.setup(self)
        if self.duration == 0:
            self.wait(2)


def molecule(code="1ubq", **kwargs):
    return Protein.from_file(DATA / f"{code}.cif", **kwargs).center()


def frame(s, p, target=None, margin=1.12):
    s.add(p)
    s.camera.frame(target if target is not None else p, margin=margin, aspect=s.width / s.height)
    s.camera.theta, s.camera.phi, s.camera.depth_cue = 0.65, 0.18, 0.3


def portrait(s, representation="cartoon", color="secondary", opacity=1, **surface):
    p = molecule()
    getattr(p, representation)(**(surface if representation == "surface" else {}))
    if representation in ("cartoon", "ribbon", "surface"):
        getattr(p, representation)(color=color, **(surface if representation == "surface" else {}))
    p.set_opacity(opacity)
    frame(s, p)


def nucleic(s, code="1bna", bases="slabs", representation="cartoon"):
    p = NucleicAcid.from_file(DATA / f"{code}.cif").center().cartoon(bases=bases)
    p.rotate(1.35 if code == "1bna" else -0.5, axis=(1, 0, 0))
    if representation != "cartoon":
        getattr(p, representation)()
    frame(s, p)


def surface_deform(s, update):
    p = molecule().surface(resolution=0.85, update=update)
    frame(s, p, margin=1.3)
    s.wait(0.3)
    s.play(Deform(p, lambda xyz: (xyz - xyz.mean(0)) * [1.35, 0.85, 1] + xyz.mean(0)), run_time=2)
    s.wait(0.5)


def details(s, sides=False, representation="cartoon"):
    p = molecule()
    getattr(p, representation)()
    site = p.select(residues=(23, 34))
    site.set_color("#f2ba67")
    frame(s, p, margin=1.0)
    s.wait(0.3)
    show, hide = (ShowSideChains, HideSideChains) if sides else (ShowAtoms, HideAtoms)
    s.play(show(site, residue_delay=0.04), run_time=1.3)
    s.wait(0.4)
    s.play(hide(site, residue_delay=0.04), run_time=1.3)
    s.wait(0.3)


def thread(s, code="1ubq", glow=12, reverse=False):
    p = molecule(code).cartoon(color="chain" if code == "4hhb" else "secondary")
    frame(s, p, margin=1.35)
    cls = Unthread if reverse else Thread
    s.play(
        cls(p, s.camera, radius=0.35, glow=glow, seed=7, stagger=0.12 if code == "4hhb" else 0), run_time=4
    )
    s.wait(0.4)


def cutaway(s, representation="surface", shape="cone", softness=0.16, wall=0.28, rings=5):
    p = molecule("2dri", chains="A")
    getattr(p, representation)(**({"resolution": 0.65} if representation == "surface" else {}))
    ligand = p.select(resname="RIP")
    ligand.set_color("#f2ba67")
    frame(s, p, margin=1.05)
    s.camera.theta = 1.5
    s.wait(0.3)
    s.play(
        Reveal(s.camera, ligand, shape=shape, window=1.6, softness=softness, band=2, wall=wall, rings=rings),
        run_time=1.4,
    )
    s.play(s.camera.animate.orbit(0.22, 0.05), run_time=1)
    s.play(Conceal(s.camera), run_time=1)
    s.wait(0.3)


def hetero(s, code="4hhb", representation="cartoon", water=False, hidden=False):
    p = molecule(code, include_water=water)
    getattr(p, representation)(**({"resolution": 0.8} if representation == "surface" else {}))
    if hidden:
        p.hide_atoms()
    frame(s, p)


def layers(s, representation="cartoon"):
    p = molecule()
    getattr(p, representation)()
    p.set_color("#5dcfbc").set_opacity(0.45).shift((-5, 0, -5))
    q = p.copy().set_color("#f4ae69").set_opacity(0.55).shift((10, 0, 10))
    s.add(p, q)
    s.camera.frame(p, q, margin=1.1, aspect=s.width / s.height)
    s.camera.depth_cue = 0
    s.play(s.camera.animate.orbit(0.7), run_time=3)


def direct_styling(s):
    p = molecule().cartoon()
    p.with_secondary_structure("C" * len(p.topology.residues))
    p.color_residues("#f2ba67", chain="A", residues=(23, 34))
    p.set_residue_opacity(0.35)
    site = p.select(residues=(23, 34))
    site.set_opacity(1).show_atoms()
    frame(s, p)
    s.play(
        site.animate.set_color("#65ddc7"),
        site.animate.set_opacity(0.7),
        site.animate.hide_atoms(),
        run_time=1.2,
    )
    s.play(site.animate.show_atoms(), site.animate.set_opacity(1), run_time=1.2)


def transform(s):
    p = molecule().cartoon(color="#65ddc7")
    p.rotate(0.15, axis=(1, 0, 0)).scale(0.9).shift((-2, 0, 0))
    q = p.copy().set_color("#f2ba67").shift((27, 0, 0)).scale(0.7)
    s.add(p, q)
    s.camera.frame(p, q, margin=1.25, aspect=s.width / s.height)
    s.play(
        p.animate.shift((5, 2, 0)).rotate(0.8).scale(1.25).set_opacity(0.6),
        p.animate.set_color("#70a8ee"),
        run_time=2,
    )
    s.play(s.camera.animate.zoom(1.15).depth_cue(0.9), run_time=1.5)


def camera_tracking(s):
    p = molecule().cartoon()
    site = p.select(residues=(23, 34))
    frame(s, p)
    s.camera.theta, s.camera.phi = s.camera.clearest_view(site)
    s.focus(site, margin=1.7, run_time=1)
    s.play(p.animate.shift((7, 0, 0)).rotate(0.6), run_time=2)
    s.play(s.camera.animate.focus(p, aspect=s.width / s.height), run_time=1)


def lens(s):
    p = molecule().ball_and_stick()
    frame(s, p)
    s.camera.set_focus(p.select(residues=8), fstop=0.8)
    s.play(FocusPull(s.camera, p.select(residues=70), fstop=2), run_time=3)
    s.play(s.camera.animate.set_focus(p.select(residues=8), fstop=0.8), run_time=1)


def fog(s, value):
    portrait(s, representation="ball_and_stick")
    s.camera.depth_cue = value


def fade(s, representation):
    p = molecule()
    getattr(p, representation)()
    frame(s, p)
    s.play(FadeIn(p), run_time=1)
    s.wait(0.5)
    s.play(FadeOut(p), run_time=1)


def rates_scene(s):
    proteins = []
    for i, (name, rate) in enumerate(
        (("linear", linear), ("smooth", smooth), ("sine", ease_in_out_sine), ("there + back", there_and_back))
    ):
        p = molecule().cartoon().scale(0.45).shift(((i - 1.5) * 24, 0, 0))
        proteins.append((p, rate))
        s.add(p, Text(name, position=(0.035 + i * 0.245, 0.82), font_size=32))
    s.camera.frame(*(p for p, _ in proteins), margin=1.12, aspect=s.width / s.height)
    s.play(*(Rotate(p, np.pi * 1.5, rate_func=rate) for p, rate in proteins), run_time=4)


def ruler(s, mode="3d", style="dashed", world=False):
    p = molecule().cartoon()
    a, b = p.select(residues=23, atoms="CA"), p.select(residues=34, atoms="CA")
    if world:
        q = p.copy().shift((26, 0, 0)).scale(0.8)
        b = q.select(residues=23, atoms="CA")
        s.add(p, q)
        s.camera.frame(p, q, aspect=s.width / s.height)
    else:
        frame(s, p)
    r = a.distance_to(
        b,
        mode=mode,
        style=style,
        space="world" if world else "model",
        font_size=42,
        radius=0.13,
        unit="nm" if world else "Å",
    )
    s.play(Write(r), run_time=1)
    s.play(s.camera.animate.orbit(0.4), run_time=1.5)


def hbonds(s, mode="3d", explicit=True):
    from alpha_helix_hbonds import analyze_helix, ideal_helix

    p = ideal_helix().ball_and_stick(atom_scale=0.25, bond_radius=0.09)
    h = analyze_helix(p) if explicit else p.hydrogen_bonds(hydrogens="backbone")
    frame(s, p)
    lines = h.highlight(
        mode=mode,
        endpoints="hydrogen_acceptor",
        show_distances=False,
        color="#f2ba67",
        radius=0.07,
    )
    s.play(Write(lines), run_time=1.4)
    s.play(s.camera.animate.orbit(0.5), run_time=1.4)


def label_options(s, tip="dot"):
    p = molecule()
    frame(s, p, margin=1.25)
    a = p.select(residues=23).label(offset=(-290, -180), font_size=38, tip=tip)
    b = p.select(residues=34).callout(
        "End of helix", subtitle="GLU 34", tip=tip, position=(0.73, 0.65), font_size=36
    )
    s.play(Write(a), Write(b), run_time=1)
    s.play(a.animate.set_offset((-310, 90)), b.animate.shift((-0.1, -0.1)), run_time=1)
    s.play(Unwrite(a), Unwrite(b), run_time=1)


def text_styles(s):
    for i, font in enumerate(("regular", "semibold")):
        t = Text(
            "α β → Ca²⁺ · 3.2 Å\nMultiline text / 0123456789",
            font=font,
            font_size=55,
            position=(0.08, 0.13 + i * 0.31),
            line_spacing=1.5,
        )
        t.set_text(t.text)
        s.add(t)
    t = Text("Corner placement", font_size=30).to_corner("DR")
    s.play(Write(t, reverse=True), run_time=1)
    s.play(t.animate.shift((-0.2, -0.1)).set_opacity(0.4), run_time=1)
    s.play(Unwrite(t, reverse=True), run_time=1)


def values(s):
    p = molecule()
    vals = ResidueValues.from_mapping(
        p, {("A", i): (i - 23) / 11 for i in range(23, 35)}, name="Partial values"
    )
    scale = ColorScale.from_values(vals, missing="#a765a3")
    frame(s, p)
    s.add(ColorLegend(scale, title="Missing values: purple", position=(0.06, 0.80)))
    s.play(ColorByProperty(p, vals, scale=scale, thickness=(0.55, 1.7)), run_time=2)
    s.wait(0.5)


def density(s, axis=None, opacity=0.28):
    p = molecule().ball_and_stick()
    p.set_residue_opacity(0.06)
    site = p.select(residues=(23, 34)).set_opacity(1)
    m = DensityMap.from_file(DATA / "1ubq.ccp4").crop(site, padding=2.5)
    obj = (
        m.isosurface(1.5, opacity=opacity, follow=p)
        if axis is None
        else m.slice(axis, 0.5, follow=p, resolution=96)
    )
    frame(s, p, target=site, margin=1.6)
    s.add(obj)


def large(s, representation="cartoon"):
    p = molecule("1aon")
    getattr(p, representation)(
        **({"resolution": 1.5, "color": "chain"} if representation == "surface" else {"color": "chain"})
    )
    frame(s, p, margin=1.0)


def all_atoms(s):
    p = molecule().cartoon().show_atoms()
    p.select(residues=(1, 10)).hide_atoms()
    frame(s, p)


def copied_markers(s):
    p = molecule()
    frame(s, p)
    marker = p.select(residues=(23, 34)).highlight(style="sphere")
    # copy() deliberately rejects live highlights; independently create the second.
    try:
        marker.copy()
    except ValueError as exc:
        assert "another region.highlight" in str(exc)
    else:
        raise AssertionError("RegionHighlight.copy unexpectedly succeeded")
    other = p.select(residues=(2, 7)).highlight(style="sphere", color="#77aadd")
    s.add(marker, other)
    label = p.select(residues=23).label().set_offset((-180, -120))
    label.set_opacity(0.7)
    title = Text("Copied highlight / direct label placement", font_size=32)
    title.move_to((0.07, 0.10)).shift((0.01, 0))
    s.add(label, title)


def trajectory_input(s, kind):
    if kind == "xtc":
        traj = Trajectory.from_mdanalysis(
            DATA / "showcase-calmodulin.pdb", DATA / "showcase-calmodulin.xtc", stride=3
        )
        p = Protein.from_trajectory(traj).center()
    else:
        p = molecule()
        xyz = p.positions
        frames = np.stack([xyz, xyz * [1.2, 0.9, 1.0], xyz]) / 10
        with tempfile.TemporaryDirectory(prefix="proteinmotion-audit-") as tmp:
            file = Path(tmp) / "frames.npy"
            np.save(file, frames)
            mapped = Trajectory.from_npy(file, topology=p.topology, units="nm")
            # Materialize before the temporary file is removed; conversion is tested.
            traj = Trajectory([mapped.frame(i).copy() for i in range(len(mapped))], topology=p.topology)
        p = Protein.from_trajectory(traj).center()
    frame(s, p)
    s.play(PlayTrajectory(p), run_time=3)


def electrostatic(s, pqr=False):
    p = molecule()
    field = p.electrostatics()
    if pqr:
        # Formal-charge diagnostic fixture, not a force-field parameterization.
        lines = [
            f"ATOM {i + 1} {a.name} {a.resname} {a.chain} {a.resid}{a.icode} {xyz[0]:.5f} {xyz[1]:.5f} {xyz[2]:.5f} {q:.6f} 1.5"
            for i, (a, xyz, q) in enumerate(zip(p.topology.atoms, p.positions, field.charges))
        ]
        with tempfile.TemporaryDirectory(prefix="proteinmotion-audit-") as tmp:
            file = Path(tmp) / "formal.pqr"
            file.write_text("\n".join(lines))
            assert np.allclose(charges_from_pqr(p, file), field.charges)
            field = Electrostatics.from_pqr(p, file)
        p.ball_and_stick()
        frame(s, p)
        s.add(field.highlight(mode="3d", max_pairs=12, show_distances=False))
    else:
        values = field.potential(p.positions[[r.trace_atom for r in p.topology.residues]])
        scale = ColorScale(-1.2, 1.2, colors=("#538ada", "#eee9df", "#dc716a"))
        p.color_by(ResidueValues(p, values), scale=scale)
        frame(s, p)
        s.add(ColorLegend(scale, title="Illustrative screened potential", unit="kcal/mol/e"))
    pair = field.pairs[0]
    assert pair.as_dict()["energy"] == pair.energy
    energy = field.pair_energy(pair.a, pair.b)
    s.add(Text(f"Strongest pair: {energy:.3f} kcal/mol", position=(0.06, 0.09), font_size=30))


def direct_density(s):
    p = molecule().ball_and_stick()
    site = p.select(residues=(23, 34))
    p.set_residue_opacity(0.1)
    site.set_opacity(1)
    m = DensityMap.from_file(DATA / "1ubq.ccp4").crop(site, padding=2.5)
    shell = m.isosurface(1.5, opacity=0.4, follow=p).set_level(2.5)
    section = m.slice("z", 0.5, follow=p).set_slice(0.3)
    frame(s, p, target=site, margin=1.6)
    s.add(shell, section)


def plot_reveal(s):
    times = np.linspace(0, 4, 81)
    values = np.sin(times * 2)
    values[30:40] = np.nan
    s.add(
        TimeSeriesPlot(
            times,
            values,
            reveal=True,
            grid=True,
            title="Trace reveal / missing interval",
            position=(0.12, 0.18),
            size=(0.76, 0.65),
        )
    )
    s.wait(4)


def explicit_match(s):
    from proteinmotion import BackboneMorph, ContactMatch

    p, q = molecule("1cll", chains="A"), molecule("1ncx", chains="A")
    saved = ContactMatch.load(DATA / "calmodulin-troponin-match.json")
    match = ContactMatch.from_pairs(p, q, saved.source_indices, saved.target_indices)
    with tempfile.TemporaryDirectory(prefix="proteinmotion-audit-") as tmp:
        file = Path(tmp) / "match.json"
        match.save(file)
        match = ContactMatch.load(file)
    frame(s, p)
    s.play(BackboneMorph(p, q, match=match), run_time=3)
    s.wait(0.5)


def sampled_density(s):
    p = molecule().ball_and_stick()
    site = p.select(residues=(23, 34))
    p.set_residue_opacity(0.08)
    site.set_opacity(1)
    m = DensityMap.from_file(DATA / "1ubq.ccp4").crop(site, padding=2.5)
    values = ResidueValues(p, m.sample(p.positions[[r.trace_atom for r in p.topology.residues]]))
    scale = ColorScale.from_values(values)
    p.color_by(values, scale=scale)
    shell = m.isosurface(1.5, opacity=0.15, follow=p)
    section = m.slice("z", 0.5, opacity=0.6, follow=p)
    s.add(p, shell, section, ColorLegend(scale, title="Sampled map at Cα"))
    s.camera.frame(shell, section, margin=1.1, aspect=s.width / s.height)
    s.camera.theta, s.camera.phi = 0.55, 0.2


def cases():
    from backbone_morph import BackboneDemo, BallAndStickDemo
    from density_maps import DensityMaps
    from dna_morph import DNAMorph
    from dna_styles import DNAStyles
    from docs_examples import EXAMPLES
    from ligands_and_side_chains import LigandsAndSideChains
    from nucleic_ions import NucleicIons
    from numerical_properties import NumericalProperties
    from rna_styles import RNAStyles
    from studio_buried_ligand import BuriedRibose
    from synchronized_plots import SynchronizedPlots

    out = []

    def add(
        group, title, setup, api, *, notes="", times=(), video=False, look=None, options=None, scene=False
    ):
        out.append(
            Case(
                f"R{len(out) + 1:03d}",
                group,
                title,
                setup if scene else partial(Diagnostic, setup),
                api.split(),
                notes,
                times,
                video,
                look or {},
                options or {},
            )
        )

    # Documented examples exercise the user-facing composition paths unchanged.
    api = [
        "ProteinScene Scene Rotate FadeIn Representation Animate.shift Animate.orbit",
        "Protein.cartoon Protein.ribbon Protein.ball_and_stick Representation",
        "Focus ProteinScene.focus Region.highlight FadeIn FadeOut",
        "RegionHighlight Region.highlight",
        "Text Write Unwrite",
        "Text.to_corner AnnotationAnimate.move_to AnnotationAnimate.set_opacity",
        "Callout Region.callout",
        "ResidueLabels Protein.label_residues",
        "Region.set_color Region.set_opacity Protein.select Region.select",
        "Colorize",
        "SetOpacity",
        "Protein.surface",
        "Distance",
        "HydrogenBonds Interaction InteractionHighlight",
        "Electrostatics InteractionHighlight",
        "Deform Morph Protein.set_positions",
        "Trajectory Protein.from_trajectory PlayTrajectory Trajectory.aligned",
    ]
    for (slug, cls, title, caption, poster), names in zip(EXAMPLES, api):
        add("Documented workflows", title, cls, names, notes=caption, times=(poster,), video=True, scene=True)

    for title, cls, names, times in [
        (
            "Protein backbone morph",
            BackboneDemo,
            "BackboneMorph ContactMatch ContactMatch.load",
            (0.5, 3.8, 7.5),
        ),
        ("Backbone morph · atom detail", BallAndStickDemo, "BackboneMorph", (0.5, 3.8, 7.5)),
        ("DNA backbone morph", DNAMorph, "BackboneMorph match_backbones", (1.5, 4.5, 8.5)),
        (
            "DNA style transitions",
            DNAStyles,
            "NucleicAcid BaseStyle Protein.set_bases",
            (1.5, 6, 9, 12, 17, 22),
        ),
        ("RNA / modified nucleotides", RNAStyles, "NucleicAcid BaseStyle ColorByProperty", (2, 5.5, 8, 13)),
        (
            "Ion coordination side chains",
            LigandsAndSideChains,
            "ShowSideChains Protein.select",
            (0.5, 4.3, 6.5, 12),
        ),
        ("Nucleic ions and ligands", NucleicIons, "NucleicAcid ShowAtoms BaseStyle", (3, 7, 13, 21, 27)),
        (
            "Property colors and thickness",
            NumericalProperties,
            "ResidueValues ColorByProperty ColorScale ColorLegend Protein.color_by",
            (1, 3, 5, 8),
        ),
        (
            "Synchronized ensemble plots",
            SynchronizedPlots,
            "ContactMap TimeSeriesPlot TimeSeriesPlot.distance SequenceTrack ResidueValues.rmsf",
            (0.5, 4, 9),
        ),
        (
            "Animated density / slice",
            DensityMaps,
            "DensityMap DensitySurface DensitySlice DensityAnimate.set_level DensityAnimate.set_slice",
            (0.5, 2.5, 6, 8),
        ),
    ]:
        add("Extended workflows", title, cls, names, times=times, video=True, scene=True)

    for rep in ("cartoon", "ribbon", "ball_and_stick", "surface"):
        add(
            "Representations",
            rep.replace("_", " ").title(),
            partial(portrait, representation=rep),
            f"Protein.{rep}",
        )
    for color in ("secondary", "chain", "rainbow", "#67cbb5"):
        add(
            "Color",
            f"Cartoon color · {color}",
            partial(portrait, color=color),
            "Protein.cartoon Protein.set_color",
        )
    for kind in ("ses", "sas", "vdw"):
        add(
            "Surfaces",
            f"Surface kind · {kind.upper()}",
            partial(portrait, representation="surface", kind=kind, resolution=0.65),
            "Protein.surface",
            notes="Both use the requested molecular surface. Inspect matching cavities and silhouette under different shading.",
        )
    for probe in (0.5, 2.8):
        add(
            "Surfaces",
            f"Probe radius · {probe} Å",
            partial(portrait, representation="surface", kind="sas", probe_radius=probe, resolution=0.65),
            "Protein.surface",
            notes="Checks whether probe radius changes the rendered surface.",
        )
    for resolution in (0.4, 1.2):
        add(
            "Surfaces",
            f"Grid resolution · {resolution} Å",
            partial(portrait, representation="surface", resolution=resolution),
            "Protein.surface",
        )
    for update in ("rebuild", "deform"):
        add(
            "Surfaces",
            f"Moving surface · {update}",
            partial(surface_deform, update=update),
            "Protein.surface Deform",
            video=True,
            times=(0.3, 1.3, 2.7),
        )

    for code, name in (("1bna", "DNA"), ("1ehz", "RNA")):
        for bases in ("slabs", "rings", "sticks", "ladder", "none"):
            add(
                "Nucleic acids",
                f"{name} bases · {bases}",
                partial(nucleic, code=code, bases=bases),
                "NucleicAcid BaseStyle NucleicAcid.cartoon Protein.set_bases",
            )
        for rep in ("ball_and_stick", "surface"):
            add(
                "Nucleic acids",
                f"{name} · {rep.replace('_', ' ')}",
                partial(nucleic, code=code, representation=rep),
                f"NucleicAcid Protein.{rep}",
            )

    for rep in ("cartoon", "ribbon", "surface"):
        add(
            "Atom detail",
            f"Show / hide atoms over {rep}",
            partial(details, representation=rep),
            "ShowAtoms HideAtoms",
            video=True,
            times=(0.2, 1, 1.8, 3.5),
        )
    add(
        "Atom detail",
        "Show / hide side chains",
        partial(details, sides=True),
        "ShowSideChains HideSideChains Region.side_chains",
        video=True,
        times=(0.2, 1, 1.8, 3.5),
    )
    for code, name, water in (
        ("4hhb", "Heme ligands", False),
        ("1cll", "Calcium ions", False),
        ("1ubq", "Deposited waters", True),
    ):
        for rep in ("cartoon", "surface"):
            add(
                "Hetero atoms",
                f"{name} · {rep}",
                partial(hetero, code=code, representation=rep, water=water),
                "Protein.from_file Protein.select",
                notes="Ligands, ions and waters outside traced chains need explicit review in surface mode.",
            )
    add("Hetero atoms", "Hide all default ligand detail", partial(hetero, hidden=True), "Protein.hide_atoms")

    for rep in ("cartoon", "ball_and_stick", "surface"):
        add(
            "Transparency",
            f"Global opacity 0.35 · {rep}",
            partial(portrait, representation=rep, opacity=0.35),
            "Protein.set_opacity",
        )
        add(
            "Transparency",
            f"Intersecting transparent {rep}",
            partial(layers, representation=rep),
            "Protein.copy Protein.set_opacity",
            video=True,
        )
        add(
            "Transparency",
            f"Fade in / out · {rep}",
            partial(fade, representation=rep),
            "FadeIn FadeOut",
            video=True,
            times=(0.4, 1.25, 2.1),
        )

    for code, glow, rev, name in (
        ("1ubq", 12, False, "Thread with glow"),
        ("1ubq", 0, False, "Thread without glow"),
        ("1ubq", 12, True, "Unthread"),
        ("4hhb", 12, False, "Staggered multichain thread"),
        ("1bna", 12, False, "DNA threading"),
    ):
        add(
            "Threading",
            name,
            partial(thread, code=code, glow=glow, reverse=rev),
            "Unthread" if rev else "Thread",
            video=True,
            times=(0.8, 2, 3.3, 4.3),
            notes="Seed 7; procedural presentation animation, not folding dynamics.",
        )
    for rep in ("cartoon", "ball_and_stick", "surface"):
        for shape in ("cone", "tunnel"):
            add(
                "Cutaways",
                f"{shape.title()} cutaway · {rep}",
                partial(cutaway, representation=rep, shape=shape),
                "Reveal Conceal Camera.cutaway_geometry",
                video=True,
                times=(0.2, 1, 1.8, 2.65, 3.95),
            )
    add(
        "Cutaways",
        "Soft edge / low wall opacity",
        partial(cutaway, shape="tunnel", softness=0.5, wall=0.08, rings=2.5),
        "Reveal",
        video=True,
        times=(0.2, 1.8, 2.65),
    )

    add(
        "Styling and transforms",
        "Direct and fluent region styling",
        direct_styling,
        "Protein.with_secondary_structure Protein.color_residues Protein.set_residue_opacity RegionAnimate.set_color RegionAnimate.set_opacity RegionAnimate.show_atoms RegionAnimate.hide_atoms Region.show_atoms Region.hide_atoms",
        video=True,
    )
    add(
        "Styling and transforms",
        "Object transforms and copies",
        transform,
        "Protein.center Protein.shift Protein.rotate Protein.scale Protein.copy Animate.shift Animate.rotate Animate.scale Animate.set_color Animate.set_opacity Animate.zoom Animate.depth_cue",
        video=True,
    )
    add(
        "Camera",
        "Automatic view and camera tracking",
        camera_tracking,
        "Camera.clearest_view Camera.frame Camera.focus Camera.update_tracking Focus",
        video=True,
    )
    for value in (0, 0.65, 1):
        add("Camera", f"Depth cue · {value}", partial(fog, value=value), "Camera.depth_cue")
    add(
        "Camera",
        "Lens focus control (EEVEE only)",
        lens,
        "FocusPull Camera.set_focus Animate.set_focus",
        video=True,
        notes="Control case: neither native nor studio implements EEVEE depth of field. No focus blur is expected; this is a documented capability boundary.",
    )
    add(
        "Timing",
        "All four rate functions",
        rates_scene,
        "rates linear smooth ease_in_out_sine there_and_back",
        video=True,
    )

    for mode in ("2d", "3d"):
        for style in ("solid", "dashed"):
            add(
                "Interactions",
                f"Distance · {mode} / {style}",
                partial(ruler, mode=mode, style=style),
                "Distance Region.distance_to",
                video=True,
            )
        add(
            "Interactions",
            f"Explicit H-bond network · {mode}",
            partial(hbonds, mode=mode),
            "HydrogenBonds Interaction InteractionHighlight",
            video=True,
            notes="Ideal 16-residue helix with explicit H atoms; detector recovers all 12 i→i+4 bonds.",
        )
    add(
        "Interactions",
        "Distance between transformed objects",
        partial(ruler, world=True),
        "Distance",
        video=True,
    )
    add(
        "Interactions",
        "Inferred backbone H bonds",
        partial(hbonds, explicit=False),
        "HydrogenBonds Protein.hydrogen_bonds",
        video=True,
    )

    for tip in ("dot", "arrow", "none"):
        add(
            "Annotations",
            f"Residue label / callout · {tip} tip",
            partial(label_options, tip=tip),
            "ResidueLabel Region.label Callout AnnotationAnimate.set_offset AnnotationAnimate.shift Write Unwrite",
            video=True,
        )
    add(
        "Annotations",
        "Text glyphs / font / reverse writing",
        text_styles,
        "Text Text.set_text Text.to_corner AnnotationAnimate.shift AnnotationAnimate.set_opacity Write Unwrite",
        video=True,
    )
    add(
        "Properties",
        "Partial values / missing color / thickness",
        values,
        "ResidueValues.from_mapping ColorScale.from_values ColorByProperty ColorLegend",
        video=True,
    )
    for opacity in (0.28, 1):
        add(
            "Density",
            f"Density isosurface · opacity {opacity}",
            partial(density, opacity=opacity),
            "DensityMap.from_file DensityMap.crop DensityMap.isosurface DensitySurface",
        )
    for axis in ("x", "y", "z"):
        add(
            "Density",
            f"Density slice · {axis.upper()}",
            partial(density, axis=axis),
            "DensityMap.slice DensitySlice",
        )

    for rep in ("cartoon", "surface"):
        add(
            "Scale",
            f"GroEL–GroES · {rep}",
            partial(large, representation=rep),
            f"Protein.{rep}",
            notes="Full 1AON complex; surface grid 1.5 Å. A scale and multichain diagnostic.",
        )
    for color, name in (("#ffffff", "White"), ("#000000", "Black"), ("#b9c9dc", "Light blue")):
        add("Output", f"{name} background", portrait, "ProteinScene", options={"background": color})
    for msaa in (1, 4):
        add(
            "Output",
            f"MSAA ×{msaa}",
            partial(portrait, representation="ball_and_stick"),
            "ProteinScene",
            options={"msaa": msaa},
        )

    # Native remains a fixed reference for these intentionally studio-only options.
    for name in ("studio", "soft", "dramatic", "flat"):
        add(
            "Studio presets",
            f"Lighting · {name}",
            partial(portrait, representation="surface"),
            "StudioLook",
            look={"lighting": name, "effects": "clean"},
            notes="Studio-only preset; native left is the unchanged reference.",
        )
    for name in ("rough", "medium", "glossy"):
        add(
            "Studio presets",
            f"Material · {name}",
            partial(portrait, representation="surface"),
            "StudioLook",
            look={"material": name, "effects": "clean"},
        )
    for name in ("clean", "subtle", "film", "dreamy"):
        add(
            "Studio presets",
            f"Global effects · {name}",
            partial(portrait, representation="ball_and_stick"),
            "StudioLook",
            look={"effects": name},
            video=name == "film",
        )
    for effect, names in (
        ("grain", ("off", "fine", "film", "coarse")),
        ("bloom", ("off", "soft", "strong")),
        ("halation", ("off", "subtle", "warm")),
    ):
        for name in names:
            add(
                "Studio presets",
                f"{effect.title()} · {name}",
                partial(portrait, representation="ball_and_stick"),
                "StudioLook",
                look={"effects": "clean", effect: name},
                video=effect == "grain" and name == "coarse",
            )
    add(
        "Additional API paths", "Whole-protein atom detail", all_atoms, "Protein.show_atoms Region.hide_atoms"
    )
    add(
        "Additional API paths",
        "Independent highlights / direct placement",
        copied_markers,
        "RegionHighlight.copy Annotation.move_to Annotation.shift Annotation.set_opacity ResidueLabel.set_offset",
        notes="RegionHighlight.copy deliberately rejects copying live annotations; this scene verifies that guard, then creates a second highlight independently.",
    )
    add(
        "Additional API paths",
        "NumPy trajectory / nanometer conversion",
        partial(trajectory_input, kind="npy"),
        "Trajectory.from_npy Trajectory.frame Protein.from_trajectory",
        video=True,
        notes="Synthetic coordinate deformation for testing the loader; not molecular dynamics.",
    )
    add(
        "Additional API paths",
        "XTC trajectory reader",
        partial(trajectory_input, kind="xtc"),
        "Trajectory.from_mdanalysis PlayTrajectory",
        video=True,
        notes="Repository showcase trajectory; illustrative motion, not a new simulation.",
    )
    add(
        "Additional API paths",
        "Electrostatic potential coloring",
        electrostatic,
        "Protein.electrostatics Electrostatics.potential Electrostatics.pair_energy",
    )
    add(
        "Additional API paths",
        "PQR charges / 3D electrostatic network",
        partial(electrostatic, pqr=True),
        "charges_from_pqr Electrostatics.from_pqr Electrostatics InteractionHighlight",
        notes="PQR fixture contains illustrative formal charges; this is a parser/render diagnostic, not a force-field calculation.",
    )
    add(
        "Additional API paths",
        "Direct density level / slice setters",
        direct_density,
        "DensitySurface.set_level DensitySlice.set_slice",
    )
    add(
        "Additional API paths",
        "Time-series reveal / missing samples",
        plot_reveal,
        "TimeSeriesPlot",
        video=True,
        times=(0.5, 2, 3.8),
    )
    add(
        "Export integration",
        "Buried ribose · public movie export",
        BuriedRibose,
        "ProteinScene.render Renderer.enqueue Renderer.drain Thread Representation Reveal Conceal ShowSideChains Distance",
        scene=True,
        video=True,
        times=(4.2, 10.8, 18.5, 29.5),
        notes="Original buried-ribose film, including its original in-scene captions. This clip is composed from the actual scene.render() native and studio NV12/H.264 exports. The outer labels identify the backend; the original film's STUDIO RENDERER footer remains verbatim on both sides. Gold network uses N/O heavy-atom contacts ≤3.2 Å; H atoms and angles are absent.",
    )
    add(
        "Additional API paths",
        "Explicit correspondence / save and reload",
        explicit_match,
        "ContactMatch.from_pairs ContactMatch.save ContactMatch.load BackboneMorph",
        video=True,
    )
    add(
        "Additional API paths",
        "Sample density / frame map bounds",
        sampled_density,
        "DensityMap.sample DensityMap.bounds DensitySurface.positions DensitySlice.positions Camera.frame",
    )
    return out
