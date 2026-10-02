# Morphs and contact matching

Morph between two experimental structures of the same protein with `StructureMorph`, deform coordinates with `Deform` and `Morph`, or morph different proteins with `BackboneMorph` and contact matching.

## Morph between two structures of the same protein

`StructureMorph(source, target)` animates the change between two deposited states, such as deoxy and oxy hemoglobin or open and closed adenylate kinase. The files can differ in chains, missing residues, ligands, and atoms. Requires ProteinMotion 0.14.0 or later. Upgrade with `python -m pip install --upgrade "proteinmotion>=0.14.0"`.

```python
deoxy = Protein.from_file("2dn2.cif").cartoon()
oxy = Protein.from_file("2dn1.cif", assembly="1").cartoon()
dimer = deoxy.select(chain=["A", "B"])
motion = domain_motion(deoxy, oxy, moving=deoxy.select(chain=["C", "D"]), fixed=dimer)
print(f"α2β2 turns {motion.angle:.1f}°")    # 14.1°
deoxy.center()
self.add(deoxy)
self.camera.look_along(motion.axis, deoxy)
self.camera.frame(deoxy, margin=1.1, aspect=self.width / self.height)
self.play(StructureMorph(deoxy, oxy, align=dimer), run_time=6)
```

```bash output=gallery-hemoglobin-morph
proteinmotion render examples/structure_morph.py HemoglobinMorph \
  -o hemoglobin-morph.mp4 --fps 60
```

```bash output=gallery-adenylate-kinase-morph
proteinmotion render examples/structure_morph.py AdenylateKinaseMorph \
  -o adenylate-kinase-morph.mp4 --fps 60
```

The same example file morphs *S. pyogenes* Cas9 from its guide-RNA complex (PDB 4ZT0) to the R-loop complex with target DNA (PDB 5F9R), about 13,600 atoms. It needs no `align` region: superposed on the parts that stay put, the HNH nuclease domain turns 117° and REC2 64° while the DNA fades in. The scene downloads both entries on first use.

```python
bound = Protein.fetch("4ZT0", chains=["A", "B"]).cartoon()   # Cas9 and guide RNA
rloop = Protein.fetch("5F9R").cartoon()                      # with target DNA
hnh = bound.select(chain="A", residues=(775, 908))
turn = domain_motion(bound, rloop, moving=hnh)                # 117°
self.play(StructureMorph(bound, rloop), run_time=7)
```

```bash
proteinmotion render examples/structure_morph.py Cas9Morph -o cas9.mp4 --fps 60
```

`match_structures(source, target)` builds the correspondence:

- Chains with the same ID and sequence are paired. Identical copies, as in a homo-oligomer, are paired by position after the first pair is superposed. `chains={"A": "C"}` sets pairs explicitly.
- Residues are paired by author number when the numbering agrees, otherwise by sequence alignment. Aligned runs shorter than four residues between gaps are left unpaired, since such fragments, like a loop matched to part of a longer stem, rarely occupy the same place. Their atoms fade instead of flying across the morph.
- Atoms are paired by name. Symmetric names, such as phenylalanine CD1/CD2 or carboxylate oxygens, are paired the way that lies closer, so rings do not flip.
- Ligands, ions, and waters are paired by chain, name, and number, such as the four hemes in hemoglobin.

By default, the target is superposed on the rigid core: the matched Cα (or C4′) atoms that superpose within 2 Å after the worst pairs are pruned and the fit is repeated, as in ChimeraX matchmaker. Moving domains then do not shift the frame. `match.report` gives the core's size and RMSD. `align=region` superposes on that part of the source instead, such as α1β1 or the adenylate kinase CORE. Each matched residue and ligand then follows a screw path: the rotation and translation that carry it to its target position are applied in proportion. A rigid domain therefore turns along an arc and keeps its bond lengths, instead of moving along a straight line that shortens its bonds. Side chains turn about their bonds toward the target χ angles. Bond lengths then relax toward values interpolated between the two structures. In the adenylate kinase example, no bond is more than 0.10 Å from its length at the start of the morph at the halfway point, compared with 1.42 Å for linear interpolation.

| Option | Default | Meaning |
|---|---|---|
| `align` | rigid core | Region of the source used to superpose the target |
| `chains` | automatic | Source-to-target chain IDs |
| `fade_out`, `fade_in` | (0, 0.35), (0.65, 1) | When atoms in only one structure fade, as fractions of the clip |
| `style` | `"source"` | The target takes the source's representation and colors; `"target"` keeps its own |
| `secondary` | `"auto"` | Blend residues whose DSSP state differs between the structures |
| `easing` | `"smooth"` | Easing of the whole morph |

Atoms in only one structure move with their residue, or the nearest matched residue, while they fade. At the end, the target replaces the source in the source's place; continue animating `morph.result`, which is the target. Hide crystallization additives on the target before the morph, for example `oxy.select(resname="MBN").hide_atoms()`.

`domain_motion(source, target, moving, fixed=None)` reports how a domain turns between the structures, after superposing on `fixed` (by default the rigid core): `angle` in degrees, `axis` as a unit vector in the source's frame, `point` on the axis, and `translation` along it in Å. `camera.look_along(axis, protein)` points the view down that axis, so the domain turns in the image plane.

`Protein.from_file(path, assembly="1")` builds a biological assembly from the deposited chains; R-state hemoglobin entries such as 2DN1 deposit one αβ dimer. Generated copies get new chain IDs and keep their helix and sheet records. `Protein.fetch("2DN1", assembly="1")` downloads a PDB entry, or an AlphaFold DB model such as `"AF-P0DP23-F1"`, once and keeps it in `~/.cache/proteinmotion` (or `PROTEINMOTION_CACHE`).

The morph shows how the two endpoints differ. The path between them is an illustration, not a calculated transition pathway: it does not avoid clashes, so a domain that turns far, such as the Cas9 HNH domain, can pass partly through its neighbors on the way.

## Deformations and same-topology morphs

The rendered excerpts use [docs_examples.py](docs_examples.py). Its `ubiquitin()` helper loads PDB 1UBQ, and `frame()` adds the model and sets the camera. Run each excerpt inside `construct()`; the full script includes the imports and setup. Run the full script from a repository checkout, which includes the input structures.

```python output=deformation
p = ubiquitin()
frame(self, p, margin=1.05)
rest = p.positions.copy()

def stretch(xyz):
    center = xyz.mean(axis=0)
    return center + (xyz - center) * [1.4, 1, 1]

self.play(Deform(p, stretch), run_time=2)
self.wait(0.5)
self.play(Morph(p, rest, align=False), run_time=2)
self.wait(0.5)
```

`Morph` matches atoms by `(chain, residue number, insertion code, residue name, atom name)`. Input atom order can differ. By default, it aligns Cα atoms for proteins and C1′ atoms for DNA/RNA with a Kabsch rigid fit. A nucleotide fit requires at least three anchors; use `align=False` when those coordinates are unavailable. Protein-only inputs retain the all-atom fallback when fewer than three Cα atoms exist.

Use `atom_map` to map every source atom to a unique target index. A raw coordinate array must follow source atom order. `Morph` keeps the source topology fixed and interpolates coordinates for visualization.

## DNA and RNA morphs

Use the same `match_backbones` and `BackboneMorph` calls with `NucleicAcid` objects. C1′ positions define the contact maps, rigid alignment, and delayed motion. Every atom in a matched nucleotide moves with its C1′; the sugar, phosphate, and base retain each endpoint’s internal geometry as their representations crossfade. Cartoon traces continue to use C4′.

Select one strand per endpoint with `source_chain` and `target_chain`, or load it with `chains="A"`. Delays follow deposited residue order, usually 5′ to 3′. Residues missing C1′ are excluded from automatic matching and join the unmatched fades. An explicit mapping that includes such a residue raises an error. Legacy `C1*` atom names are accepted.

The match report records `source_anchor_atom`, `target_anchor_atom`, `source_anchor_count`, `target_anchor_count`, and `aligned_anchor_rmsd_angstrom`. Protein reports also keep the older Cα field names. Saved matches record the anchor choice and validate it when used.

See the [DNA morph script and rendered result](https://pdpppd.github.io/proteinmotion/docs/dna-rna/#dna-morph) for an example using two deposited structures.

## Contact-guided backbone morphing

`BackboneMorph` handles proteins or nucleic acids with different sequences, residue counts, and atom sets. It uses Cα for proteins and C1′ for DNA/RNA to compare contact maps, align the structures, and move matched residues. Unmatched residues fade. Each endpoint must select the same polymer type.

```python
from proteinmotion import BackboneMorph, Protein, Rotate, match_backbones

# Inside construct():
source = Protein.from_file("examples/data/1cll.cif", chains="A").cartoon().center()
target = Protein.from_file("examples/data/1ncx.cif", chains="A").cartoon().center()
pairs = match_backbones(
    source, target,
    cutoff=8.0,                # Å, contact midpoint
    softness=1.5,              # Å, contact sigmoid width
    max_contact_error=0.30,    # maximum absolute discrepancy for EVERY selected pair
    search_seconds=15,        # branch-and-bound time budget; preparation is extra
    max_candidates=6000,      # None considers all pairings, subject to the size limit
)
pairs.save("correspondence.json")
print(pairs.report)            # coverage, contact errors, RMSD, search/proof status
self.add(source)              # play() adds the destination at the morph's start
self.camera.frame(source, target)
self.play(
    BackboneMorph(
        source, target, match=pairs,
        residue_delay=0.025,  # seconds between successive selected residues
        motion_easing="smooth",  # quintic, or "linear"
        fade_out=(0.0, 0.35), # unmatched source; fractions of the morph duration
        fade_in=(0.65, 1.0),  # unmatched target
        align=True,          # proper rigid alignment on the selected Cα pairs
    ),
    run_time=6,
)
self.play(Rotate(target, 1.0), run_time=2)
```

Omit `match` to run the search automatically and pass matching options to `BackboneMorph`. Use `ContactMatch.load(path)` to reuse a saved mapping.

For a manual mapping, call `ContactMatch.from_pairs(source, target, [0, 2, 5], [1, 4, 7])`. These lists contain **zero-based topology residue indices**. Saved mappings also store residue identities, which are checked when loaded.

For selected correspondence `(iₖ, jₖ)`, the soft contact map is
`C(i,j) = 1 / (1 + exp((distance(i,j) - cutoff) / softness))`, with zero diagonal.
The objective is lexicographic: **maximize the number of matched residues** subject
to `abs(Csource(iₖ,iₗ) - Ctarget(jₖ,jₗ)) <= max_contact_error` for every selected
pair, then **minimize the sum of squared contact errors** among equal-sized sets.
Both source and destination indices increase strictly in deposited chain order:
N to C for proteins, normally 5′ to 3′ for nucleic acids. This gives an injective,
order-preserving correspondence. All off-diagonal contacts, including sequence
neighbors, participate. Error is dimensionless; RMSD is separately reported in Å.

The search starts from contact fingerprints, structural fragments, rigid fits, and dynamic programming. It builds a graph of compatible residue pairs and uses branch-and-bound to search for a larger set.

The search has a time and candidate limit. Check `full_candidate_space`, `search_completed`, `cardinality_proved`, and `global_optimal` in the report to see what was established. Small completed searches can prove the optimum. Larger searches may return a feasible result before reaching it.

`max_candidates=None` considers all pairings up to 30,000 graph vertices. At a fixed time budget, this can return fewer matches than a restricted search. Numerical comparisons use feasibility tolerance 1e-10 and objective tolerance 1e-14.

The calmodulin → troponin C example matches **114 of 144 source Cα residues** and **114 of 162 target Cα residues**. Its RMS soft-contact error is **0.02746** and maximum error is **0.29362**, within the 0.30 limit. The report marks global optimality as unproved.

Thirty source residues fade out and 48 target residues fade in. With a 25 ms delay, the last matched residue starts 2.825 s after the first. Each residue moves for 3.175 s during the 6 s morph.

For `K` matched pairs, each move lasts `run_time - (K-1)*residue_delay` seconds. This value must be positive. Each matched source and target anchor follows the same path. Their representations crossfade along that path.

The GPU stores endpoint coordinates and residue timing. Playback updates animation parameters and supports backward seeking.

`BackboneMorph` supports **cartoon, nucleotide base styles, ribbon, ball-and-stick, and surfaces**, with one selected chain per endpoint. The shorter chain can contain at most 800 residues with the required anchor atom. For multi-chain inputs, set `source_chain` and `target_chain`; remaining residues join the unmatched fades. Correspondences must preserve residue order.

The morph controls both proteins’ transforms, coordinates, and opacity. Use camera motion for rotation during the morph. With `align=False`, the target finishes at its original world coordinates. With `align=True`, it finishes in the aligned position. Continue the scene by animating the target object; the source is hidden.

The interpolated path can contain stretched bonds and atomic clashes. It describes a visual transition rather than a calculated molecular pathway.

For ball-and-stick, call `.ball_and_stick()` on both proteins before the morph. Each matched residue moves with its Cα or C1′ and keeps its internal geometry. Source atoms fade out as target atoms fade in. The correspondence is between residues; different side-chain atoms have no individual mapping.

Bonds use the lower opacity of their endpoints. Inter-residue bonds can stretch during the transition. The `BallAndStickDemo` and `BackboneDemo` examples use the same residue mapping, camera, and timing.

```bash output=gallery-morph
proteinmotion render examples/backbone_morph.py BackboneDemo -o backbone-morph.mp4 --fps 60
```

```bash output=gallery-morph-atoms
proteinmotion render examples/backbone_morph.py BallAndStickDemo -o ball-and-stick-morph.mp4 --fps 60
```

```bash
proteinmotion render examples/large_protein.py GroELSubunit -o groel-subunit.mp4 --fps 60
```

```bash output=gallery-groel
proteinmotion render examples/large_protein.py GroELComplex -o groel-complex.mp4 --fps 60
```

```bash
pip install -e '.[plots]'
python examples/contact_report.py --output contact-report
```

The larger examples use the actual 524-residue GroEL chain and the complete
21-chain GroEL/GroES coordinates: **8,015 Cα residues, 58,870 selected heavy atoms**.
The complex movie rotates the cartoon, then switches to ball-and-stick.
