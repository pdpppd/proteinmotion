# Morphs and contact matching

```python
target = Protein.from_file("alternate-conformation.cif")
self.play(Morph(p, target, align=True), run_time=2)

rest = p.positions.copy()


def stretch(xyz):
    center = xyz.mean(axis=0)
    return center + (xyz - center) * [1.25, 1, 1]


self.play(Deform(p, stretch), run_time=2)
self.play(Morph(p, rest, align=False), run_time=2)
```

`Morph` matches atoms by `(chain, residue number, insertion code, residue name, atom name)`. Input atom order can differ. By default, it aligns matched Cα atoms with a Kabsch rigid fit. It uses all atoms if Cα atoms are insufficient.

Use `atom_map` to map every source atom to a unique target index. A raw coordinate array must follow source atom order. `Morph` keeps the source topology fixed and interpolates coordinates for visualization.

## Different proteins: contact-guided backbone morphing

`BackboneMorph` handles proteins with different sequences, residue counts, and atom sets. It matches Cα atoms using their structural contacts, moves the selected residues, and fades the unmatched residues.

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
Both source and destination indices increase strictly N-to-C, giving an injective,
order-preserving correspondence. All off-diagonal contacts, including sequence
neighbors, participate. Error is dimensionless; RMSD is separately reported in Å.

The search starts from contact fingerprints, structural fragments, rigid fits, and dynamic programming. It builds a graph of compatible residue pairs and uses branch-and-bound to search for a larger set.

The search has a time and candidate limit. Check `full_candidate_space`, `search_completed`, `cardinality_proved`, and `global_optimal` in the report to see what was established. Small completed searches can prove the optimum. Larger searches may return a feasible result before reaching it.

`max_candidates=None` considers all pairings up to 30,000 graph vertices. At a fixed time budget, this can return fewer matches than a restricted search. Numerical comparisons use feasibility tolerance 1e-10 and objective tolerance 1e-14.

The calmodulin → troponin C example matches **114 of 144 source Cα residues** and **114 of 162 target Cα residues**. Its RMS soft-contact error is **0.02746** and maximum error is **0.29362**, within the 0.30 limit. The report marks global optimality as unproved.

Thirty source residues fade out and 48 target residues fade in. With a 25 ms delay, the last matched residue starts 2.825 s after the first. Each residue moves for 3.175 s during the 6 s morph.

For `K` matched pairs, each move lasts `run_time - (K-1)*residue_delay` seconds. This value must be positive. Each matched source and target Cα follows the same path. Their representations crossfade along that path.

The GPU stores endpoint coordinates and residue timing. Playback updates animation parameters and supports backward seeking.

`BackboneMorph` supports **cartoon, ribbon, and ball-and-stick**, with one selected chain per endpoint. The shorter chain can contain at most 800 Cα residues. For multi-chain inputs, set `source_chain` and `target_chain`; remaining residues join the unmatched fades. Correspondences must preserve residue order.

The morph controls both proteins’ transforms, coordinates, and opacity. Use camera motion for rotation during the morph. With `align=False`, the target finishes at its original world coordinates. With `align=True`, it finishes in the aligned position. Continue the scene by animating the target object; the source is hidden.

The interpolated path can contain stretched bonds and atomic clashes. It describes a visual transition rather than a calculated molecular pathway.

For ball-and-stick, call `.ball_and_stick()` on both proteins before the morph. Each matched residue moves with its Cα and keeps its internal geometry. Source atoms fade out as target atoms fade in. The correspondence is between residues; different side-chain atoms have no individual mapping.

Bonds use the lower opacity of their endpoints. Inter-residue bonds can stretch during the transition. The `BallAndStickDemo` and `BackboneDemo` examples use the same residue mapping, camera, and timing.

```bash
proteinmotion render examples/backbone_morph.py BackboneDemo -o backbone-morph.mp4 --fps 60
proteinmotion render examples/backbone_morph.py BallAndStickDemo -o ball-and-stick-morph.mp4 --fps 60
proteinmotion render examples/large_protein.py GroELSubunit -o groel-subunit.mp4 --fps 60
proteinmotion render examples/large_protein.py GroELComplex -o groel-complex.mp4 --fps 60
pip install -e '.[plots]'
python examples/contact_report.py --output contact-report
```

The larger examples use the actual 524-residue GroEL chain and the complete
21-chain GroEL/GroES coordinates: **8,015 Cα residues, 58,870 selected heavy atoms**.
The complex movie rotates the cartoon, then switches to ball-and-stick.
