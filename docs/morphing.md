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

`Morph` matches `(chain, residue number, insertion code, residue name, atom name)`.
Input atom order can differ. By default it uses a proper Kabsch rigid alignment on
matched alpha carbons, falling back to all atoms when necessary. Explicit `atom_map`
supports mapping every source atom to a unique target index. Output topology stays
fixed: unmatched atoms are not created/deleted. Raw coordinate targets are explicitly
ordered like the source. Morphs and deformations are **visual interpolation**, not
energy minimization, a physical transition pathway, or an MD simulation.

## Different proteins: contact-guided backbone morphing

`BackboneMorph` supports different sequences, residue counts, and atom topologies.
It moves a selected Cα correspondence and fades unmatched residues. The matcher uses
structure/contact information, without requiring sequence identity.

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

Omit `match` to search automatically; pass the same matching options directly to
`BackboneMorph`. `ContactMatch.load(path)` reuses a saved mapping.
`ContactMatch.from_pairs(source, target, [0, 2, 5], [1, 4, 7])` accepts a manual
mapping. These are **zero-based topology residue indices**, not atom indices or PDB
residue numbers. Saved mappings also contain residue identities, validated on use.

For selected correspondence `(iₖ, jₖ)`, the soft contact map is
`C(i,j) = 1 / (1 + exp((distance(i,j) - cutoff) / softness))`, with zero diagonal.
The objective is lexicographic: **maximize the number of matched residues** subject
to `abs(Csource(iₖ,iₗ) - Ctarget(jₖ,jₗ)) <= max_contact_error` for every selected
pair, then **minimize the sum of squared contact errors** among equal-sized sets.
Both source and destination indices increase strictly N-to-C, giving an injective,
order-preserving correspondence. All off-diagonal contacts, including sequence
neighbors, participate. Error is dimensionless; RMSD is separately reported in Å.

Feasible seeds come from contact/distance fingerprints, local structural fragments,
rigid fits, and dynamic programming. A compatibility graph represents allowed pairings;
bitset branch-and-bound searches cliques with coloring upper bounds. This is a
combinatorial optimization, so a finite budget does **not** promise a global maximum.
Check `full_candidate_space`, `search_completed`, `cardinality_proved`, and
`global_optimal` in the report. Candidate pruning and timeouts are reported explicitly.
`max_candidates=None` searches the full pairing space, up to 30,000 vertices; it can
spend more time branching and return a poorer incumbent than a restricted search at
the same time budget. Small completed searches can prove the optimum. Numerical
comparisons use 1e-10 feasibility and 1e-14 objective tolerances.

The included calmodulin → troponin C example matches **114 of 144 source Cα residues
and 114 of 162 target Cα residues**, with RMS soft-contact error **0.02746** and
maximum error **0.29362**, below 0.30. Global optimality was not proved. Thirty source
residues fade out; 48 target residues fade in. At 25 ms delay, the last matched
residue starts 2.825 s after the first; each moves for 3.175 s in the 6 s morph.

For `K` pairs, each residue's move lasts `run_time - (K-1)*residue_delay`, which must
be positive. Matching source/destination Cα positions follow the same world-space
path; smooth alpha fades exchange source/target ribbon connectivity and
color along that path. GPU buffers hold the coordinate endpoints and per-residue
timing/visibility controls; normal playback changes uniforms rather than rebuilding
geometry each frame. Backward seeking reproduces the same motion.

This operation supports **cartoon, ribbon, and ball-and-stick** representations, one selected chain
per endpoint, and at most 800 Cα residues in the shorter chain. Multi-chain inputs
need `source_chain`/`target_chain`; other residues fade with the unmatched sets.
Domain permutations and crossing correspondences are not supported. Use the camera
for concurrent rotation: the morph owns both endpoint transforms, coordinates, and
opacity. `align=False` ends at the destination's original world coordinates;
`align=True` ends at its aligned pose. Subsequent animations should target the
destination object. The source remains hidden. The path is a visual interpolation;
bond lengths, clashes, energies, and physical kinetics are not constrained.

For an atom-level view, call `.ball_and_stick()` on both endpoints before the morph.
Every atom in a matched residue translates with its Cα; each endpoint keeps its
internal residue geometry, while the source and target atom sets crossfade. Different
side-chain atoms are not assigned an atom-to-atom correspondence. Bonds fade with their
less-visible endpoint so no half-bonds remain attached to hidden atoms. Inter-residue
bonds can stretch during the visual transition; this mode is not an all-atom simulation.
The `BallAndStickDemo` scene uses the same mapping, camera, and timing as `BackboneDemo`.

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
