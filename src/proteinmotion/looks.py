"""Named studio lighting, materials, and global film effects.

StudioLook holds every setting of the Studio renderer (studio.py). StudioRenderer compiles
the lights and the material into its scene shader when it is created (STUDIO_SHADE), and
reads the effect settings again for every frame. Scenes, the preview window and the CLI
(cli.py) build a look from preset names plus explicit overrides.

A preset fills only the fields that are still None, so the precedence is: explicit values,
then the grain, bloom and halation presets, then the ``effects`` recipe, then the lighting
and material presets, then the fallbacks in StudioLook.__post_init__.
"""

from dataclasses import dataclass, fields, replace

import numpy as np


class _PresetFloat(float):
    """Resolved default carrying its recipe through dataclasses.replace().

    The type marks a value that a preset filled in, as opposed to one the user set.
    ``recipe`` is what the field held before resolution: the preset name for the grain,
    bloom or halation field itself, None for every other filled-in field.
    """

    def __new__(cls, value, recipe=None):
        result = super().__new__(cls, value)
        result.recipe = recipe
        return result


class _PresetTuple(tuple):
    """A color or direction that a preset filled in (see _PresetFloat)."""


# Lighting presets. Every light is fixed to the camera, so a molecule stays lit from the
# same side of the screen while the camera orbits.
#   key, fill, rim      Intensity: a unitless multiplier of the light's color. Diffuse light
#                       is base color / π × intensity, so an intensity of about 3.14 lights a
#                       surface that faces a white light to its own base color. The rim light
#                       also adds a colored band along silhouettes.
#   *_color             Linear RGB color, multiplied by the intensity. Values may exceed 1.
#   *_direction         Direction from the surface toward the light, in camera axes: x to the
#                       right of the screen, y up, z toward the viewer. Negative z puts a light
#                       behind the molecule, as for the rim light. The shader normalizes it.
#   ambient, sky,       Hemisphere ambient light: base color × ambient × a blend from ground
#   ground              (normals pointing down the screen) to sky (pointing up). Unlike the
#                       lights, the ambient term is not divided by π.
#   ambient_occlusion,  Occlusion strength in [0, 1] for this lighting, and the bloom strength
#   bloom               used when no effects recipe or bloom preset sets one.
LIGHTING_PRESETS = {
    # Warm key from the upper left, cool fill, mint rim: the default film look.
    "studio": dict(
        key=3.4,
        key_color=(1.0, 0.94, 0.86),
        key_direction=(-0.55, 0.75, 0.62),
        fill=0.8,
        fill_color=(0.72, 0.83, 1.0),
        fill_direction=(0.85, -0.08, 0.45),
        rim=1.6,
        rim_color=(0.62, 1.0, 0.84),
        rim_direction=(0.30, 0.45, -0.85),
        ambient=0.75,
        sky=(0.30, 0.34, 0.40),
        ground=(0.09, 0.08, 0.08),
        ambient_occlusion=1.0,
        bloom=0.35,
    ),
    # Large, even light and bright ambient: low contrast, for figures and dense complexes.
    "soft": dict(
        key=2.0,
        key_color=(1.0, 0.98, 0.95),
        key_direction=(-0.35, 0.55, 0.76),
        fill=1.5,
        fill_color=(0.92, 0.95, 1.0),
        fill_direction=(0.70, 0.10, 0.70),
        rim=0.6,
        rim_color=(0.95, 0.97, 1.0),
        rim_direction=(0.20, 0.60, -0.78),
        ambient=1.35,
        sky=(0.36, 0.38, 0.42),
        ground=(0.20, 0.19, 0.18),
        ambient_occlusion=0.7,
        bloom=0.2,
    ),
    # Hard key from high on the side, little fill, strong cool rim: deep shadows and silhouettes.
    "dramatic": dict(
        key=5.0,
        key_color=(1.0, 0.90, 0.78),
        key_direction=(-0.85, 0.50, 0.20),
        fill=0.2,
        fill_color=(0.60, 0.72, 1.0),
        fill_direction=(0.90, -0.20, 0.40),
        rim=3.2,
        rim_color=(0.55, 0.78, 1.0),
        rim_direction=(0.60, 0.35, -0.72),
        ambient=0.3,
        sky=(0.22, 0.26, 0.34),
        ground=(0.04, 0.04, 0.05),
        ambient_occlusion=1.0,
        bloom=0.5,
    ),
    # Light from the camera and strong ambient: nearly uniform colors, like an illustration.
    "flat": dict(
        key=1.1,
        key_color=(1.0, 1.0, 1.0),
        key_direction=(0.0, 0.25, 1.0),
        fill=0.0,
        fill_color=(1.0, 1.0, 1.0),
        fill_direction=(1.0, 0.0, 0.0),
        rim=0.35,
        rim_color=(1.0, 1.0, 1.0),
        rim_direction=(0.0, 0.5, -0.86),
        ambient=2.1,
        sky=(0.34, 0.34, 0.34),
        ground=(0.30, 0.30, 0.30),
        ambient_occlusion=0.45,
        bloom=0.1,
    ),
}

# Material presets: base roughness, specular strength, and a clearcoat layer.
#   roughness             GGX roughness in [0.02, 1], squared in the shader (alpha =
#                         roughness²). Low values give small, sharp highlights.
#   specular              Strength of the base specular lobe, whose Fresnel reflectance is
#                         4% at normal incidence, as for a typical dielectric. It also scales
#                         a faint reflection of the ambient sky.
#   clearcoat             Strength of a second specular lobe added on top, like a lacquer.
#   clearcoat_roughness   GGX roughness of the clearcoat lobe.
MATERIAL_PRESETS = {
    "rough": dict(roughness=0.85, specular=0.45, clearcoat=0.0, clearcoat_roughness=0.4),
    "medium": dict(roughness=0.55, specular=0.8, clearcoat=0.25, clearcoat_roughness=0.3),
    "glossy": dict(roughness=0.38, specular=1.0, clearcoat=0.85, clearcoat_roughness=0.18),
}

# Named recipes can be combined, or overridden with a numeric strength (0 disables).
# Thresholds compare the luminance of the scene's HDR color, in which 1 is display white.
#   grain                 Amplitude of monochrome noise in display values (0 to 1), largest
#                         in midtones. grain_size is the noise cell in 1080p pixels.
#   bloom                 Multiplier of the blurred highlights added onto the image.
#                         bloom_threshold is where blooming starts (with a soft knee);
#                         bloom_radius widens the blur filter (1 is the standard tent).
#   halation              Multiplier of the warm fringe around the outside of bright edges.
#                         halation_threshold is where bright areas start to spill;
#                         halation_radius is the Gaussian blur sigma in 1080p pixels.
GRAIN_PRESETS = {
    "off": dict(grain=0.0),
    "fine": dict(grain=0.018, grain_size=0.75),
    "film": dict(grain=0.04, grain_size=1.2),
    "coarse": dict(grain=0.07, grain_size=1.8),
}
BLOOM_PRESETS = {
    "off": dict(bloom=0.0),
    "soft": dict(bloom=0.25, bloom_threshold=0.85, bloom_radius=1.0),
    "strong": dict(bloom=0.65, bloom_threshold=0.65, bloom_radius=1.4),
}
HALATION_PRESETS = {
    "off": dict(halation=0.0),
    "subtle": dict(halation=0.2, halation_threshold=0.8, halation_radius=3.0),
    "warm": dict(halation=0.5, halation_threshold=0.65, halation_radius=6.0),
}
# Global recipes (``effects=``) that set grain, bloom and halation together.
EFFECTS_PRESETS = {
    "clean": dict(grain=0.0, bloom=0.0, halation=0.0),
    "subtle": dict(grain=0.014, grain_size=0.75, bloom=0.2, halation=0.12),
    "film": dict(
        grain=0.04, grain_size=1.2, bloom=0.35, halation=0.4, halation_threshold=0.7, halation_radius=5.0
    ),
    "dreamy": dict(
        grain=0.025,
        grain_size=1.0,
        bloom=0.8,
        bloom_threshold=0.65,
        bloom_radius=1.5,
        halation=0.5,
        halation_threshold=0.65,
        halation_radius=8.0,
    ),
}


@dataclass
class StudioLook:
    """Studio lighting, material and post-processing.

    Choose a ``lighting`` preset (studio, soft, dramatic, flat) and a ``material`` preset
    (rough, medium, glossy). Global ``effects`` recipes are clean, subtle, film and
    dreamy. Each effect also accepts a named preset or numeric strength, e.g.
    ``StudioLook(lighting="dramatic", material="medium", grain="fine",
    bloom="soft", halation="warm")``. Zero disables an effect.

    Lights follow the camera. Their directions are in camera axes (x right, y up, z toward
    the viewer) and point toward the light. Light colors are linear RGB multipliers, and the
    ``key``, ``fill`` and ``rim`` intensities scale them: about 3.14 lights a surface facing
    a white light to its own base color. ``sky`` and ``ground`` color the ambient light
    from above and below the screen, scaled by ``ambient``.

    Explicit controls override individual effect presets, which override the global
    recipe. With no effects recipe, the original lighting-dependent bloom is retained.
    Grain size and halation radius use 1080p design pixels and scale with output height.
    Grain is monochrome, strongest in midtones, and seeded by scene time at grain_fps.
    Surfaces use the shared VDW/SAS/SES molecular mesh by default. Set
    ``surface_mode="density"`` for the original artistic GPU density surface instead;
    only that mode uses surface_inflation, surface_blobbiness and surface_voxels.
    ``cartoon_style="classic"`` uses smooth helix bands, flat sheet arrows and
    round loop tubes. ``"legacy"`` retains the original elliptical sweep.
    """

    # Preset names. A field left as None below takes its value from a preset.
    lighting: str = "studio"
    material: str = "glossy"
    effects: str | None = None
    # Lights and ambient (see LIGHTING_PRESETS), compiled into the scene shader.
    key: float = None
    key_color: tuple = None
    key_direction: tuple = None
    fill: float = None
    fill_color: tuple = None
    fill_direction: tuple = None
    rim: float = None
    rim_color: tuple = None
    rim_direction: tuple = None
    ambient: float = None
    sky: tuple = None
    ground: tuple = None
    # Material (see MATERIAL_PRESETS), compiled into the scene shader.
    roughness: float = None
    specular: float = None
    clearcoat: float = None
    clearcoat_roughness: float = None
    # Post-processing (see GRAIN_PRESETS), read again for every frame.
    ambient_occlusion: float = None
    bloom: float | str | None = None
    bloom_threshold: float | None = None
    bloom_radius: float | None = None
    grain: float | str | None = None
    grain_size: float | None = None
    grain_fps: float = 24.0  # Grain pattern changes per second of scene time
    grain_seed: int = 0
    halation: float | str | None = None
    halation_threshold: float | None = None
    halation_radius: float | None = None
    halation_color: tuple = (1.0, 0.22, 0.045)  # RGB of the halation fringe
    knee: float = 0.72  # Brightest channel above which highlights roll off toward 1
    vignette: float = 0.0  # Strength in [0, 1]; corners blend toward the background color
    # Surfaces and cartoons. surface_mode and cartoon_style are fixed when the renderer is
    # created; the density-mode settings are read for every frame.
    surface_mode: str = "molecular"
    surface_inflation: float = 0.45  # Å added to each atom radius in density mode
    surface_blobbiness: float = 1.6  # Density falloff per atom; higher merges atoms less
    surface_voxels: int = 3_000_000  # Approximate voxel budget of a density-mode grid
    cartoon_style: str = "classic"

    def __post_init__(self):
        # While _resolving is set, __setattr__ assigns plainly instead of re-resolving.
        object.__setattr__(self, "_resolving", True)
        # dataclasses.replace() passes resolved values back in. Turn every value that a
        # preset filled in back into what the field held before (None, or the preset name),
        # so the presets apply afresh and only explicit values carry over.
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, (_PresetFloat, _PresetTuple)):
                object.__setattr__(self, item.name, getattr(value, "recipe", None))

        def apply(kind, name, presets):
            """Fill the fields that are still None from ``presets[name]``."""
            if not isinstance(name, str) or name not in presets:
                raise ValueError(f"Unknown {kind} preset {name!r}; choose from {', '.join(presets)}")
            for field_name, value in presets[name].items():
                if getattr(self, field_name) is None:
                    resolved = _PresetTuple(value) if isinstance(value, tuple) else _PresetFloat(value)
                    # The grain, bloom or halation strength remembers its preset's name.
                    if field_name == kind and isinstance(resolved, _PresetFloat):
                        resolved.recipe = name
                    object.__setattr__(self, field_name, resolved)

        # Most specific first; each step fills only what earlier steps left as None.
        # A preset name in grain, bloom or halation is replaced by the values it names.
        for kind, presets in (
            ("grain", GRAIN_PRESETS),
            ("bloom", BLOOM_PRESETS),
            ("halation", HALATION_PRESETS),
        ):
            name = getattr(self, kind)
            if isinstance(name, str):
                setattr(self, kind, None)
                apply(kind, name, presets)
        if self.effects is not None:
            apply("effects", self.effects, EFFECTS_PRESETS)
        for kind, name, presets in (
            ("lighting", self.lighting, LIGHTING_PRESETS),
            ("material", self.material, MATERIAL_PRESETS),
        ):
            apply(kind, name, presets)
        # Effect settings that no preset set. They are marked as preset values too, so a
        # later with_presets() can still fill them from a new recipe.
        defaults = dict(
            bloom_threshold=0.85,
            bloom_radius=1.0,
            grain=0.0,
            grain_size=1.0,
            halation=0.0,
            halation_threshold=0.8,
            halation_radius=4.0,
        )
        for name, value in defaults.items():
            if getattr(self, name) is None:
                object.__setattr__(self, name, _PresetFloat(value))
        self.validate()
        object.__setattr__(self, "_resolving", False)

    def __setattr__(self, name, value):
        # After construction, assigning lighting, material or effects, or a preset name to
        # grain, bloom or halation, re-resolves the look as with_presets() does. Other values
        # are stored as explicit overrides. The dataclass __init__ runs before _resolving
        # exists, so its assignments are plain.
        if not getattr(self, "_resolving", True) and (
            name in ("lighting", "material", "effects")
            or (name in ("grain", "bloom", "halation") and isinstance(value, str))
        ):
            # Resolve a replacement first so an invalid preset leaves this look intact.
            updated = self.with_presets(**{name: value})
            self.__dict__.update(updated.__dict__)
        else:
            object.__setattr__(self, name, value)

    def with_presets(self, **changes):
        """Return a new look, reapplying changed presets while retaining explicit overrides."""
        return replace(self, **changes)

    def validate(self):
        """Reject invalid values before creating GPU resources or compiling shaders."""
        if self.surface_mode not in ("molecular", "density"):
            raise ValueError("surface_mode must be 'molecular' or 'density'")
        if self.cartoon_style not in ("classic", "legacy"):
            raise ValueError("cartoon_style must be 'classic' or 'legacy'")
        ranges = {
            "grain": (0, 1),
            "grain_size": (0.2, 8),
            "grain_fps": (0.01, 240),
            "bloom": (0, 4),
            "bloom_threshold": (0, 16),
            "bloom_radius": (0.1, 4),
            "halation": (0, 4),
            "halation_threshold": (0, 16),
            "halation_radius": (0.2, 64),
            "roughness": (0.02, 1),
            "clearcoat_roughness": (0.02, 1),
            "ambient_occlusion": (0, 1),
            "vignette": (0, 1),
            "knee": (0, 0.999),
            "surface_inflation": (0, 10),
            "surface_blobbiness": (0.01, 20),
        }
        # Light, ambient and specular strengths share one generous range.
        ranges.update({name: (0, 100) for name in ("key", "fill", "rim", "ambient", "specular", "clearcoat")})
        for name, (lo, hi) in ranges.items():
            value = getattr(self, name)
            if (
                not isinstance(value, (int, float, np.number))
                or not np.isfinite(value)
                or not lo <= value <= hi
            ):
                raise ValueError(f"{name} must be finite and in [{lo}, {hi}]")
        for name in (
            "key_color",
            "fill_color",
            "rim_color",
            "sky",
            "ground",
            "halation_color",
            "key_direction",
            "fill_direction",
            "rim_direction",
        ):
            # Colors may be given as "#RRGGBB"; store them as RGB tuples in [0, 1].
            if not name.endswith("direction") and isinstance(getattr(self, name), str):
                from .math3d import color

                object.__setattr__(self, name, tuple(color(getattr(self, name))))
            try:
                value = np.asarray(getattr(self, name), dtype=float)
            except (ValueError, TypeError) as error:
                raise ValueError(f"{name} needs three finite numbers") from error
            if value.shape != (3,) or not np.isfinite(value).all():
                raise ValueError(f"{name} needs three finite numbers")
            # The shader normalizes directions, so only a zero vector is invalid.
            if name.endswith("direction"):
                if np.linalg.norm(value) < 1e-6:
                    raise ValueError(f"{name} must be nonzero")
            elif np.any(value < 0):
                raise ValueError(f"{name} must be nonnegative")
        # The grain seed is uploaded as a uint32.
        for name, lo, hi in (("grain_seed", 0, 2**32 - 1), ("surface_voxels", 1, 2**31 - 1)):
            value = getattr(self, name)
            if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or not lo <= value <= hi:
                raise ValueError(f"{name} must be an integer in [{lo}, {hi}]")
        return self
