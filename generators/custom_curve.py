"""Custom curve generator with spline interpolation."""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.integrate import cumulative_trapezoid

from generators.base import BaseGenerator
from models.labels import LabelConfig


@dataclass
class ControlPoint:
    """A control point on the probability curve."""
    x: float  # Position (0-1 normalized)
    y: float  # Probability weight (0-1)

    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)


@dataclass
class CurveConfig:
    """Configuration for a custom probability curve."""
    points: List[ControlPoint] = field(default_factory=list)
    flatten_amount: float = 0.0  # -1 (sharpen) to 1 (flatten toward uniform)

    def __post_init__(self):
        if not self.points:
            # Default: uniform distribution
            self.points = [
                ControlPoint(0.0, 0.5),
                ControlPoint(0.5, 0.5),
                ControlPoint(1.0, 0.5),
            ]

    def get_points_array(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get x and y arrays from control points."""
        sorted_points = sorted(self.points, key=lambda p: p.x)
        x = np.array([p.x for p in sorted_points])
        y = np.array([p.y for p in sorted_points])
        return x, y

    def apply_flatten(self, y: np.ndarray) -> np.ndarray:
        """Apply flatten/sharpen transformation."""
        if abs(self.flatten_amount) < 0.01:
            return y

        uniform = np.ones_like(y) * np.mean(y)

        if self.flatten_amount > 0:
            # Flatten toward uniform
            return y * (1 - self.flatten_amount) + uniform * self.flatten_amount
        else:
            # Sharpen: exaggerate peaks
            centered = y - np.mean(y)
            sharpened = np.mean(y) + centered * (1 - self.flatten_amount)
            return np.clip(sharpened, 0.01, None)

    def with_bias(self, bias: float) -> 'CurveConfig':
        """
        Create a new config with shifted bias.
        bias: -1.0 (shift toward low) to 1.0 (shift toward high)
        """
        if abs(bias) < 0.01:
            return self

        new_points = []
        for p in self.points:
            # Shift x positions based on bias
            if bias > 0:
                # Shift toward high: compress low end, expand high end
                new_x = p.x ** (1 - bias * 0.7)
            else:
                # Shift toward low: expand low end, compress high end
                new_x = 1 - (1 - p.x) ** (1 + bias * 0.7)
            new_x = max(0.0, min(1.0, new_x))
            new_points.append(ControlPoint(new_x, p.y))

        return CurveConfig(points=new_points, flatten_amount=self.flatten_amount)

    def with_spread(self, spread: float) -> 'CurveConfig':
        """
        Create a new config with adjusted spread.
        spread: -1.0 (more peaked/concentrated) to 1.0 (more spread out/uniform)
        """
        # This modifies flatten_amount
        new_flatten = self.flatten_amount + spread * 0.8
        new_flatten = max(-1.0, min(1.0, new_flatten))
        return CurveConfig(points=self.points.copy(), flatten_amount=new_flatten)

    def to_dict(self) -> dict:
        return {
            "points": [(p.x, p.y) for p in self.points],
            "flatten_amount": self.flatten_amount
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'CurveConfig':
        points = [ControlPoint(x, y) for x, y in data.get("points", [])]
        return cls(
            points=points,
            flatten_amount=data.get("flatten_amount", 0.0)
        )


# Preset curve configurations organized by category
CURVE_PRESETS = {
    # === BASIC SHAPES ===
    "uniform": {
        "name": "Uniform",
        "category": "Basic",
        "icon": "▬",
        "description": "All values equally likely",
        "points": [(0.0, 0.5), (0.5, 0.5), (1.0, 0.5)],
    },
    "bell": {
        "name": "Bell Curve",
        "category": "Basic",
        "icon": "🔔",
        "description": "Classic normal distribution",
        "points": [(0.0, 0.05), (0.2, 0.25), (0.5, 1.0), (0.8, 0.25), (1.0, 0.05)],
    },
    "soft_bell": {
        "name": "Soft Bell",
        "category": "Basic",
        "icon": "◠",
        "description": "Gentle curve, less extreme",
        "points": [(0.0, 0.3), (0.25, 0.6), (0.5, 0.9), (0.75, 0.6), (1.0, 0.3)],
    },

    # === FAVOR LOW ===
    "favor_low": {
        "name": "Favor Low",
        "category": "Skewed",
        "icon": "◢",
        "description": "Lower values more common",
        "points": [(0.0, 1.0), (0.3, 0.5), (0.6, 0.2), (1.0, 0.05)],
    },
    "strongly_low": {
        "name": "Strongly Low",
        "category": "Skewed",
        "icon": "⊿",
        "description": "Heavily favors minimum",
        "points": [(0.0, 1.0), (0.15, 0.4), (0.4, 0.1), (1.0, 0.02)],
    },
    "low_plateau": {
        "name": "Low Plateau",
        "category": "Skewed",
        "icon": "▖",
        "description": "Flat low section, drops off",
        "points": [(0.0, 0.9), (0.3, 0.9), (0.5, 0.4), (0.75, 0.1), (1.0, 0.05)],
    },

    # === FAVOR HIGH ===
    "favor_high": {
        "name": "Favor High",
        "category": "Skewed",
        "icon": "◣",
        "description": "Higher values more common",
        "points": [(0.0, 0.05), (0.4, 0.2), (0.7, 0.5), (1.0, 1.0)],
    },
    "strongly_high": {
        "name": "Strongly High",
        "category": "Skewed",
        "icon": "⊾",
        "description": "Heavily favors maximum",
        "points": [(0.0, 0.02), (0.6, 0.1), (0.85, 0.4), (1.0, 1.0)],
    },
    "high_plateau": {
        "name": "High Plateau",
        "category": "Skewed",
        "icon": "▗",
        "description": "Builds up to flat high section",
        "points": [(0.0, 0.05), (0.25, 0.1), (0.5, 0.4), (0.7, 0.9), (1.0, 0.9)],
    },

    # === CENTERED PEAKS ===
    "spike": {
        "name": "Sharp Spike",
        "category": "Peaked",
        "icon": "△",
        "description": "Tight cluster in center",
        "points": [(0.0, 0.02), (0.35, 0.05), (0.5, 1.0), (0.65, 0.05), (1.0, 0.02)],
    },
    "needle": {
        "name": "Needle",
        "category": "Peaked",
        "icon": "▲",
        "description": "Extremely tight center",
        "points": [(0.0, 0.01), (0.4, 0.02), (0.48, 0.2), (0.5, 1.0), (0.52, 0.2), (0.6, 0.02), (1.0, 0.01)],
    },

    # === OFF-CENTER PEAKS ===
    "early_peak": {
        "name": "Early Peak",
        "category": "Peaked",
        "icon": "◤",
        "description": "Peak in lower third",
        "points": [(0.0, 0.3), (0.25, 1.0), (0.5, 0.3), (0.75, 0.1), (1.0, 0.05)],
    },
    "late_peak": {
        "name": "Late Peak",
        "category": "Peaked",
        "icon": "◥",
        "description": "Peak in upper third",
        "points": [(0.0, 0.05), (0.25, 0.1), (0.5, 0.3), (0.75, 1.0), (1.0, 0.3)],
    },

    # === MULTI-MODAL ===
    "bimodal": {
        "name": "Bimodal",
        "category": "Multi-peak",
        "icon": "∩∩",
        "description": "Two peaks, rare middle",
        "points": [(0.0, 0.6), (0.2, 1.0), (0.5, 0.1), (0.8, 1.0), (1.0, 0.6)],
    },
    "bimodal_wide": {
        "name": "Wide Bimodal",
        "category": "Multi-peak",
        "icon": "◠◠",
        "description": "Two soft peaks",
        "points": [(0.0, 0.7), (0.25, 0.9), (0.5, 0.3), (0.75, 0.9), (1.0, 0.7)],
    },
    "trimodal": {
        "name": "Trimodal",
        "category": "Multi-peak",
        "icon": "∩∩∩",
        "description": "Three distinct peaks",
        "points": [(0.0, 0.8), (0.15, 1.0), (0.3, 0.2), (0.5, 0.9), (0.7, 0.2), (0.85, 1.0), (1.0, 0.8)],
    },

    # === EXTREMES ===
    "bathtub": {
        "name": "Bathtub",
        "category": "Extremes",
        "icon": "∪",
        "description": "Favors both ends",
        "points": [(0.0, 1.0), (0.25, 0.2), (0.5, 0.1), (0.75, 0.2), (1.0, 1.0)],
    },
    "soft_bathtub": {
        "name": "Soft Bathtub",
        "category": "Extremes",
        "icon": "◡",
        "description": "Gentle U-shape",
        "points": [(0.0, 0.8), (0.3, 0.4), (0.5, 0.3), (0.7, 0.4), (1.0, 0.8)],
    },
    "avoid_middle": {
        "name": "Avoid Middle",
        "category": "Extremes",
        "icon": "⋁",
        "description": "Sharp dip in center",
        "points": [(0.0, 0.8), (0.35, 0.6), (0.5, 0.05), (0.65, 0.6), (1.0, 0.8)],
    },

    # === PLATEAUS ===
    "center_plateau": {
        "name": "Center Plateau",
        "category": "Plateau",
        "icon": "▬",
        "description": "Flat elevated middle",
        "points": [(0.0, 0.1), (0.2, 0.8), (0.5, 0.85), (0.8, 0.8), (1.0, 0.1)],
    },
    "wide_plateau": {
        "name": "Wide Plateau",
        "category": "Plateau",
        "icon": "━",
        "description": "Most of range equally likely",
        "points": [(0.0, 0.1), (0.1, 0.7), (0.5, 0.75), (0.9, 0.7), (1.0, 0.1)],
    },
    "stepped": {
        "name": "Stepped",
        "category": "Plateau",
        "icon": "▄▀",
        "description": "Low then high",
        "points": [(0.0, 0.3), (0.4, 0.3), (0.5, 0.6), (0.6, 0.9), (1.0, 0.9)],
    },

    # === RAMPS ===
    "linear_up": {
        "name": "Linear Up",
        "category": "Ramp",
        "icon": "╱",
        "description": "Steady increase",
        "points": [(0.0, 0.1), (0.5, 0.5), (1.0, 0.9)],
    },
    "linear_down": {
        "name": "Linear Down",
        "category": "Ramp",
        "icon": "╲",
        "description": "Steady decrease",
        "points": [(0.0, 0.9), (0.5, 0.5), (1.0, 0.1)],
    },
    "ease_in": {
        "name": "Ease In",
        "category": "Ramp",
        "icon": "⌒╱",
        "description": "Slow start, fast finish",
        "points": [(0.0, 0.05), (0.3, 0.1), (0.6, 0.3), (0.8, 0.6), (1.0, 1.0)],
    },
    "ease_out": {
        "name": "Ease Out",
        "category": "Ramp",
        "icon": "╱⌒",
        "description": "Fast start, slow finish",
        "points": [(0.0, 1.0), (0.2, 0.6), (0.4, 0.3), (0.7, 0.1), (1.0, 0.05)],
    },

    # === SPECIAL ===
    "chaotic": {
        "name": "Chaotic",
        "category": "Special",
        "icon": "〰",
        "description": "Unpredictable pattern",
        "points": [(0.0, 0.4), (0.15, 0.9), (0.3, 0.2), (0.45, 0.7), (0.6, 0.3), (0.75, 0.8), (0.9, 0.4), (1.0, 0.6)],
    },
    "rare_jackpot": {
        "name": "Rare Jackpot",
        "category": "Special",
        "icon": "🎰",
        "description": "Usually low, rare high",
        "points": [(0.0, 0.8), (0.3, 0.6), (0.6, 0.3), (0.85, 0.1), (0.95, 0.4), (1.0, 0.9)],
    },
    "common_with_outliers": {
        "name": "With Outliers",
        "category": "Special",
        "icon": "●··●",
        "description": "Clustered center, rare extremes",
        "points": [(0.0, 0.3), (0.1, 0.1), (0.3, 0.6), (0.5, 1.0), (0.7, 0.6), (0.9, 0.1), (1.0, 0.3)],
    },
}


def get_preset_config(preset_name: str) -> CurveConfig:
    """Get a CurveConfig from a preset name."""
    if preset_name not in CURVE_PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}")

    preset = CURVE_PRESETS[preset_name]
    points = [ControlPoint(x, y) for x, y in preset["points"]]
    return CurveConfig(points=points)


def get_presets_by_category() -> dict:
    """Get presets organized by category."""
    categories = {}
    for key, preset in CURVE_PRESETS.items():
        cat = preset.get("category", "Other")
        if cat not in categories:
            categories[cat] = []
        categories[cat].append((key, preset))
    return categories


def get_category_order() -> list:
    """Get the display order for categories."""
    return ["Basic", "Skewed", "Peaked", "Multi-peak", "Extremes", "Plateau", "Ramp", "Special"]


class CustomCurveGenerator(BaseGenerator):
    """Generator using a custom probability curve."""

    def __init__(
        self,
        min_val: float = 0,
        max_val: float = 100,
        curve_config: Optional[CurveConfig] = None,
        label_config: Optional[LabelConfig] = None,
    ):
        super().__init__(min_val, max_val, label_config)
        self.curve_config = curve_config or CurveConfig()
        self._build_sampler()

    def _build_sampler(self) -> None:
        """Build the inverse CDF for sampling."""
        # Get control points
        x_ctrl, y_ctrl = self.curve_config.get_points_array()

        # Apply flatten/sharpen
        y_ctrl = self.curve_config.apply_flatten(y_ctrl)

        # Ensure we have enough points
        if len(x_ctrl) < 2:
            x_ctrl = np.array([0.0, 1.0])
            y_ctrl = np.array([0.5, 0.5])

        # Create smooth interpolation using PCHIP (monotonic, no overshoots)
        try:
            interp = PchipInterpolator(x_ctrl, y_ctrl)
        except ValueError:
            # Fallback to linear if PCHIP fails
            interp = lambda x: np.interp(x, x_ctrl, y_ctrl)

        # Sample the curve at many points
        self._x_fine = np.linspace(0, 1, 500)
        self._pdf = np.maximum(interp(self._x_fine), 0.001)  # Ensure positive

        # Normalize
        self._pdf = self._pdf / np.trapz(self._pdf, self._x_fine)

        # Build CDF for inverse transform sampling
        self._cdf = np.zeros_like(self._x_fine)
        self._cdf[1:] = cumulative_trapezoid(self._pdf, self._x_fine)
        self._cdf = self._cdf / self._cdf[-1]  # Normalize to [0, 1]

    def _generate_raw(self, count: int) -> np.ndarray:
        """Generate using inverse transform sampling."""
        # Generate uniform random values
        u = np.random.random(count)

        # Inverse CDF lookup
        normalized = np.interp(u, self._cdf, self._x_fine)

        # Scale to actual range
        return self.min_val + normalized * self.range_width

    def get_name(self) -> str:
        return "Custom Curve"

    def get_description(self) -> str:
        return "User-defined probability curve"

    def get_probability_at(self, value: float) -> float:
        """Get probability at a specific value."""
        if value < self.min_val or value > self.max_val:
            return 0.0
        normalized = (value - self.min_val) / self.range_width
        return np.interp(normalized, self._x_fine, self._pdf)

    def get_curve_for_display(self, num_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Get the curve for visualization (in actual value space)."""
        x = np.linspace(self.min_val, self.max_val, num_points)
        y = np.array([self.get_probability_at(v) for v in x])
        return x, y
