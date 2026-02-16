"""Preset distribution generators."""

from typing import Optional
import numpy as np
from scipy import stats
import re

from generators.base import BaseGenerator
from models.labels import LabelConfig


class UniformGenerator(BaseGenerator):
    """Uniform distribution - all values equally likely."""

    def _generate_raw(self, count: int) -> np.ndarray:
        return np.random.uniform(self.min_val, self.max_val, count)

    def get_name(self) -> str:
        return "Uniform"

    def get_description(self) -> str:
        return "All values between min and max are equally likely."

    def get_probability_at(self, value: float) -> float:
        if self.min_val <= value <= self.max_val:
            return 1.0 / self.range_width
        return 0.0


class NormalGenerator(BaseGenerator):
    """Normal (Gaussian) distribution."""

    def __init__(
        self,
        min_val: float = 0,
        max_val: float = 100,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        label_config: Optional[LabelConfig] = None
    ):
        super().__init__(min_val, max_val, label_config)
        # Default mean to center, std to 1/6 of range (99.7% within bounds)
        self.mean = mean if mean is not None else (min_val + max_val) / 2
        self.std = std if std is not None else self.range_width / 6

    def _generate_raw(self, count: int) -> np.ndarray:
        return np.random.normal(self.mean, self.std, count)

    def get_name(self) -> str:
        return "Normal"

    def get_description(self) -> str:
        return "Bell curve distribution centered around the mean."

    def get_params(self) -> dict:
        params = super().get_params()
        params.update({"mean": self.mean, "std": self.std})
        return params

    def get_probability_at(self, value: float) -> float:
        return stats.norm.pdf(value, self.mean, self.std)


class DiceRollGenerator(BaseGenerator):
    """Dice roll notation (e.g., 2d6, 3d8+5)."""

    def __init__(
        self,
        notation: str = "1d6",
        label_config: Optional[LabelConfig] = None
    ):
        self.notation = notation
        self._parse_notation(notation)

        # Calculate actual min/max from dice
        actual_min = self.num_dice + self.modifier
        actual_max = self.num_dice * self.die_faces + self.modifier

        super().__init__(actual_min, actual_max, label_config)

    def _parse_notation(self, notation: str) -> None:
        """Parse dice notation like 2d6+3 or 3d8-2."""
        pattern = r'^(\d+)d(\d+)([+-]\d+)?$'
        match = re.match(pattern, notation.lower().replace(' ', ''))

        if not match:
            raise ValueError(
                f"Invalid dice notation: '{notation}'. "
                "Use format like '2d6', '3d8+5', or '1d20-2'"
            )

        self.num_dice = int(match.group(1))
        self.die_faces = int(match.group(2))
        self.modifier = int(match.group(3)) if match.group(3) else 0

        if self.num_dice < 1:
            raise ValueError("Number of dice must be at least 1")
        if self.die_faces < 2:
            raise ValueError("Die must have at least 2 faces")

    def _generate_raw(self, count: int) -> np.ndarray:
        # Roll dice: sum of num_dice rolls of die_faces-sided die
        rolls = np.random.randint(1, self.die_faces + 1, (count, self.num_dice))
        return rolls.sum(axis=1) + self.modifier

    def get_name(self) -> str:
        return f"Dice ({self.notation})"

    def get_description(self) -> str:
        return f"Roll {self.num_dice} dice with {self.die_faces} faces each."

    def get_params(self) -> dict:
        return {"notation": self.notation}

    def get_probability_at(self, value: float) -> float:
        # Approximate with normal for visualization
        mean = self.num_dice * (self.die_faces + 1) / 2 + self.modifier
        std = np.sqrt(self.num_dice * (self.die_faces**2 - 1) / 12)
        return stats.norm.pdf(value, mean, std)


class TriangularGenerator(BaseGenerator):
    """Triangular distribution with a mode (peak)."""

    def __init__(
        self,
        min_val: float = 0,
        max_val: float = 100,
        mode: Optional[float] = None,
        label_config: Optional[LabelConfig] = None
    ):
        super().__init__(min_val, max_val, label_config)
        # Default mode to center
        self.mode = mode if mode is not None else (min_val + max_val) / 2

        if not (min_val <= self.mode <= max_val):
            raise ValueError(
                f"Mode ({self.mode}) must be between min ({min_val}) and max ({max_val})"
            )

    def _generate_raw(self, count: int) -> np.ndarray:
        return np.random.triangular(self.min_val, self.mode, self.max_val, count)

    def get_name(self) -> str:
        return "Triangular"

    def get_description(self) -> str:
        return "Triangle-shaped distribution peaking at the mode."

    def get_params(self) -> dict:
        params = super().get_params()
        params["mode"] = self.mode
        return params

    def get_probability_at(self, value: float) -> float:
        if value < self.min_val or value > self.max_val:
            return 0.0
        return stats.triang.pdf(
            value,
            (self.mode - self.min_val) / self.range_width,
            loc=self.min_val,
            scale=self.range_width
        )


class ExponentialGenerator(BaseGenerator):
    """Exponential distribution - rare highs, common lows."""

    def __init__(
        self,
        min_val: float = 0,
        max_val: float = 100,
        scale: Optional[float] = None,
        label_config: Optional[LabelConfig] = None
    ):
        super().__init__(min_val, max_val, label_config)
        # Default scale so that ~95% of values are in range
        self.scale = scale if scale is not None else self.range_width / 3

    def _generate_raw(self, count: int) -> np.ndarray:
        # Generate exponential and shift to start at min_val
        return np.random.exponential(self.scale, count) + self.min_val

    def get_name(self) -> str:
        return "Exponential"

    def get_description(self) -> str:
        return "Common low values, rare high values."

    def get_params(self) -> dict:
        params = super().get_params()
        params["scale"] = self.scale
        return params

    def get_probability_at(self, value: float) -> float:
        if value < self.min_val:
            return 0.0
        return stats.expon.pdf(value - self.min_val, scale=self.scale)


class WeightedTableGenerator(BaseGenerator):
    """Weighted table for loot-table style generation."""

    def __init__(
        self,
        entries: list[dict],
        label_config: Optional[LabelConfig] = None
    ):
        """
        Args:
            entries: List of {"value": x, "weight": w} dicts
        """
        if not entries:
            raise ValueError("Entries cannot be empty")

        self.entries = entries
        self.values = np.array([e["value"] for e in entries])
        self.weights = np.array([e["weight"] for e in entries], dtype=float)

        # Normalize weights
        total = self.weights.sum()
        if total <= 0:
            raise ValueError("Total weight must be positive")
        self.probabilities = self.weights / total

        super().__init__(
            min_val=float(self.values.min()),
            max_val=float(self.values.max()),
            label_config=label_config
        )

    def _generate_raw(self, count: int) -> np.ndarray:
        return np.random.choice(self.values, size=count, p=self.probabilities)

    def get_name(self) -> str:
        return "Weighted Table"

    def get_description(self) -> str:
        return "Custom weighted outcomes like a loot table."

    def get_params(self) -> dict:
        return {"entries": self.entries}

    def get_probability_at(self, value: float) -> float:
        idx = np.where(np.isclose(self.values, value))[0]
        if len(idx) > 0:
            return self.probabilities[idx[0]]
        return 0.0


class AverageOfNGenerator(BaseGenerator):
    """Average of N random values - demonstrates Central Limit Theorem."""

    def __init__(
        self,
        min_val: float = 0,
        max_val: float = 100,
        n: int = 3,
        label_config: Optional[LabelConfig] = None
    ):
        super().__init__(min_val, max_val, label_config)
        if n < 1:
            raise ValueError("n must be at least 1")
        self.n = n

    def _generate_raw(self, count: int) -> np.ndarray:
        # Generate n uniform values and average them
        samples = np.random.uniform(self.min_val, self.max_val, (count, self.n))
        return samples.mean(axis=1)

    def get_name(self) -> str:
        return f"Average of {self.n}"

    def get_description(self) -> str:
        return f"Average of {self.n} uniform samples. Higher N = more normal."

    def get_params(self) -> dict:
        params = super().get_params()
        params["n"] = self.n
        return params

    def get_probability_at(self, value: float) -> float:
        # Approximate with normal distribution (CLT)
        mean = (self.min_val + self.max_val) / 2
        std = self.range_width / (2 * np.sqrt(3 * self.n))
        return stats.norm.pdf(value, mean, std)


# Registry of all preset generators
PRESET_GENERATORS = {
    "uniform": UniformGenerator,
    "normal": NormalGenerator,
    "dice": DiceRollGenerator,
    "triangular": TriangularGenerator,
    "exponential": ExponentialGenerator,
    "weighted": WeightedTableGenerator,
    "average": AverageOfNGenerator,
}


def get_preset_names() -> list[str]:
    """Get list of available preset names."""
    return list(PRESET_GENERATORS.keys())


def create_generator(preset_name: str, **kwargs) -> BaseGenerator:
    """
    Create a generator from a preset name.

    Args:
        preset_name: Name of the preset
        **kwargs: Parameters for the generator

    Returns:
        Configured generator instance
    """
    if preset_name not in PRESET_GENERATORS:
        raise ValueError(
            f"Unknown preset: '{preset_name}'. "
            f"Available: {', '.join(PRESET_GENERATORS.keys())}"
        )

    return PRESET_GENERATORS[preset_name](**kwargs)
