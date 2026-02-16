"""Base generator class for random number generation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Union, Any
import numpy as np

from models.labels import LabelConfig


@dataclass
class GeneratorResult:
    """Result of a generation operation."""

    values: List[float]  # Raw numeric values
    labels: Optional[List[str]] = None  # Labels if label_config provided

    def __len__(self) -> int:
        return len(self.values)

    def __iter__(self):
        if self.labels:
            return iter(zip(self.values, self.labels))
        return iter(self.values)

    def get_display_values(self) -> List[str]:
        """Get values formatted for display."""
        if self.labels:
            return self.labels
        return [self._format_value(v) for v in self.values]

    def _format_value(self, value: float) -> str:
        """Format a single value for display."""
        if value == int(value):
            return str(int(value))
        return f"{value:.2f}"


class BaseGenerator(ABC):
    """Abstract base class for random number generators."""

    def __init__(
        self,
        min_val: float = 0,
        max_val: float = 100,
        label_config: Optional[LabelConfig] = None
    ):
        """
        Initialize the generator.

        Args:
            min_val: Minimum value (inclusive)
            max_val: Maximum value (inclusive)
            label_config: Optional label configuration for output mapping
        """
        if min_val >= max_val:
            raise ValueError(f"min_val ({min_val}) must be less than max_val ({max_val})")

        self.min_val = min_val
        self.max_val = max_val
        self.label_config = label_config

    @property
    def range_width(self) -> float:
        """Return the width of the value range."""
        return self.max_val - self.min_val

    @abstractmethod
    def _generate_raw(self, count: int) -> np.ndarray:
        """
        Generate raw random values.

        Args:
            count: Number of values to generate

        Returns:
            NumPy array of random values
        """
        pass

    def generate(self, count: int = 1) -> GeneratorResult:
        """
        Generate random values.

        Args:
            count: Number of values to generate

        Returns:
            GeneratorResult with values and optional labels
        """
        if count < 1:
            raise ValueError("count must be at least 1")

        # Generate raw values
        raw_values = self._generate_raw(count)

        # Clamp to range
        values = np.clip(raw_values, self.min_val, self.max_val).tolist()

        # Apply labels if configured
        labels = None
        if self.label_config and self.label_config.enabled:
            labels = [self.label_config.get_label(v) for v in values]

        return GeneratorResult(values=values, labels=labels)

    def generate_one(self) -> Union[float, str]:
        """Generate a single value. Returns label if configured, else value."""
        result = self.generate(1)
        if result.labels:
            return result.labels[0]
        return result.values[0]

    @abstractmethod
    def get_name(self) -> str:
        """Return the display name of this generator."""
        pass

    @abstractmethod
    def get_description(self) -> str:
        """Return a description of this generator."""
        pass

    def get_params(self) -> dict:
        """Return generator parameters for serialization."""
        return {
            "min_val": self.min_val,
            "max_val": self.max_val
        }

    def get_probability_at(self, value: float) -> float:
        """
        Get the probability density at a specific value.
        Used for visualization.

        Args:
            value: The value to get probability for

        Returns:
            Probability density (not necessarily normalized)
        """
        return 1.0  # Default uniform

    def get_probability_distribution(self, num_points: int = 100) -> tuple:
        """
        Get arrays for plotting the probability distribution.

        Args:
            num_points: Number of points to sample

        Returns:
            Tuple of (x_values, y_values) for plotting
        """
        x = np.linspace(self.min_val, self.max_val, num_points)
        y = np.array([self.get_probability_at(v) for v in x])

        # Normalize so area under curve is 1
        total = np.trapz(y, x)
        if total > 0:
            y = y / total

        return x, y
