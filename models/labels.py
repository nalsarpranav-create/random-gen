"""Label configuration for mapping number ranges to word labels."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import re
import math


@dataclass
class LabelRange:
    """A single label range mapping a numeric range to a word label."""

    label: str
    min_val: float
    max_val: float
    color: str = "#888888"

    def validate(self) -> List[str]:
        """Validate this range and return list of error messages (empty if valid)."""
        errors = []

        # Check label is not empty
        if not self.label or not self.label.strip():
            errors.append("Label cannot be empty")

        # Check label length
        if len(self.label) > 50:
            errors.append("Label must be 50 characters or less")

        # Check min < max
        if self.min_val >= self.max_val:
            errors.append(f"Min ({self.min_val}) must be less than Max ({self.max_val})")

        # Check for NaN/Inf
        if math.isnan(self.min_val) or math.isinf(self.min_val):
            errors.append("Min value cannot be NaN or Infinity")
        if math.isnan(self.max_val) or math.isinf(self.max_val):
            errors.append("Max value cannot be NaN or Infinity")

        # Validate color format
        if not re.match(r'^#[0-9A-Fa-f]{6}$', self.color):
            errors.append("Color must be valid hex format (e.g., #FF0000)")

        return errors

    def contains(self, value: float) -> bool:
        """Check if value falls within this range (inclusive on both ends)."""
        if math.isnan(value) or math.isinf(value):
            return False
        return self.min_val <= value <= self.max_val

    def width(self) -> float:
        """Return the width of this range."""
        return self.max_val - self.min_val

    def overlaps(self, other: 'LabelRange') -> bool:
        """Check if this range overlaps with another range."""
        return self.min_val <= other.max_val and self.max_val >= other.min_val

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "label": self.label,
            "min_val": self.min_val,
            "max_val": self.max_val,
            "color": self.color
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LabelRange':
        """Create from dictionary (JSON deserialization)."""
        return cls(
            label=data.get("label", ""),
            min_val=float(data.get("min_val", 0)),
            max_val=float(data.get("max_val", 100)),
            color=data.get("color", "#888888")
        )


@dataclass
class LabelConfig:
    """Configuration for label ranges with validation and gap detection."""

    ranges: List[LabelRange] = field(default_factory=list)
    enabled: bool = False

    # Undo/redo history
    _history: List[List[LabelRange]] = field(default_factory=list, repr=False)
    _history_index: int = field(default=-1, repr=False)
    _max_history: int = field(default=20, repr=False)

    def validate(self) -> Dict[str, List[str]]:
        """
        Validate the entire configuration.

        Returns:
            Dict with 'errors' (blocking) and 'warnings' (informational)
        """
        errors = []
        warnings = []

        if not self.ranges:
            if self.enabled:
                warnings.append("Labels enabled but no ranges defined")
            return {"errors": errors, "warnings": warnings}

        # Validate individual ranges
        for i, r in enumerate(self.ranges):
            for err in r.validate():
                errors.append(f"Range '{r.label}' (#{i+1}): {err}")

        # Sort ranges by min_val for overlap/gap detection
        sorted_ranges = sorted(self.ranges, key=lambda r: r.min_val)

        # Check for overlaps
        for i in range(len(sorted_ranges) - 1):
            curr = sorted_ranges[i]
            next_r = sorted_ranges[i + 1]
            if curr.overlaps(next_r):
                errors.append(
                    f"Overlap: '{curr.label}' ({curr.min_val}-{curr.max_val}) "
                    f"and '{next_r.label}' ({next_r.min_val}-{next_r.max_val})"
                )

        # Check for gaps
        for i in range(len(sorted_ranges) - 1):
            curr = sorted_ranges[i]
            next_r = sorted_ranges[i + 1]
            # Gap exists if there's space between ranges (allowing for float precision)
            gap_size = next_r.min_val - curr.max_val
            if gap_size > 0.001:  # Small tolerance for float precision
                gap_start = curr.max_val
                gap_end = next_r.min_val
                warnings.append(f"Gap: {gap_start:.2f} to {gap_end:.2f} has no label")

        # Check for duplicate labels
        labels = [r.label for r in self.ranges]
        seen = set()
        duplicates = set()
        for label in labels:
            if label in seen:
                duplicates.add(label)
            seen.add(label)
        if duplicates:
            warnings.append(f"Duplicate label names: {', '.join(duplicates)}")

        # Check for single-point ranges
        for r in self.ranges:
            if abs(r.max_val - r.min_val) < 0.001:
                warnings.append(f"'{r.label}' is a single-value range")

        return {"errors": errors, "warnings": warnings}

    def is_valid(self) -> bool:
        """Return True if there are no errors (warnings are OK)."""
        return len(self.validate()["errors"]) == 0

    def get_label(self, value: float) -> str:
        """
        Get the label for a value.

        Args:
            value: The numeric value to label

        Returns:
            The matching label, or formatted value with "(unlabeled)" if no match
        """
        if not self.enabled or not self.ranges:
            return str(value)

        # Handle NaN/Inf
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"

        # Find matching range
        for r in self.ranges:
            if r.contains(value):
                return r.label

        # No match - return value with indicator
        if isinstance(value, float) and value == int(value):
            return f"{int(value)} (unlabeled)"
        return f"{value:.2f} (unlabeled)"

    def get_sorted_ranges(self) -> List[LabelRange]:
        """Return ranges sorted by min_val."""
        return sorted(self.ranges, key=lambda r: r.min_val)

    def find_gaps(self, global_min: float, global_max: float) -> List[Tuple[float, float]]:
        """
        Find gaps in coverage between global_min and global_max.

        Returns:
            List of (start, end) tuples representing gaps
        """
        gaps = []
        if not self.ranges:
            return [(global_min, global_max)]

        sorted_ranges = self.get_sorted_ranges()

        # Gap before first range
        if sorted_ranges[0].min_val > global_min:
            gaps.append((global_min, sorted_ranges[0].min_val))

        # Gaps between ranges
        for i in range(len(sorted_ranges) - 1):
            curr_max = sorted_ranges[i].max_val
            next_min = sorted_ranges[i + 1].min_val
            if next_min > curr_max:
                gaps.append((curr_max, next_min))

        # Gap after last range
        if sorted_ranges[-1].max_val < global_max:
            gaps.append((sorted_ranges[-1].max_val, global_max))

        return gaps

    def auto_fill_gaps(self, global_min: float, global_max: float) -> int:
        """
        Fill gaps with auto-generated labels.

        Args:
            global_min: The minimum of the overall range
            global_max: The maximum of the overall range

        Returns:
            Number of gaps filled
        """
        self._save_to_history()

        if not self.ranges:
            self.ranges.append(LabelRange(
                label="Default",
                min_val=global_min,
                max_val=global_max,
                color="#888888"
            ))
            return 1

        gaps = self.find_gaps(global_min, global_max)
        filled = 0

        for i, (gap_start, gap_end) in enumerate(gaps):
            self.ranges.append(LabelRange(
                label=f"Unlabeled_{filled + 1}",
                min_val=gap_start,
                max_val=gap_end,
                color="#CCCCCC"
            ))
            filled += 1

        return filled

    def add_range(self, label: str, min_val: float, max_val: float,
                  color: str = "#888888") -> Optional[str]:
        """
        Add a new range.

        Returns:
            Error message if invalid, None if successful
        """
        new_range = LabelRange(label=label, min_val=min_val, max_val=max_val, color=color)
        errors = new_range.validate()
        if errors:
            return "; ".join(errors)

        # Check for overlaps with existing ranges
        for existing in self.ranges:
            if new_range.overlaps(existing):
                return f"Overlaps with existing range '{existing.label}'"

        self._save_to_history()
        self.ranges.append(new_range)
        return None

    def remove_range(self, index: int) -> bool:
        """Remove a range by index. Returns True if successful."""
        if 0 <= index < len(self.ranges):
            self._save_to_history()
            self.ranges.pop(index)
            return True
        return False

    def update_range(self, index: int, **kwargs) -> Optional[str]:
        """
        Update a range at index with new values.

        Returns:
            Error message if invalid, None if successful
        """
        if not (0 <= index < len(self.ranges)):
            return "Invalid range index"

        # Create updated range
        old_range = self.ranges[index]
        new_range = LabelRange(
            label=kwargs.get("label", old_range.label),
            min_val=kwargs.get("min_val", old_range.min_val),
            max_val=kwargs.get("max_val", old_range.max_val),
            color=kwargs.get("color", old_range.color)
        )

        # Validate
        errors = new_range.validate()
        if errors:
            return "; ".join(errors)

        # Check for overlaps (excluding self)
        for i, existing in enumerate(self.ranges):
            if i != index and new_range.overlaps(existing):
                return f"Overlaps with existing range '{existing.label}'"

        self._save_to_history()
        self.ranges[index] = new_range
        return None

    def _save_to_history(self) -> None:
        """Save current state to undo history."""
        # Truncate any redo history
        if self._history_index < len(self._history) - 1:
            self._history = self._history[:self._history_index + 1]

        # Save current state
        state = [LabelRange(
            label=r.label,
            min_val=r.min_val,
            max_val=r.max_val,
            color=r.color
        ) for r in self.ranges]
        self._history.append(state)

        # Limit history size
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        self._history_index = len(self._history) - 1

    def undo(self) -> bool:
        """Undo last change. Returns True if successful."""
        if self._history_index > 0:
            self._history_index -= 1
            self.ranges = [LabelRange(
                label=r.label,
                min_val=r.min_val,
                max_val=r.max_val,
                color=r.color
            ) for r in self._history[self._history_index]]
            return True
        return False

    def redo(self) -> bool:
        """Redo last undone change. Returns True if successful."""
        if self._history_index < len(self._history) - 1:
            self._history_index += 1
            self.ranges = [LabelRange(
                label=r.label,
                min_val=r.min_val,
                max_val=r.max_val,
                color=r.color
            ) for r in self._history[self._history_index]]
            return True
        return False

    def can_undo(self) -> bool:
        """Check if undo is available."""
        return self._history_index > 0

    def can_redo(self) -> bool:
        """Check if redo is available."""
        return self._history_index < len(self._history) - 1

    def reset(self) -> None:
        """Clear all ranges."""
        self._save_to_history()
        self.ranges = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "ranges": [r.to_dict() for r in self.ranges],
            "enabled": self.enabled
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LabelConfig':
        """Create from dictionary (JSON deserialization)."""
        if not data:
            return cls()

        ranges = []
        for r_data in data.get("ranges", []):
            try:
                ranges.append(LabelRange.from_dict(r_data))
            except (ValueError, TypeError):
                # Skip invalid ranges
                continue

        return cls(
            ranges=ranges,
            enabled=data.get("enabled", False)
        )


# Default color palette for new labels
DEFAULT_COLORS = [
    "#4CAF50",  # Green
    "#2196F3",  # Blue
    "#9C27B0",  # Purple
    "#FF9800",  # Orange
    "#F44336",  # Red
    "#00BCD4",  # Cyan
    "#FFEB3B",  # Yellow
    "#795548",  # Brown
    "#607D8B",  # Blue Grey
    "#E91E63",  # Pink
]


def get_default_color(index: int) -> str:
    """Get a default color for a label at the given index."""
    return DEFAULT_COLORS[index % len(DEFAULT_COLORS)]
