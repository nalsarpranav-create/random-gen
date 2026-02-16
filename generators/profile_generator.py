"""Generator for profile-based batch attribute generation."""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import numpy as np

from models.profile import Profile, AttributeConfig, DistributionType, DependencyType
from models.labels import LabelConfig
from generators.presets import (
    UniformGenerator,
    NormalGenerator,
    DiceRollGenerator,
    TriangularGenerator,
    ExponentialGenerator,
    WeightedTableGenerator,
)
from generators.custom_curve import CustomCurveGenerator, CurveConfig


@dataclass
class AttributeResult:
    """Result for a single attribute."""
    name: str
    value: float
    label: Optional[str] = None

    def display(self) -> str:
        """Get display string."""
        if self.label:
            return self.label
        if self.value == int(self.value):
            return str(int(self.value))
        return f"{self.value:.1f}"


@dataclass
class ProfileResult:
    """Result of generating a full profile."""
    profile_name: str
    attributes: List[AttributeResult]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile_name": self.profile_name,
            "attributes": {a.name: {"value": a.value, "label": a.label} for a in self.attributes}
        }


def create_generator_for_attribute(
    attr: AttributeConfig,
    generated_values: Dict[str, float]
) -> Any:
    """Create the appropriate generator for an attribute."""

    min_val = attr.min_val
    max_val = attr.max_val

    # Apply dependencies if any
    if attr.depends_on and attr.depends_on in generated_values:
        dep_value = generated_values[attr.depends_on]

        if attr.dependency_type == DependencyType.SCALED:
            # Scale the output range by the dependency
            scale_factor = attr.dependency_params.get("factor", 0.1)
            offset = dep_value * scale_factor
            min_val += offset
            max_val += offset

        elif attr.dependency_type == DependencyType.BOUNDED:
            # Dependency sets bounds
            bound_type = attr.dependency_params.get("bound", "max")
            if bound_type == "max":
                max_val = min(max_val, dep_value)
            elif bound_type == "min":
                min_val = max(min_val, dep_value)

    # Ensure valid range
    if min_val >= max_val:
        min_val = max_val - 1

    # Create generator based on distribution type
    if attr.distribution == DistributionType.UNIFORM:
        return UniformGenerator(
            min_val=min_val,
            max_val=max_val,
            label_config=attr.label_config
        )

    elif attr.distribution == DistributionType.NORMAL:
        mean = attr.params.get("mean", (min_val + max_val) / 2)
        std = attr.params.get("std", (max_val - min_val) / 6)
        return NormalGenerator(
            min_val=min_val,
            max_val=max_val,
            mean=mean,
            std=std,
            label_config=attr.label_config
        )

    elif attr.distribution == DistributionType.DICE:
        notation = attr.params.get("notation", "1d6")
        return DiceRollGenerator(
            notation=notation,
            label_config=attr.label_config
        )

    elif attr.distribution == DistributionType.TRIANGULAR:
        mode = attr.params.get("mode", (min_val + max_val) / 2)
        return TriangularGenerator(
            min_val=min_val,
            max_val=max_val,
            mode=mode,
            label_config=attr.label_config
        )

    elif attr.distribution == DistributionType.EXPONENTIAL:
        scale = attr.params.get("scale", (max_val - min_val) / 3)
        return ExponentialGenerator(
            min_val=min_val,
            max_val=max_val,
            scale=scale,
            label_config=attr.label_config
        )

    elif attr.distribution == DistributionType.WEIGHTED:
        entries = attr.params.get("entries", [])
        if not entries:
            # Fallback to uniform if no entries
            return UniformGenerator(
                min_val=min_val,
                max_val=max_val,
                label_config=attr.label_config
            )
        return WeightedTableGenerator(
            entries=entries,
            label_config=attr.label_config
        )

    elif attr.distribution == DistributionType.CUSTOM:
        curve_data = attr.params.get("curve", {})
        curve_config = CurveConfig.from_dict(curve_data) if curve_data else CurveConfig()
        return CustomCurveGenerator(
            min_val=min_val,
            max_val=max_val,
            curve_config=curve_config,
            label_config=attr.label_config
        )

    else:
        # Fallback to uniform
        return UniformGenerator(
            min_val=min_val,
            max_val=max_val,
            label_config=attr.label_config
        )


def generate_profile(profile: Profile) -> ProfileResult:
    """Generate all attributes in a profile."""

    # Validate first
    validation = profile.validate()
    if validation["errors"]:
        raise ValueError(f"Invalid profile: {validation['errors']}")

    # Get generation order (respects dependencies)
    order = profile.get_generation_order()

    # Track generated values for dependencies
    generated_values: Dict[str, float] = {}
    results: List[AttributeResult] = []

    for idx in order:
        attr = profile.attributes[idx]

        # Create generator with dependency context
        generator = create_generator_for_attribute(attr, generated_values)

        # Generate value
        result = generator.generate(1)
        value = result.values[0]
        label = result.labels[0] if result.labels else None

        # Store for dependencies
        generated_values[attr.name] = value

        results.append(AttributeResult(
            name=attr.name,
            value=value,
            label=label
        ))

    # Sort results back to original order
    results_by_name = {r.name: r for r in results}
    ordered_results = [results_by_name[attr.name] for attr in profile.attributes]

    return ProfileResult(
        profile_name=profile.name,
        attributes=ordered_results
    )


def generate_profile_batch(profile: Profile, count: int) -> List[ProfileResult]:
    """Generate multiple instances of a profile."""
    return [generate_profile(profile) for _ in range(count)]
