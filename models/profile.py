"""Profile system for batch attribute generation."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import json
from pathlib import Path

from models.labels import LabelConfig, LabelRange


class DistributionType(Enum):
    UNIFORM = "uniform"
    NORMAL = "normal"
    DICE = "dice"
    TRIANGULAR = "triangular"
    EXPONENTIAL = "exponential"
    WEIGHTED = "weighted"
    CUSTOM = "custom"


class DependencyType(Enum):
    NONE = "none"
    SCALED = "scaled"  # Output is scaled by another attribute
    BOUNDED = "bounded"  # Another attribute sets min/max
    CONDITIONAL = "conditional"  # Different distribution based on condition


@dataclass
class AttributeConfig:
    """Configuration for a single attribute in a profile."""

    name: str
    distribution: DistributionType = DistributionType.UNIFORM
    min_val: float = 1
    max_val: float = 100

    # Distribution-specific params
    params: Dict[str, Any] = field(default_factory=dict)

    # Labels
    label_config: Optional[LabelConfig] = None

    # Dependencies
    depends_on: Optional[str] = None
    dependency_type: DependencyType = DependencyType.NONE
    dependency_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "distribution": self.distribution.value,
            "min_val": self.min_val,
            "max_val": self.max_val,
            "params": self.params,
            "label_config": self.label_config.to_dict() if self.label_config else None,
            "depends_on": self.depends_on,
            "dependency_type": self.dependency_type.value,
            "dependency_params": self.dependency_params,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AttributeConfig':
        label_config = None
        if data.get("label_config"):
            label_config = LabelConfig.from_dict(data["label_config"])

        return cls(
            name=data.get("name", "Unnamed"),
            distribution=DistributionType(data.get("distribution", "uniform")),
            min_val=float(data.get("min_val", 1)),
            max_val=float(data.get("max_val", 100)),
            params=data.get("params", {}),
            label_config=label_config,
            depends_on=data.get("depends_on"),
            dependency_type=DependencyType(data.get("dependency_type", "none")),
            dependency_params=data.get("dependency_params", {}),
        )


@dataclass
class Profile:
    """A collection of attributes to generate together."""

    name: str = "New Profile"
    attributes: List[AttributeConfig] = field(default_factory=list)

    def add_attribute(self, attr: AttributeConfig) -> None:
        """Add an attribute to the profile."""
        self.attributes.append(attr)

    def remove_attribute(self, index: int) -> bool:
        """Remove an attribute by index."""
        if 0 <= index < len(self.attributes):
            self.attributes.pop(index)
            return True
        return False

    def get_attribute_names(self) -> List[str]:
        """Get list of attribute names."""
        return [attr.name for attr in self.attributes]

    def get_generation_order(self) -> List[int]:
        """
        Get indices in order they should be generated (respecting dependencies).
        Uses topological sort.
        """
        # Build dependency graph
        name_to_idx = {attr.name: i for i, attr in enumerate(self.attributes)}
        in_degree = [0] * len(self.attributes)
        dependents = [[] for _ in self.attributes]

        for i, attr in enumerate(self.attributes):
            if attr.depends_on and attr.depends_on in name_to_idx:
                dep_idx = name_to_idx[attr.depends_on]
                dependents[dep_idx].append(i)
                in_degree[i] += 1

        # Kahn's algorithm
        order = []
        queue = [i for i, deg in enumerate(in_degree) if deg == 0]

        while queue:
            idx = queue.pop(0)
            order.append(idx)
            for dependent in dependents[idx]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        # Check for cycles
        if len(order) != len(self.attributes):
            # Cycle detected - just return natural order
            return list(range(len(self.attributes)))

        return order

    def validate(self) -> Dict[str, List[str]]:
        """Validate the profile configuration."""
        errors = []
        warnings = []

        if not self.attributes:
            warnings.append("Profile has no attributes")

        # Check for duplicate names
        names = self.get_attribute_names()
        seen = set()
        for name in names:
            if name in seen:
                errors.append(f"Duplicate attribute name: '{name}'")
            seen.add(name)

        # Check dependencies
        for attr in self.attributes:
            if attr.depends_on:
                if attr.depends_on not in names:
                    errors.append(f"'{attr.name}' depends on non-existent '{attr.depends_on}'")
                elif attr.depends_on == attr.name:
                    errors.append(f"'{attr.name}' cannot depend on itself")

        # Check for circular dependencies
        order = self.get_generation_order()
        if len(order) != len(self.attributes):
            errors.append("Circular dependency detected")

        return {"errors": errors, "warnings": warnings}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "attributes": [attr.to_dict() for attr in self.attributes],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Profile':
        attributes = [AttributeConfig.from_dict(a) for a in data.get("attributes", [])]
        return cls(
            name=data.get("name", "Unnamed Profile"),
            attributes=attributes,
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> 'Profile':
        return cls.from_dict(json.loads(json_str))


# Preset profiles
PRESET_PROFILES = {
    "rpg_character": Profile(
        name="RPG Character",
        attributes=[
            AttributeConfig(name="Strength", distribution=DistributionType.DICE, params={"notation": "3d6"}),
            AttributeConfig(name="Dexterity", distribution=DistributionType.DICE, params={"notation": "3d6"}),
            AttributeConfig(name="Constitution", distribution=DistributionType.DICE, params={"notation": "3d6"}),
            AttributeConfig(name="Intelligence", distribution=DistributionType.DICE, params={"notation": "3d6"}),
            AttributeConfig(name="Wisdom", distribution=DistributionType.DICE, params={"notation": "3d6"}),
            AttributeConfig(name="Charisma", distribution=DistributionType.DICE, params={"notation": "3d6"}),
        ]
    ),
    "loot_drop": Profile(
        name="Loot Drop",
        attributes=[
            AttributeConfig(
                name="Rarity",
                distribution=DistributionType.UNIFORM,
                min_val=1,
                max_val=100,
                label_config=LabelConfig(
                    enabled=True,
                    ranges=[
                        # Will be created with proper LabelRange objects
                    ]
                )
            ),
            AttributeConfig(name="Gold Value", distribution=DistributionType.EXPONENTIAL, min_val=1, max_val=1000),
            AttributeConfig(name="Durability", distribution=DistributionType.NORMAL, min_val=1, max_val=100),
        ]
    ),
    "npc_stats": Profile(
        name="NPC Stats",
        attributes=[
            AttributeConfig(name="Level", distribution=DistributionType.UNIFORM, min_val=1, max_val=20),
            AttributeConfig(name="Health", distribution=DistributionType.NORMAL, min_val=10, max_val=200),
            AttributeConfig(name="Aggression", distribution=DistributionType.UNIFORM, min_val=0, max_val=100),
        ]
    ),
    "romantic_interest": Profile(
        name="Romantic Interest Generator",
        attributes=[
            # Ethnicity (weighted table - first attribute)
            AttributeConfig(
                name="Ethnicity",
                distribution=DistributionType.WEIGHTED,
                min_val=1,
                max_val=39,
                params={
                    "entries": [
                        {"value": 1, "weight": 120},   # Karnataka: Bengaluru/Mysuru belt
                        {"value": 2, "weight": 60},    # Karnataka: North Karnataka
                        {"value": 3, "weight": 55},    # Karnataka: Coastal Karnataka
                        {"value": 4, "weight": 25},    # Karnataka: Kalyana Karnataka
                        {"value": 5, "weight": 20},    # Karnataka: Other districts
                        {"value": 6, "weight": 110},   # Telugu: Telangana
                        {"value": 7, "weight": 60},    # Telugu: Coastal Andhra
                        {"value": 8, "weight": 30},    # Telugu: Rayalaseema
                        {"value": 9, "weight": 70},    # Tamil: Chennai + North TN
                        {"value": 10, "weight": 40},   # Tamil: Kongu belt
                        {"value": 11, "weight": 25},   # Tamil: Cauvery delta
                        {"value": 12, "weight": 15},   # Tamil: South TN
                        {"value": 13, "weight": 55},   # Hindi belt: UP
                        {"value": 14, "weight": 35},   # Hindi belt: Bihar/Jharkhand
                        {"value": 15, "weight": 20},   # Hindi belt: MP/Chhattisgarh
                        {"value": 16, "weight": 20},   # Hindi belt: Rajasthan
                        {"value": 17, "weight": 10},   # Hindi belt: Uttarakhand/Himachal
                        {"value": 18, "weight": 10},   # Hindi belt: Haryana/Delhi-NCR
                        {"value": 19, "weight": 30},   # Malayalam: Malabar
                        {"value": 20, "weight": 25},   # Malayalam: Central Kerala
                        {"value": 21, "weight": 15},   # Malayalam: South Kerala
                        {"value": 22, "weight": 25},   # Marathi: Mumbai/Pune
                        {"value": 23, "weight": 10},   # Marathi: Vidarbha
                        {"value": 24, "weight": 5},    # Marathi: Marathwada
                        {"value": 25, "weight": 18},   # Bengali: Kolkata
                        {"value": 26, "weight": 7},    # Bengali: North Bengal
                        {"value": 27, "weight": 5},    # Bengali: Other WB
                        {"value": 28, "weight": 18},   # Gujarati
                        {"value": 29, "weight": 12},   # Marwari/Rajasthani business
                        {"value": 30, "weight": 10},   # Odia: Coastal
                        {"value": 31, "weight": 5},    # Odia: Western/Southern
                        {"value": 32, "weight": 10},   # Punjabi
                        {"value": 33, "weight": 5},    # Sindhi
                        {"value": 34, "weight": 4},    # Northeast: Assamese
                        {"value": 35, "weight": 2},    # Northeast: Manipuri
                        {"value": 36, "weight": 2},    # Northeast: Naga
                        {"value": 37, "weight": 2},    # Northeast: Mizo
                        {"value": 38, "weight": 7},    # Other Indian
                        {"value": 39, "weight": 3},    # Foreign nationals/expats
                    ]
                },
                label_config=LabelConfig(
                    enabled=True,
                    ranges=[
                        LabelRange(label="Karnataka: Bengaluru/Mysuru belt", min_val=1, max_val=1, color="#4CAF50"),
                        LabelRange(label="Karnataka: North Karnataka (Belagavi–Hubballi–Dharwad–Ballari)", min_val=2, max_val=2, color="#4CAF50"),
                        LabelRange(label="Karnataka: Coastal Karnataka (Udupi–DK; Tulu/Konkani)", min_val=3, max_val=3, color="#4CAF50"),
                        LabelRange(label="Karnataka: Kalyana Karnataka / Hyderabad-Karnataka", min_val=4, max_val=4, color="#4CAF50"),
                        LabelRange(label="Karnataka: Other districts", min_val=5, max_val=5, color="#4CAF50"),
                        LabelRange(label="Telugu: Telangana (incl. Hyderabad)", min_val=6, max_val=6, color="#2196F3"),
                        LabelRange(label="Telugu: Coastal Andhra", min_val=7, max_val=7, color="#2196F3"),
                        LabelRange(label="Telugu: Rayalaseema", min_val=8, max_val=8, color="#2196F3"),
                        LabelRange(label="Tamil: Chennai + North TN corridor", min_val=9, max_val=9, color="#9C27B0"),
                        LabelRange(label="Tamil: Kongu belt (Coimbatore–Erode)", min_val=10, max_val=10, color="#9C27B0"),
                        LabelRange(label="Tamil: Cauvery delta", min_val=11, max_val=11, color="#9C27B0"),
                        LabelRange(label="Tamil: South TN (Madurai–Tirunelveli)", min_val=12, max_val=12, color="#9C27B0"),
                        LabelRange(label="Hindi belt: Uttar Pradesh", min_val=13, max_val=13, color="#FF9800"),
                        LabelRange(label="Hindi belt: Bihar/Jharkhand", min_val=14, max_val=14, color="#FF9800"),
                        LabelRange(label="Hindi belt: Madhya Pradesh/Chhattisgarh", min_val=15, max_val=15, color="#FF9800"),
                        LabelRange(label="Hindi belt: Rajasthan (non-mercantile)", min_val=16, max_val=16, color="#FF9800"),
                        LabelRange(label="Hindi belt: Uttarakhand/Himachal", min_val=17, max_val=17, color="#FF9800"),
                        LabelRange(label="Hindi belt: Haryana/Delhi-NCR", min_val=18, max_val=18, color="#FF9800"),
                        LabelRange(label="Malayalam: Malabar (North Kerala)", min_val=19, max_val=19, color="#00BCD4"),
                        LabelRange(label="Malayalam: Central Kerala", min_val=20, max_val=20, color="#00BCD4"),
                        LabelRange(label="Malayalam: South Kerala", min_val=21, max_val=21, color="#00BCD4"),
                        LabelRange(label="Marathi: Mumbai/Pune metro", min_val=22, max_val=22, color="#F44336"),
                        LabelRange(label="Marathi: Vidarbha (Nagpur belt)", min_val=23, max_val=23, color="#F44336"),
                        LabelRange(label="Marathi: Marathwada", min_val=24, max_val=24, color="#F44336"),
                        LabelRange(label="Bengali: Kolkata metro", min_val=25, max_val=25, color="#E91E63"),
                        LabelRange(label="Bengali: North Bengal", min_val=26, max_val=26, color="#E91E63"),
                        LabelRange(label="Bengali: Other WB diaspora", min_val=27, max_val=27, color="#E91E63"),
                        LabelRange(label="Gujarati: Ahmedabad/Surat + Saurashtra/Kutch", min_val=28, max_val=28, color="#FFEB3B"),
                        LabelRange(label="Marwari/Rajasthani business families", min_val=29, max_val=29, color="#795548"),
                        LabelRange(label="Odia: Coastal (Cuttack–Bhubaneswar)", min_val=30, max_val=30, color="#607D8B"),
                        LabelRange(label="Odia: Western/Southern Odisha", min_val=31, max_val=31, color="#607D8B"),
                        LabelRange(label="Punjabi", min_val=32, max_val=32, color="#8BC34A"),
                        LabelRange(label="Sindhi", min_val=33, max_val=33, color="#03A9F4"),
                        LabelRange(label="Northeast: Assamese", min_val=34, max_val=34, color="#673AB7"),
                        LabelRange(label="Northeast: Manipuri", min_val=35, max_val=35, color="#673AB7"),
                        LabelRange(label="Northeast: Naga groups", min_val=36, max_val=36, color="#673AB7"),
                        LabelRange(label="Northeast: Mizo", min_val=37, max_val=37, color="#673AB7"),
                        LabelRange(label="Other Indian (Goan/Konkani, Nepali, Kashmiri, etc.)", min_val=38, max_val=38, color="#9E9E9E"),
                        LabelRange(label="Foreign nationals/expats", min_val=39, max_val=39, color="#FF5722"),
                    ]
                )
            ),
            # Skin colour
            AttributeConfig(name="Skin colour", distribution=DistributionType.NORMAL, min_val=1, max_val=100, params={"mean": 50}),
            # Face
            AttributeConfig(name="Face - sharpness of features", distribution=DistributionType.NORMAL, min_val=1, max_val=100, params={"mean": 50}),
            AttributeConfig(name="Face - prettiness", distribution=DistributionType.NORMAL, min_val=1, max_val=100, params={"mean": 65}),
            AttributeConfig(name="Grooming and health of skin, hair etc.", distribution=DistributionType.NORMAL, min_val=1, max_val=100, params={"mean": 65}),
            # Style/Presentation
            AttributeConfig(name="Aesthetic taste and discernment", distribution=DistributionType.NORMAL, min_val=1, max_val=100, params={"mean": 65}),
            AttributeConfig(name="Femininity in presentation", distribution=DistributionType.NORMAL, min_val=1, max_val=100, params={"mean": 60}),
            AttributeConfig(name="Comfort with skin-show", distribution=DistributionType.NORMAL, min_val=1, max_val=100, params={"mean": 50}),
            AttributeConfig(name="Comfort with form-fitting clothing", distribution=DistributionType.NORMAL, min_val=1, max_val=100, params={"mean": 50}),
            AttributeConfig(name="Intensity of styling effort", distribution=DistributionType.NORMAL, min_val=1, max_val=100, params={"mean": 65}),
            # Build
            AttributeConfig(name="Build (slender vs thick-set)", distribution=DistributionType.UNIFORM, min_val=1, max_val=100),  # Flat
            AttributeConfig(name="Fitness level", distribution=DistributionType.NORMAL, min_val=1, max_val=100, params={"mean": 65}),
            AttributeConfig(name="Upper body femininity", distribution=DistributionType.NORMAL, min_val=1, max_val=100, params={"mean": 65}),
            AttributeConfig(name="Waist definition", distribution=DistributionType.NORMAL, min_val=1, max_val=100, params={"mean": 65}),
            AttributeConfig(name="Lower body prominence", distribution=DistributionType.NORMAL, min_val=1, max_val=100, params={"mean": 65}),
            AttributeConfig(name="Limb length", distribution=DistributionType.NORMAL, min_val=1, max_val=100, params={"mean": 65}),
            AttributeConfig(name="Limb shapeliness", distribution=DistributionType.NORMAL, min_val=1, max_val=100, params={"mean": 65}),
            # Body Features
            AttributeConfig(name="Breast size", distribution=DistributionType.NORMAL, min_val=1, max_val=100, params={"mean": 50}),
            AttributeConfig(name="Breast shape", distribution=DistributionType.NORMAL, min_val=1, max_val=100, params={"mean": 50}),
            AttributeConfig(name="Hip width", distribution=DistributionType.NORMAL, min_val=1, max_val=100, params={"mean": 50}),
            AttributeConfig(name="Hip height", distribution=DistributionType.NORMAL, min_val=1, max_val=100, params={"mean": 50}),
            AttributeConfig(name="Rear protrusion", distribution=DistributionType.NORMAL, min_val=1, max_val=100, params={"mean": 65}),
            AttributeConfig(name="Rear apex (X)", distribution=DistributionType.NORMAL, min_val=1, max_val=100, params={"mean": 60}),
            AttributeConfig(name="Rear apex (Y)", distribution=DistributionType.NORMAL, min_val=1, max_val=100, params={"mean": 60}),
            AttributeConfig(name="Thigh gap", distribution=DistributionType.NORMAL, min_val=1, max_val=100, params={"mean": 50}),
            # Demeanor
            AttributeConfig(name="Posture and gait", distribution=DistributionType.NORMAL, min_val=1, max_val=100, params={"mean": 65}),
            AttributeConfig(name="Hardness of demeanor", distribution=DistributionType.NORMAL, min_val=1, max_val=100, params={"mean": 50}),
            AttributeConfig(name="Confidence", distribution=DistributionType.NORMAL, min_val=1, max_val=100, params={"mean": 65}),
            AttributeConfig(name="Energy", distribution=DistributionType.NORMAL, min_val=1, max_val=100, params={"mean": 65}),
            AttributeConfig(name="Propensity to fidget", distribution=DistributionType.NORMAL, min_val=1, max_val=100, params={"mean": 65}),
            AttributeConfig(name="Lady-likeness of body language", distribution=DistributionType.NORMAL, min_val=1, max_val=100, params={"mean": 65}),
            AttributeConfig(name="Vigilance for unwanted attention", distribution=DistributionType.NORMAL, min_val=1, max_val=100, params={"mean": 65}),
        ]
    ),
}


def get_preset_profile(name: str) -> Profile:
    """Get a copy of a preset profile."""
    if name not in PRESET_PROFILES:
        raise ValueError(f"Unknown preset: {name}")
    return Profile.from_dict(PRESET_PROFILES[name].to_dict())
