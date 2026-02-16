"""Save/Load system for profiles and configurations."""

import json
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from models.profile import Profile


# Default storage directory
DEFAULT_STORAGE_DIR = Path.home() / ".random_gen" / "profiles"


def get_storage_dir() -> Path:
    """Get or create the storage directory."""
    storage_dir = DEFAULT_STORAGE_DIR
    storage_dir.mkdir(parents=True, exist_ok=True)
    return storage_dir


def sanitize_filename(name: str) -> str:
    """Convert a profile name to a safe filename."""
    # Remove/replace unsafe characters
    safe = "".join(c if c.isalnum() or c in " -_" else "_" for c in name)
    safe = safe.strip().replace(" ", "_")
    return safe[:50] or "unnamed"


def save_profile(profile: Profile, overwrite: bool = False) -> Path:
    """
    Save a profile to disk.

    Args:
        profile: The profile to save
        overwrite: If True, overwrite existing file. If False, create unique name.

    Returns:
        Path to the saved file
    """
    storage_dir = get_storage_dir()
    base_name = sanitize_filename(profile.name)

    if overwrite:
        filepath = storage_dir / f"{base_name}.json"
    else:
        # Find unique name
        filepath = storage_dir / f"{base_name}.json"
        counter = 1
        while filepath.exists():
            filepath = storage_dir / f"{base_name}_{counter}.json"
            counter += 1

    # Add metadata
    data = profile.to_dict()
    data["_metadata"] = {
        "saved_at": datetime.now().isoformat(),
        "version": "1.0"
    }

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    return filepath


def load_profile(filepath: Path) -> Profile:
    """
    Load a profile from disk.

    Args:
        filepath: Path to the profile JSON file

    Returns:
        Loaded Profile
    """
    with open(filepath, "r") as f:
        data = json.load(f)

    # Remove metadata before parsing
    data.pop("_metadata", None)

    return Profile.from_dict(data)


def list_saved_profiles() -> List[dict]:
    """
    List all saved profiles.

    Returns:
        List of dicts with 'name', 'path', 'saved_at' keys
    """
    storage_dir = get_storage_dir()
    profiles = []

    for filepath in storage_dir.glob("*.json"):
        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            metadata = data.get("_metadata", {})
            profiles.append({
                "name": data.get("name", filepath.stem),
                "path": filepath,
                "saved_at": metadata.get("saved_at", "Unknown"),
                "num_attributes": len(data.get("attributes", []))
            })
        except (json.JSONDecodeError, KeyError):
            # Skip invalid files
            continue

    # Sort by save date (newest first)
    profiles.sort(key=lambda x: x["saved_at"], reverse=True)
    return profiles


def delete_profile(filepath: Path) -> bool:
    """
    Delete a saved profile.

    Args:
        filepath: Path to the profile file

    Returns:
        True if deleted, False if not found
    """
    try:
        filepath.unlink()
        return True
    except FileNotFoundError:
        return False


def export_profile_json(profile: Profile) -> str:
    """Export profile as JSON string for sharing."""
    return profile.to_json()


def import_profile_json(json_str: str) -> Profile:
    """Import profile from JSON string."""
    return Profile.from_json(json_str)
