"""Hybrid label range editor with visual bar and table."""

import streamlit as st
import pandas as pd
from typing import Optional

from models.labels import LabelConfig, LabelRange, get_default_color


def init_label_config_state(key: str = "label_config") -> None:
    """Initialize label config in session state if not present."""
    if key not in st.session_state:
        st.session_state[key] = LabelConfig()


def get_label_config(key: str = "label_config") -> LabelConfig:
    """Get label config from session state."""
    init_label_config_state(key)
    return st.session_state[key]


def set_label_config(config: LabelConfig, key: str = "label_config") -> None:
    """Set label config in session state."""
    st.session_state[key] = config


def render_visual_bar(
    config: LabelConfig,
    global_min: float,
    global_max: float,
    key: str = "label_config"
) -> None:
    """
    Render the visual bar showing label ranges as colored segments.

    Args:
        config: The label configuration
        global_min: Minimum value of the overall range
        global_max: Maximum value of the overall range
        key: Session state key for the config
    """
    if not config.ranges:
        st.info("No label ranges defined. Add labels using the table below.")
        return

    sorted_ranges = config.get_sorted_ranges()
    total_width = global_max - global_min

    # Find gaps
    gaps = config.find_gaps(global_min, global_max)

    # Build segments (ranges + gaps)
    segments = []
    current_pos = global_min

    for r in sorted_ranges:
        # Add gap before this range if exists
        if r.min_val > current_pos:
            segments.append({
                "type": "gap",
                "start": current_pos,
                "end": r.min_val,
                "width": (r.min_val - current_pos) / total_width
            })

        # Add the range
        segments.append({
            "type": "range",
            "range": r,
            "start": r.min_val,
            "end": r.max_val,
            "width": r.width() / total_width
        })
        current_pos = r.max_val

    # Add gap after last range
    if current_pos < global_max:
        segments.append({
            "type": "gap",
            "start": current_pos,
            "end": global_max,
            "width": (global_max - current_pos) / total_width
        })

    # Render segments using columns with relative widths
    # Ensure minimum display width
    min_width_pct = 0.05
    widths = [max(s["width"], min_width_pct) for s in segments]

    # Normalize widths
    total = sum(widths)
    widths = [w / total for w in widths]

    cols = st.columns(widths)

    for col, segment in zip(cols, segments):
        with col:
            if segment["type"] == "gap":
                # Render gap with striped pattern
                st.markdown(
                    f"""
                    <div style="
                        background: repeating-linear-gradient(
                            45deg,
                            #f0f0f0,
                            #f0f0f0 10px,
                            #e0e0e0 10px,
                            #e0e0e0 20px
                        );
                        padding: 20px 5px;
                        text-align: center;
                        border-radius: 4px;
                        font-size: 11px;
                        color: #666;
                    ">
                        Gap<br>
                        <small>{segment['start']:.0f}-{segment['end']:.0f}</small>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                r = segment["range"]
                # Render range with color
                st.markdown(
                    f"""
                    <div style="
                        background: {r.color};
                        padding: 20px 5px;
                        text-align: center;
                        border-radius: 4px;
                        color: white;
                        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
                        font-weight: bold;
                    ">
                        {r.label}<br>
                        <small style="font-weight: normal;">{r.min_val:.0f}-{r.max_val:.0f}</small>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


def render_table_editor(
    config: LabelConfig,
    global_min: float,
    global_max: float,
    key: str = "label_config"
) -> LabelConfig:
    """
    Render the editable table for label ranges.

    Args:
        config: The label configuration
        global_min: Minimum value of the overall range
        global_max: Maximum value of the overall range
        key: Session state key for the config

    Returns:
        Updated LabelConfig (may be the same if no changes)
    """
    # Convert to DataFrame for editing
    if config.ranges:
        df = pd.DataFrame([
            {
                "Label": r.label,
                "Min": r.min_val,
                "Max": r.max_val,
                "Color": r.color
            }
            for r in config.ranges
        ])
    else:
        df = pd.DataFrame(columns=["Label", "Min", "Max", "Color"])

    # Editable table
    edited_df = st.data_editor(
        df,
        column_config={
            "Label": st.column_config.TextColumn(
                "Label",
                help="Name for this range",
                max_chars=50,
                required=True
            ),
            "Min": st.column_config.NumberColumn(
                "Min",
                help="Minimum value (inclusive)",
                min_value=global_min,
                max_value=global_max,
                required=True
            ),
            "Max": st.column_config.NumberColumn(
                "Max",
                help="Maximum value (inclusive)",
                min_value=global_min,
                max_value=global_max,
                required=True
            ),
            "Color": st.column_config.TextColumn(
                "Color",
                help="Hex color code (e.g., #FF0000)",
                max_chars=7
            ),
        },
        num_rows="dynamic",
        use_container_width=True,
        key=f"{key}_table"
    )

    # Convert back to LabelConfig
    new_ranges = []
    for _, row in edited_df.iterrows():
        if pd.notna(row["Label"]) and pd.notna(row["Min"]) and pd.notna(row["Max"]):
            color = row["Color"] if pd.notna(row["Color"]) else get_default_color(len(new_ranges))
            new_ranges.append(LabelRange(
                label=str(row["Label"]),
                min_val=float(row["Min"]),
                max_val=float(row["Max"]),
                color=color
            ))

    new_config = LabelConfig(ranges=new_ranges, enabled=config.enabled)
    return new_config


def render_validation_messages(config: LabelConfig) -> bool:
    """
    Render validation errors and warnings.

    Args:
        config: The label configuration

    Returns:
        True if there are no errors (warnings OK), False if there are errors
    """
    validation = config.validate()

    if validation["errors"]:
        for error in validation["errors"]:
            st.error(f"Error: {error}")

    if validation["warnings"]:
        for warning in validation["warnings"]:
            st.warning(f"Warning: {warning}")

    return len(validation["errors"]) == 0


def render_label_editor(
    global_min: float = 1,
    global_max: float = 100,
    key: str = "label_config"
) -> LabelConfig:
    """
    Render the complete hybrid label editor.

    Args:
        global_min: Minimum value of the overall range
        global_max: Maximum value of the overall range
        key: Session state key for the config

    Returns:
        Current LabelConfig
    """
    config = get_label_config(key)

    # Enable toggle
    enabled = st.toggle(
        "Enable Labels",
        value=config.enabled,
        help="When enabled, generated numbers will be displayed as labels",
        key=f"{key}_enable_toggle"
    )

    if enabled != config.enabled:
        config.enabled = enabled
        set_label_config(config, key)

    if not enabled:
        st.caption("Enable labels to map number ranges to word labels like 'Common', 'Rare', etc.")
        return config

    st.markdown("---")

    # Visual bar
    st.subheader("Range Preview")
    render_visual_bar(config, global_min, global_max, key)

    st.markdown("---")

    # Action buttons
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("Add Label", key=f"{key}_add"):
            # Find a gap to fill, or append at end
            gaps = config.find_gaps(global_min, global_max)
            if gaps:
                gap_start, gap_end = gaps[0]
            else:
                # Append after last range
                if config.ranges:
                    last = max(r.max_val for r in config.ranges)
                    gap_start = last
                    gap_end = global_max
                else:
                    gap_start = global_min
                    gap_end = global_max

            error = config.add_range(
                label=f"Label_{len(config.ranges) + 1}",
                min_val=gap_start,
                max_val=gap_end,
                color=get_default_color(len(config.ranges))
            )
            if error:
                st.error(error)
            else:
                set_label_config(config, key)
                st.rerun()

    with col2:
        if st.button("Auto-fill Gaps", key=f"{key}_autofill"):
            filled = config.auto_fill_gaps(global_min, global_max)
            set_label_config(config, key)
            if filled > 0:
                st.success(f"Filled {filled} gap(s)")
                st.rerun()
            else:
                st.info("No gaps to fill")

    with col3:
        if st.button("Reset", key=f"{key}_reset"):
            config.reset()
            set_label_config(config, key)
            st.rerun()

    with col4:
        undo_col, redo_col = st.columns(2)
        with undo_col:
            if st.button("Undo", key=f"{key}_undo", disabled=not config.can_undo()):
                config.undo()
                set_label_config(config, key)
                st.rerun()
        with redo_col:
            if st.button("Redo", key=f"{key}_redo", disabled=not config.can_redo()):
                config.redo()
                set_label_config(config, key)
                st.rerun()

    st.markdown("---")

    # Table editor
    st.subheader("Edit Ranges")
    new_config = render_table_editor(config, global_min, global_max, key)

    # Check if config changed
    if new_config.to_dict() != config.to_dict():
        new_config.enabled = config.enabled
        set_label_config(new_config, key)
        config = new_config

    # Validation messages
    st.markdown("---")
    render_validation_messages(config)

    return config


def render_compact_label_editor(
    global_min: float = 1,
    global_max: float = 100,
    key: str = "label_config"
) -> LabelConfig:
    """
    Render a compact version of the label editor for use in expanders.

    Args:
        global_min: Minimum value of the overall range
        global_max: Maximum value of the overall range
        key: Session state key for the config

    Returns:
        Current LabelConfig
    """
    config = get_label_config(key)

    # Enable toggle
    enabled = st.checkbox(
        "Enable Labels",
        value=config.enabled,
        key=f"{key}_enable"
    )

    if enabled != config.enabled:
        config.enabled = enabled
        set_label_config(config, key)

    if not enabled:
        return config

    # Quick add presets
    st.caption("Quick Presets:")
    preset_col1, preset_col2 = st.columns(2)

    with preset_col1:
        if st.button("Rarity (4 tiers)", key=f"{key}_preset_rarity", use_container_width=True):
            config.ranges = [
                LabelRange("Common", global_min, global_min + (global_max - global_min) * 0.5, "#4CAF50"),
                LabelRange("Uncommon", global_min + (global_max - global_min) * 0.5, global_min + (global_max - global_min) * 0.8, "#2196F3"),
                LabelRange("Rare", global_min + (global_max - global_min) * 0.8, global_min + (global_max - global_min) * 0.95, "#9C27B0"),
                LabelRange("Legendary", global_min + (global_max - global_min) * 0.95, global_max, "#FF9800"),
            ]
            set_label_config(config, key)
            st.rerun()

    with preset_col2:
        if st.button("Success/Fail", key=f"{key}_preset_binary", use_container_width=True):
            mid = (global_min + global_max) / 2
            config.ranges = [
                LabelRange("Fail", global_min, mid, "#F44336"),
                LabelRange("Success", mid, global_max, "#4CAF50"),
            ]
            set_label_config(config, key)
            st.rerun()

    # Visual bar preview
    render_visual_bar(config, global_min, global_max, key)

    # Compact table
    new_config = render_table_editor(config, global_min, global_max, key)
    if new_config.to_dict() != config.to_dict():
        new_config.enabled = config.enabled
        set_label_config(new_config, key)
        config = new_config

    # Compact validation
    validation = config.validate()
    if validation["errors"]:
        st.error(f"{len(validation['errors'])} error(s)")
    elif validation["warnings"]:
        st.warning(f"{len(validation['warnings'])} warning(s)")

    return config
