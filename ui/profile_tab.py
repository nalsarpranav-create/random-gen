"""Profile builder UI for batch attribute generation."""

import streamlit as st
from typing import Optional
from pathlib import Path

from models.profile import (
    Profile, AttributeConfig, DistributionType, DependencyType,
    PRESET_PROFILES, get_preset_profile
)
from models.labels import LabelConfig, LabelRange
from generators.profile_generator import generate_profile, ProfileResult
from utils.storage import (
    save_profile, load_profile, list_saved_profiles, delete_profile
)


def init_profile_state() -> None:
    """Initialize profile state."""
    if "current_profile" not in st.session_state:
        st.session_state.current_profile = Profile(name="My Profile")
    if "profile_result" not in st.session_state:
        st.session_state.profile_result = None


def get_profile() -> Profile:
    """Get current profile."""
    init_profile_state()
    return st.session_state.current_profile


def set_profile(profile: Profile) -> None:
    """Set current profile."""
    st.session_state.current_profile = profile


def render_attribute_card(attr: AttributeConfig, index: int, all_names: list) -> Optional[AttributeConfig]:
    """Render an editable attribute card. Returns updated config or None if deleted."""

    with st.container():
        # Header with name and delete
        col1, col2 = st.columns([4, 1])
        with col1:
            new_name = st.text_input(
                "Name",
                value=attr.name,
                key=f"attr_name_{index}",
                label_visibility="collapsed",
                placeholder="Attribute name"
            )
        with col2:
            if st.button("🗑️", key=f"attr_del_{index}"):
                return None

        # Distribution selection
        dist_options = [d.value for d in DistributionType]
        dist_idx = dist_options.index(attr.distribution.value)
        new_dist = st.selectbox(
            "Distribution",
            options=dist_options,
            index=dist_idx,
            key=f"attr_dist_{index}",
            format_func=lambda x: x.title()
        )

        # Range (for most distributions)
        if new_dist != "dice":
            c1, c2 = st.columns(2)
            with c1:
                new_min = st.number_input("Min", value=int(attr.min_val), key=f"attr_min_{index}")
            with c2:
                new_max = st.number_input("Max", value=int(attr.max_val), key=f"attr_max_{index}")
        else:
            new_min, new_max = attr.min_val, attr.max_val

        # Distribution-specific params
        new_params = attr.params.copy()

        if new_dist == "dice":
            notation = st.text_input(
                "Dice Notation",
                value=attr.params.get("notation", "3d6"),
                key=f"attr_dice_{index}",
                help="e.g., 3d6, 2d8+5"
            )
            new_params["notation"] = notation

        elif new_dist == "normal":
            c1, c2 = st.columns(2)
            with c1:
                mean = st.number_input(
                    "Mean",
                    value=float(attr.params.get("mean", (new_min + new_max) / 2)),
                    key=f"attr_mean_{index}"
                )
            with c2:
                std = st.number_input(
                    "Std Dev",
                    value=float(attr.params.get("std", (new_max - new_min) / 6)),
                    key=f"attr_std_{index}",
                    min_value=0.1
                )
            new_params["mean"] = mean
            new_params["std"] = std

        elif new_dist == "triangular":
            mode = st.slider(
                "Peak",
                min_value=float(new_min),
                max_value=float(new_max),
                value=float(attr.params.get("mode", (new_min + new_max) / 2)),
                key=f"attr_mode_{index}"
            )
            new_params["mode"] = mode

        # Labels toggle (simplified)
        use_labels = st.checkbox(
            "Use Labels",
            value=attr.label_config.enabled if attr.label_config else False,
            key=f"attr_labels_{index}"
        )

        new_label_config = attr.label_config
        if use_labels and not attr.label_config:
            new_label_config = LabelConfig(enabled=True)
        elif use_labels and attr.label_config:
            new_label_config.enabled = True
        elif not use_labels and attr.label_config:
            new_label_config.enabled = False

        st.markdown("---")

        return AttributeConfig(
            name=new_name,
            distribution=DistributionType(new_dist),
            min_val=new_min,
            max_val=new_max,
            params=new_params,
            label_config=new_label_config,
            depends_on=attr.depends_on,
            dependency_type=attr.dependency_type,
            dependency_params=attr.dependency_params,
        )


def render_result_display(result: ProfileResult) -> None:
    """Render the profile generation result."""

    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 16px;
            padding: 25px;
            margin: 15px 0;
            box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
        ">
            <div style="
                text-align: center;
                font-size: 14px;
                color: rgba(255,255,255,0.8);
                margin-bottom: 15px;
                text-transform: uppercase;
                letter-spacing: 2px;
            ">
                {result.profile_name}
            </div>
        """,
        unsafe_allow_html=True
    )

    # Show each attribute
    for attr in result.attributes:
        st.markdown(
            f"""
            <div style="
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 12px 20px;
                background: rgba(255,255,255,0.1);
                border-radius: 10px;
                margin: 8px 0;
            ">
                <span style="color: rgba(255,255,255,0.9); font-size: 16px;">
                    {attr.name}
                </span>
                <span style="
                    color: white;
                    font-size: 24px;
                    font-weight: 700;
                ">
                    {attr.display()}
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("</div>", unsafe_allow_html=True)


def render_save_load_section(profile: Profile) -> None:
    """Render the save/load section."""

    with st.expander("💾 Save / Load", expanded=False):
        tab_save, tab_load = st.tabs(["Save", "Load"])

        with tab_save:
            c1, c2 = st.columns([3, 1])
            with c1:
                st.caption(f"Save '{profile.name}' to disk")
            with c2:
                if st.button("Save", use_container_width=True):
                    try:
                        path = save_profile(profile, overwrite=True)
                        st.success(f"Saved!")
                    except Exception as e:
                        st.error(f"Failed: {e}")

        with tab_load:
            saved = list_saved_profiles()

            if not saved:
                st.caption("No saved profiles yet")
            else:
                for item in saved:
                    c1, c2, c3 = st.columns([3, 1, 1])
                    with c1:
                        st.markdown(f"**{item['name']}**")
                        st.caption(f"{item['num_attributes']} attributes")
                    with c2:
                        if st.button("Load", key=f"load_{item['path']}", use_container_width=True):
                            try:
                                loaded = load_profile(item['path'])
                                set_profile(loaded)
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed: {e}")
                    with c3:
                        if st.button("🗑️", key=f"del_{item['path']}"):
                            delete_profile(item['path'])
                            st.rerun()


def render_profile_tab() -> None:
    """Render the profile builder tab."""
    st.header("Profile Builder")

    init_profile_state()
    profile = get_profile()

    # Two columns
    editor_col, result_col = st.columns([3, 2])

    with editor_col:
        # Profile name and save/load
        c1, c2 = st.columns([3, 1])
        with c1:
            profile.name = st.text_input("Profile Name", value=profile.name, label_visibility="collapsed", placeholder="Profile name")
        with c2:
            pass  # Space for alignment

        # Save/Load section
        render_save_load_section(profile)

        # Preset profiles
        st.markdown("##### Quick Start")
        preset_cols = st.columns(len(PRESET_PROFILES))
        for i, (key, preset) in enumerate(PRESET_PROFILES.items()):
            with preset_cols[i]:
                if st.button(preset.name, key=f"preset_{key}", use_container_width=True):
                    set_profile(get_preset_profile(key))
                    st.rerun()

        st.markdown("---")

        # Attributes
        st.markdown("##### Attributes")

        all_names = profile.get_attribute_names()
        updated_attrs = []

        for i, attr in enumerate(profile.attributes):
            result = render_attribute_card(attr, i, all_names)
            if result is not None:
                updated_attrs.append(result)

        profile.attributes = updated_attrs
        set_profile(profile)

        # Add attribute button
        if st.button("+ Add Attribute", use_container_width=True):
            profile.add_attribute(AttributeConfig(
                name=f"Attribute {len(profile.attributes) + 1}",
                distribution=DistributionType.UNIFORM,
            ))
            set_profile(profile)
            st.rerun()

        # Validation
        validation = profile.validate()
        if validation["errors"]:
            for err in validation["errors"]:
                st.error(err)
        if validation["warnings"]:
            for warn in validation["warnings"]:
                st.warning(warn)

    with result_col:
        st.markdown("### Generate")

        # Generate button
        can_generate = len(profile.attributes) > 0 and not validation["errors"]

        if st.button(
            "🎲  Generate All",
            type="primary",
            use_container_width=True,
            disabled=not can_generate
        ):
            try:
                st.session_state.profile_result = generate_profile(profile)
            except Exception as e:
                st.error(f"Generation failed: {e}")

        # Result display
        if st.session_state.profile_result:
            render_result_display(st.session_state.profile_result)

            # Re-roll button
            if st.button("🔄 Roll Again", use_container_width=True):
                st.session_state.profile_result = generate_profile(profile)
                st.rerun()

            # Export options
            st.markdown("##### Export")
            result = st.session_state.profile_result

            # Format as text
            text_export = f"{result.profile_name}\n{'='*40}\n"
            for attr in result.attributes:
                text_export += f"{attr.name}: {attr.display()}\n"

            # Format as JSON
            import json
            json_export = json.dumps(result.to_dict(), indent=2)

            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "📄 Download Text",
                    data=text_export,
                    file_name=f"{result.profile_name.lower().replace(' ', '_')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            with col2:
                st.download_button(
                    "📋 Download JSON",
                    data=json_export,
                    file_name=f"{result.profile_name.lower().replace(' ', '_')}.json",
                    mime="application/json",
                    use_container_width=True
                )
        else:
            st.markdown(
                """
                <div style="
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                    align-items: center;
                    min-height: 300px;
                    background: #f8f9fa;
                    border-radius: 16px;
                    border: 2px dashed #dee2e6;
                    color: #adb5bd;
                ">
                    <div style="font-size: 40px; margin-bottom: 10px;">🎭</div>
                    <div>Add attributes and click Generate</div>
                </div>
                """,
                unsafe_allow_html=True
            )
