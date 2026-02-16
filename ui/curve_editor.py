"""Custom curve editor UI with preset gallery and simple adjustments."""

import streamlit as st
import plotly.graph_objects as go
import numpy as np

from generators.custom_curve import (
    CurveConfig, CustomCurveGenerator,
    CURVE_PRESETS, get_preset_config, get_presets_by_category, get_category_order
)
from ui.label_editor import render_label_editor, get_label_config


def init_curve_state() -> None:
    """Initialize curve state."""
    if "selected_preset" not in st.session_state:
        st.session_state.selected_preset = "bell"
    if "curve_bias" not in st.session_state:
        st.session_state.curve_bias = 0.0
    if "curve_spread" not in st.session_state:
        st.session_state.curve_spread = 0.0
    if "curve_result" not in st.session_state:
        st.session_state.curve_result = None
        st.session_state.curve_count = 1


def get_adjusted_config() -> CurveConfig:
    """Get the current curve config with bias/spread adjustments applied."""
    base_config = get_preset_config(st.session_state.selected_preset)

    # Apply adjustments
    config = base_config
    if abs(st.session_state.curve_bias) > 0.01:
        config = config.with_bias(st.session_state.curve_bias)
    if abs(st.session_state.curve_spread) > 0.01:
        config = config.with_spread(st.session_state.curve_spread)

    return config


def render_mini_curve(preset_key: str, size: int = 50) -> go.Figure:
    """Create a mini curve preview figure."""
    config = get_preset_config(preset_key)
    gen = CustomCurveGenerator(min_val=0, max_val=1, curve_config=config)
    x, y = gen.get_curve_for_display(40)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y,
        fill='tozeroy',
        fillcolor='rgba(99, 102, 241, 0.4)',
        line=dict(color='rgb(99, 102, 241)', width=1.5),
        hoverinfo='skip'
    ))
    fig.update_layout(
        height=size,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    return fig


def render_preset_gallery() -> None:
    """Render the preset gallery organized by category."""
    categories = get_presets_by_category()
    category_order = get_category_order()

    for cat in category_order:
        if cat not in categories:
            continue

        presets = categories[cat]

        st.markdown(f"**{cat}**")

        # Display presets in rows of 4
        cols_per_row = 4
        for i in range(0, len(presets), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                idx = i + j
                if idx < len(presets):
                    preset_key, preset_data = presets[idx]
                    is_selected = st.session_state.selected_preset == preset_key

                    with col:
                        # Mini preview
                        fig = render_mini_curve(preset_key, size=45)
                        st.plotly_chart(
                            fig,
                            config={'displayModeBar': False},
                            use_container_width=True,
                            key=f"preview_{preset_key}"
                        )

                        # Button with selection indicator
                        btn_type = "primary" if is_selected else "secondary"
                        icon = preset_data.get("icon", "")
                        label = f"{icon} {preset_data['name']}" if icon else preset_data['name']

                        if st.button(
                            label,
                            key=f"preset_{preset_key}",
                            use_container_width=True,
                            type=btn_type,
                            help=preset_data.get("description", "")
                        ):
                            st.session_state.selected_preset = preset_key
                            # Reset adjustments when changing preset
                            st.session_state.curve_bias = 0.0
                            st.session_state.curve_spread = 0.0
                            st.rerun()

        st.markdown("")  # Spacing between categories


def render_main_curve(config: CurveConfig, min_val: float, max_val: float) -> None:
    """Render the main probability curve visualization."""
    gen = CustomCurveGenerator(min_val=min_val, max_val=max_val, curve_config=config)
    x_curve, y_curve = gen.get_curve_for_display(200)

    # Get label colors if enabled
    label_config = get_label_config("curve_labels")

    fig = go.Figure()

    if label_config.enabled and label_config.ranges:
        # Color by labels
        for r in label_config.get_sorted_ranges():
            mask = (x_curve >= r.min_val) & (x_curve <= r.max_val)
            if mask.any():
                x_region = x_curve[mask]
                y_region = y_curve[mask]
                fig.add_trace(go.Scatter(
                    x=np.concatenate([[x_region[0]], x_region, [x_region[-1]]]),
                    y=np.concatenate([[0], y_region, [0]]),
                    fill='toself',
                    fillcolor=r.color + "50",
                    line=dict(color=r.color, width=2),
                    name=r.label,
                    hoverinfo='skip'
                ))
    else:
        # Single color curve with gradient fill
        fig.add_trace(go.Scatter(
            x=x_curve, y=y_curve,
            fill='tozeroy',
            fillcolor='rgba(99, 102, 241, 0.3)',
            line=dict(color='rgb(99, 102, 241)', width=3),
            name='Probability',
            hoverinfo='skip'
        ))

    fig.update_layout(
        height=220,
        margin=dict(l=40, r=20, t=10, b=40),
        showlegend=False,
        xaxis=dict(
            title="Value",
            range=[min_val, max_val],
            gridcolor='rgba(0,0,0,0.08)',
            zeroline=False,
        ),
        yaxis=dict(
            title="Likelihood",
            gridcolor='rgba(0,0,0,0.08)',
            rangemode='tozero',
            zeroline=False,
        ),
        plot_bgcolor='white',
    )

    st.plotly_chart(fig, config={'displayModeBar': False}, use_container_width=True)


def render_adjustment_sliders() -> None:
    """Render the bias and spread adjustment sliders."""

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Bias** — _shift toward low or high_")
        new_bias = st.slider(
            "Bias",
            min_value=-1.0,
            max_value=1.0,
            value=st.session_state.curve_bias,
            step=0.05,
            format="%.2f",
            key="bias_slider",
            label_visibility="collapsed",
            help="Negative = favor lower values, Positive = favor higher values"
        )

        # Show labels under slider
        c1, c2, c3 = st.columns(3)
        with c1:
            st.caption("← Lower")
        with c2:
            st.caption("Neutral", help="No shift")
        with c3:
            st.caption("Higher →")

        if new_bias != st.session_state.curve_bias:
            st.session_state.curve_bias = new_bias

    with col2:
        st.markdown("**Spread** — _concentrated or diffuse_")
        new_spread = st.slider(
            "Spread",
            min_value=-1.0,
            max_value=1.0,
            value=st.session_state.curve_spread,
            step=0.05,
            format="%.2f",
            key="spread_slider",
            label_visibility="collapsed",
            help="Negative = more concentrated/peaked, Positive = more spread out/uniform"
        )

        # Show labels under slider
        c1, c2, c3 = st.columns(3)
        with c1:
            st.caption("← Peaked")
        with c2:
            st.caption("Normal", help="No change")
        with c3:
            st.caption("Uniform →")

        if new_spread != st.session_state.curve_spread:
            st.session_state.curve_spread = new_spread

    # Reset button
    if st.session_state.curve_bias != 0.0 or st.session_state.curve_spread != 0.0:
        if st.button("Reset Adjustments", use_container_width=True):
            st.session_state.curve_bias = 0.0
            st.session_state.curve_spread = 0.0
            st.rerun()


def render_result_display(result, count: int) -> None:
    """Render the generation result."""
    display_values = result.get_display_values()

    if count == 1:
        st.markdown(
            f"""
            <div style="
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 180px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 20px;
                margin: 15px 0;
                box-shadow: 0 10px 40px rgba(102, 126, 234, 0.4);
            ">
                <div style="
                    font-size: 56px;
                    font-weight: 800;
                    color: white;
                    text-shadow: 2px 4px 8px rgba(0,0,0,0.3);
                ">
                    {display_values[0]}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div style="
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 16px;
                padding: 15px;
                margin: 10px 0;
            ">
            """,
            unsafe_allow_html=True
        )

        cols_per_row = min(4, count)
        for i in range(0, len(display_values), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                idx = i + j
                if idx < len(display_values):
                    with col:
                        st.markdown(
                            f"""
                            <div style="
                                text-align: center;
                                padding: 10px;
                                background: rgba(255,255,255,0.15);
                                border-radius: 8px;
                                margin: 3px 0;
                                font-size: 18px;
                                font-weight: 600;
                                color: white;
                            ">
                                {display_values[idx]}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

        st.markdown("</div>", unsafe_allow_html=True)

        # Statistics
        values = result.values
        st.caption(
            f"Min: {min(values):.1f} · Max: {max(values):.1f} · "
            f"Mean: {np.mean(values):.1f}"
        )


def render_curve_editor_tab() -> None:
    """Render the complete curve editor tab."""

    init_curve_state()

    # Two-column layout
    left_col, right_col = st.columns([3, 2])

    with left_col:
        # Range settings at top
        st.markdown("### Range")
        c1, c2 = st.columns(2)
        with c1:
            min_val = st.number_input("Min", value=1, step=1, key="curve_min")
        with c2:
            max_val = st.number_input("Max", value=100, step=1, key="curve_max")

        if min_val >= max_val:
            st.error("Min must be less than Max")
            return

        # Current curve preview
        st.markdown("### Your Distribution")
        config = get_adjusted_config()
        render_main_curve(config, min_val, max_val)

        # Adjustment sliders
        render_adjustment_sliders()

        # Labels section
        with st.expander("Labels", expanded=False):
            render_label_editor(
                global_min=min_val,
                global_max=max_val,
                key="curve_labels"
            )

    with right_col:
        st.markdown("### Generate")

        # Count input
        count = st.number_input(
            "How many?",
            min_value=1,
            max_value=100,
            value=1,
            key="curve_count_input"
        )

        # Create generator
        label_config = get_label_config("curve_labels")
        generator = CustomCurveGenerator(
            min_val=min_val,
            max_val=max_val,
            curve_config=config,
            label_config=label_config
        )

        # Generate button
        if st.button("🎲  Generate", type="primary", use_container_width=True, key="curve_generate"):
            st.session_state.curve_result = generator.generate(count)
            st.session_state.curve_count = count

        # Result display
        if st.session_state.curve_result:
            render_result_display(st.session_state.curve_result, st.session_state.curve_count)
        else:
            st.markdown(
                """
                <div style="
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    min-height: 180px;
                    background: #f8f9fa;
                    border-radius: 16px;
                    border: 2px dashed #dee2e6;
                    color: #adb5bd;
                ">
                    Click Generate
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("---")

        # Preset gallery below generate
        st.markdown("### Choose a Shape")

        # Scrollable preset gallery
        with st.container(height=450):
            render_preset_gallery()
