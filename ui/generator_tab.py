"""Simple generator tab UI."""

import streamlit as st
import plotly.graph_objects as go
import numpy as np

from generators.presets import (
    UniformGenerator,
    NormalGenerator,
    DiceRollGenerator,
    TriangularGenerator,
    ExponentialGenerator,
    AverageOfNGenerator,
)
from generators.base import BaseGenerator
from ui.label_editor import render_label_editor, get_label_config


def render_distribution_chart(generator: BaseGenerator, height: int = 200) -> None:
    """Render a simple probability distribution chart."""
    x, y = generator.get_probability_distribution(100)

    fig = go.Figure()

    # Get label config for coloring
    config = get_label_config()

    if config.enabled and config.ranges:
        # Color regions by label
        for r in config.get_sorted_ranges():
            mask = (x >= r.min_val) & (x <= r.max_val)
            if mask.any():
                x_region = x[mask]
                y_region = y[mask]
                fig.add_trace(go.Scatter(
                    x=np.concatenate([[x_region[0]], x_region, [x_region[-1]]]),
                    y=np.concatenate([[0], y_region, [0]]),
                    fill='toself',
                    fillcolor=r.color + "60",
                    line=dict(color=r.color, width=1),
                    name=r.label,
                    hoverinfo='skip'
                ))
    else:
        # Single color
        fig.add_trace(go.Scatter(
            x=np.concatenate([[x[0]], x, [x[-1]]]),
            y=np.concatenate([[0], y, [0]]),
            fill='toself',
            fillcolor='rgba(99, 102, 241, 0.4)',
            line=dict(color='rgb(99, 102, 241)', width=1),
            hoverinfo='skip'
        ))

    # Minimal layout - no axes, no toolbar
    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )

    st.plotly_chart(fig, config={'displayModeBar': False}, use_container_width=True)


def render_result_display(result, count: int) -> None:
    """Render the result in a prominent way."""
    display_values = result.get_display_values()

    if count == 1:
        # Single result - VERY prominent
        value = display_values[0]
        st.markdown(
            f"""
            <div style="
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 250px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 20px;
                margin: 20px 0;
                box-shadow: 0 10px 40px rgba(102, 126, 234, 0.4);
            ">
                <div style="
                    font-size: 72px;
                    font-weight: 800;
                    color: white;
                    text-shadow: 2px 4px 8px rgba(0,0,0,0.3);
                    letter-spacing: -2px;
                ">
                    {value}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        # Multiple results - grid display
        st.markdown(
            """
            <div style="
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 16px;
                padding: 20px;
                margin: 10px 0;
                box-shadow: 0 8px 30px rgba(102, 126, 234, 0.3);
            ">
            """,
            unsafe_allow_html=True
        )

        # Display in rows
        cols_per_row = min(5, count)
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
                                padding: 12px 8px;
                                background: rgba(255,255,255,0.15);
                                border-radius: 10px;
                                margin: 4px 0;
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

        # Compact statistics
        values = result.values
        st.caption(
            f"Min: {min(values):.1f} · Max: {max(values):.1f} · "
            f"Mean: {np.mean(values):.1f} · Std: {np.std(values):.2f}"
        )


def render_generator_tab() -> None:
    """Render the simple generator tab."""

    # Initialize result in session state
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
        st.session_state.last_count = 1

    # Settings in sidebar-style left column
    settings_col, result_col = st.columns([2, 3])

    with settings_col:
        st.markdown("### Settings")

        # Distribution selection
        distribution = st.selectbox(
            "Distribution",
            options=["Uniform", "Normal", "Dice Roll", "Triangular", "Exponential", "Average of N"],
        )

        # Common parameters
        if distribution != "Dice Roll":
            c1, c2 = st.columns(2)
            with c1:
                min_val = st.number_input("Min", value=1, step=1)
            with c2:
                max_val = st.number_input("Max", value=100, step=1)

            if min_val >= max_val:
                st.error("Min must be < Max")
                return
        else:
            min_val, max_val = 1, 100

        # Distribution-specific parameters
        generator = None

        if distribution == "Uniform":
            generator = UniformGenerator(min_val=min_val, max_val=max_val)

        elif distribution == "Normal":
            c1, c2 = st.columns(2)
            with c1:
                mean = st.number_input("Mean", value=float((min_val + max_val) / 2))
            with c2:
                std = st.number_input("Std Dev", value=float((max_val - min_val) / 6), min_value=0.1)
            generator = NormalGenerator(min_val=min_val, max_val=max_val, mean=mean, std=std)

        elif distribution == "Dice Roll":
            notation = st.text_input("Dice Notation", value="2d6", help="e.g., 2d6, 3d8+5")
            try:
                generator = DiceRollGenerator(notation=notation)
                min_val, max_val = generator.min_val, generator.max_val
                st.caption(f"Range: {min_val} – {max_val}")
            except ValueError as e:
                st.error(str(e))
                return

        elif distribution == "Triangular":
            mode = st.slider("Peak", min_value=float(min_val), max_value=float(max_val),
                           value=float((min_val + max_val) / 2))
            generator = TriangularGenerator(min_val=min_val, max_val=max_val, mode=mode)

        elif distribution == "Exponential":
            scale = st.slider("Scale", min_value=1.0, max_value=float(max_val - min_val),
                            value=float((max_val - min_val) / 3))
            generator = ExponentialGenerator(min_val=min_val, max_val=max_val, scale=scale)

        elif distribution == "Average of N":
            n = st.slider("Samples to Average", min_value=1, max_value=10, value=3)
            generator = AverageOfNGenerator(min_val=min_val, max_val=max_val, n=n)

        # How many
        count = st.number_input("Count", min_value=1, max_value=100, value=1, step=1)

        # Distribution preview (small)
        if generator:
            render_distribution_chart(generator, height=100)

        # Labels section
        with st.expander("Labels", expanded=False):
            label_config = render_label_editor(
                global_min=min_val,
                global_max=max_val,
                key="generator_labels"
            )
            if generator:
                generator.label_config = label_config

    with result_col:
        st.markdown("### Result")

        if generator:
            # Big generate button
            if st.button("🎲  Generate", type="primary", use_container_width=True):
                st.session_state.last_result = generator.generate(count)
                st.session_state.last_count = count

            # Show result
            if st.session_state.last_result:
                render_result_display(st.session_state.last_result, st.session_state.last_count)
            else:
                # Placeholder
                st.markdown(
                    """
                    <div style="
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        min-height: 250px;
                        background: #f8f9fa;
                        border-radius: 20px;
                        border: 2px dashed #dee2e6;
                        color: #adb5bd;
                        font-size: 18px;
                    ">
                        Click Generate to roll
                    </div>
                    """,
                    unsafe_allow_html=True
                )
