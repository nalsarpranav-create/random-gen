"""
Random Number Generator with Custom Probability Patterns

A Streamlit web app for generating random numbers with customizable
probability distributions and label mappings.
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from ui.generator_tab import render_generator_tab
from ui.curve_editor import render_curve_editor_tab
from ui.profile_tab import render_profile_tab

# Page configuration
st.set_page_config(
    page_title="Random Generator",
    page_icon="🎲",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .main { padding: 1rem 2rem; }
    h1, h2, h3 { color: #1f1f1f; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px 8px 0 0;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """Main application entry point."""
    st.title("🎲 Random Generator")

    # Tabs for different modes
    tab1, tab2, tab3 = st.tabs([
        "**Simple**",
        "**Custom Curve**",
        "**Profile Builder**"
    ])

    with tab1:
        render_generator_tab()

    with tab2:
        render_curve_editor_tab()

    with tab3:
        render_profile_tab()


if __name__ == "__main__":
    main()
