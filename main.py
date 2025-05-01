import streamlit as st

# Initialize session state for mode tracking
if 'current_mode' not in st.session_state:
    st.session_state.current_mode = "normal"  # Default to normal mode

# Mode selection
if st.session_state.current_mode == "normal":
    from normalmode import main as normal_main
    normal_main()
else:
    from kidsmode import main as kids_main
    kids_main()