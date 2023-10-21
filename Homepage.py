import streamlit as st

# Page Title
st.markdown("# Home Page 🏠")
st.sidebar.markdown("# Home Page 🏠")

# Textual Information
st.header("Welcome to the Home Page")
st.write("This is the homepage of our application.")
st.write("You can explore data visualizations of our project within this application.")

# Additional Information as a single paragraph
additional_info = (
    "To ensure safety on highways and a smooth traffic flow, road inspectors are important. "
    "When accidents occur, they appear and make sure that the traffic can quickly resume. "
    "It is important for these people to show up as soon as possible when accidents happen. "
    "So, an optimal distribution of the inspectors to make sure that they will arrive shortly after the accident is necessary."
)

st.subheader("Introduction")
st.write(additional_info)


st.image("./Images/accident.png", caption="Accident on a highway (Stuff, 2022)", use_column_width=True)










