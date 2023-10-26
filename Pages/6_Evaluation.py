import streamlit as st

st.title("Expected Final Outcome")
st.write("The following requirements have been tried to satisfy for the optimization model. When we examine the results, we can conclude that the following requirements have been met:")

method_checkbox = st.checkbox("Method to identify the locations of road inspectors to reach the hotspots < 18 mins")
evaluate_checkbox = st.checkbox("Evaluate the method for different scenarios")
explanations_checkbox = st.checkbox("Explanations of the method(s) with respective results")
incident_types_checkbox = st.checkbox("Nice to have: Locations based on incident types")
incident_characteristics_checkbox = st.checkbox("Nice to have: Location based on incident characteristics")

# You can use the values of these checkboxes in your app logic
if method_checkbox:
    st.write("A method has been succesfully established to identify the locations of road inspectors to reach the hotspots < 18 mins")

if evaluate_checkbox:
    st.write("Evaluate the method for different scenarios")

if explanations_checkbox:
    st.write("Explanations of the method(s) with respective results")

if incident_types_checkbox:
    st.write("Nice to have: Locations based on incident types")

if incident_characteristics_checkbox:
    st.write("Nice to have: Location based on incident characteristics")




# Result section
st.header("Result:")
st.write("Automatic inspector placement based on accident probabilities")

# Future Steps section
st.header("Future Steps:")
st.markdown("• Qualitative Testing\n"
            "• Remove network flaws\n"
            "• Simulate to predict Performance", unsafe_allow_html=True)



# Load the images with captions
cover_image = st.image("cover.png", use_column_width=True, caption="100 Inspectors coverage map")
travel_image = st.image("traveld.png", use_column_width=True, caption="Distance distribution for 100 road inspectors")

# Display the images side by side
st.image([cover_image, travel_image], use_container_width=True)
