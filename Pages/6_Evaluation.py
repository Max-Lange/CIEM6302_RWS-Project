import streamlit as st

# Result section
st.header("Evaluation")
st.write("In this paragraph we will have a look at the final result with respect to the main goal of the research")

st.subheader("Result:")
st.write("The figure below demonstrates that the vast majority of the country is accessible to inspectors with a travel time of up to 15 minutes.")
st.write("We have succeeded in achieving automatic inspector placement based on accident probabilities.")

# Load the cover image with caption
cover_image = st.image("cover.png", use_column_width=True, caption="100 Inspectors coverage map")




st.subheader("Expected Final Outcome")
st.write("The following requirements have been tried to satisfy for the optimization model. When we examine the results, we can conclude that the following requirements have been met:")


method_checkbox = st.checkbox("Method to identify the locations of road inspectors to reach the hotspots < 18 mins", value=True)
evaluate_checkbox = st.checkbox("Evaluate the method for different scenarios", value=True)
explanations_checkbox = st.checkbox("Explanations of the method(s) with respective results")
incident_types_checkbox = st.checkbox("Nice to have: Locations based on incident types")
incident_characteristics_checkbox = st.checkbox("Nice to have: Location based on incident characteristics")

# You can use the values of these checkboxes in your app logic
if method_checkbox:
    st.write("- A method has been succesfully established to identify the locations of road inspectors to reach the hotspots < 18 mins!")

if evaluate_checkbox:
    st.write("- The developed algorithm is successfully capable of evaluating the method for different scenarios, including the number of road inspectors, considering various data, time frames, and types of incidents. Based on these scenarios, an inspector distribution can be determined.")

if explanations_checkbox:
    st.write("Explanations of the method(s) with respective results")

if incident_types_checkbox:
    st.write("Nice to have: Locations based on incident types")

if incident_characteristics_checkbox:
    st.write("Nice to have: Location based on incident characteristics")


# Future Steps section
st.subheader("Future Steps:")
st.write("We have successfully identified an inspector distribution. The next steps for Rijkswaterstaat to conduct are:")

st.markdown("• Qualitative Testing")
st.markdown("• Remove network flaws")
st.markdown("• Simulate to predict Performance")