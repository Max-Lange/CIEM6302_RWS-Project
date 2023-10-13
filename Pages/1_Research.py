import streamlit as st
# Lay-out 


# Page Title
st.title("Research Page 📚")
st.sidebar.title("Research Page 📚")

# Research Objective and Research Questions
st.header("Research Objective and Research Questions")

# Research Objective
st.subheader("Research Objective:")
st.write("The research objective is to determine the optimal locations for road inspectors within the Dutch road network, "
         "with a primary aim to minimize travel time to incidents. This research demonstrates an excellent ability to "
         "extract meaningful insights from the available data.")


#Main-question
st.subheader("Main research question:")
st.write("**What would be the optimal location of road inspectors in the Netherlands, "
         "such that travel times to incidents are minimized?**")

# Sub-questions
st.subheader("Sub-questions:")
st.write("1. To what extent is the data provided by Rijkswaterstaat useful for achieving the research objective?")
st.write("2. How are incidents distributed in both time and space across the Dutch road network?")
st.write("3. Which methodology is most suitable for evaluating accident probabilities effectively?")
st.write("4. What are the specific locations with a higher likelihood of accidents occurring?")
st.write("5. Which variables significantly impact the response time of road inspectors?")
st.write("6. What methods can be employed to calculate the response time of inspectors accurately?")
st.write("7. What is the optimal number of road inspectors required to meet the research objective?")
st.write("8. Where should these road inspectors be strategically located to achieve optimal response times?")




# Create a container div to hold the images in two columns
st.write('<div style="display: flex; flex-direction: row;">', unsafe_allow_html=True)

# First image on the left
st.image('tud.png', use_column_width=False, width=150, caption='Written by students of TU Delft')

# Second image on the right
st.image('rws.jpg', use_column_width=False, width=150, caption='Client: Rijkswaterstaat')

# Close the container div
st.write('</div>', unsafe_allow_html=True)
