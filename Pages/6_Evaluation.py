import streamlit as st

# Result section
st.header("Evaluation")
st.write("In this paragraph we will have a look at the final result with respect to the main goal of the research and validate them")
st.write("We mainly used two different methods to validate the results. One is to calculate the coverage distance of all inspectors, and the other is to obtain the distance between each incident and the inspector through the test set. The distribution result of 100 inspectors based on all incidents is our verification object. If you want to know more about its operation principle and result analysis, please read our final notebook.")




st.subheader("Covering Area:")
st.write("The figure below demonstrates that the vast majority of the country is accessible to inspectors with a travel time of up to 14 minutes.")
st.write("We calculated the covered length, coverage ratio, redundant length and redundant ratio. These indicators can be used as meaningful reference indicators. And we have succeeded in achieving automatic inspector placement based on accident probabilities.")

# Load the cover image with caption
cover_image = st.image("cover.png", use_column_width=True, caption="100 Inspectors coverage map")



st.subheader("Average Distance to Incidents")
st.write("The following requirements have been tried to satisfy for the optimization model. When we examine the results, we can conclude that the following requirements have been met:")

st.markdown("#### Based on Sampled Test Set")
st.write("We generated the number of possible daily occurrences of incidnets through the lognormal distribution, and then generated their occurrence locations through previous KDE results. The image below shows the location of the dataset we generated.")

st.image('Images/sample_test_set.png', use_column_width=True, caption='Sampled Incidnets')

st.write("The code draws a histogram of 'Travel distance distribution', showing the distance distribution from the incident point to the nearest inspector point. The average distance is also marked in the figure.")
st.image('Images/sampled_distance_distribution.png', use_column_width=True, caption='Distance Distribution (sampled test data)')

st.markdown("#### Based on Real Test Set")
st.write("We randomly selected the accident on 1-10-2019 as the test set, with a total of 1021 pieces of data.")
st.write("Its validation result is shown below")
st.image('Images/real_distance_distribution.png', use_column_width=True, caption='Distance Distribution (real test data)')


st.write("The results on both test sets show that our results can respond quickly to incidents.")

st.markdown("#### Samll Tips")
st.write("If you are not satisfied of our current data set, you can use the code in our final notebook to generate the test set randomly or choose real incidents of anytime and any type by data filtering. Our Streamlit is temporarily unable to provide this service because calculating the distance from 1000 incidents to inspectors may take up to 10 minutes.")

# method_checkbox = st.checkbox("Method to identify the locations of road inspectors to reach the hotspots < 18 mins", value=True)
# evaluate_checkbox = st.checkbox("Evaluate the method for different scenarios", value=True)
# explanations_checkbox = st.checkbox("Explanations of the method(s) with respective results")
# incident_types_checkbox = st.checkbox("Nice to have: Locations based on incident types")
# incident_characteristics_checkbox = st.checkbox("Nice to have: Location based on incident characteristics")

# # You can use the values of these checkboxes in your app logic
# if method_checkbox:
#     st.write("- A method has been succesfully established to identify the locations of road inspectors to reach the hotspots < 18 mins!")

# if evaluate_checkbox:
#     st.write("- The developed algorithm is successfully capable of evaluating the method for different scenarios, including the number of road inspectors, considering various data, time frames, and types of incidents. Based on these scenarios, an inspector distribution can be determined.")

# if explanations_checkbox:
#     st.write("Explanations of the method(s) with respective results")

# if incident_types_checkbox:
#     st.write("Nice to have: Locations based on incident types")

# if incident_characteristics_checkbox:
#     st.write("Nice to have: Location based on incident characteristics")


# # Future Steps section
# st.subheader("Future Steps:")
# st.write("We have successfully identified an inspector distribution. The next steps for Rijkswaterstaat to conduct are:")

# st.markdown("• Qualitative Testing")
# st.markdown("• Remove network flaws")
# st.markdown("• Simulate to predict Performance")