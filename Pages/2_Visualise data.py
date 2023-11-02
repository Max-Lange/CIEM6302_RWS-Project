
import streamlit as st
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from datetime import datetime
import data_filtering
import geopandas as gpd
import folium
from folium.plugins import HeatMap
import json
import pandas as pd
from shapely.geometry import Point
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
import seaborn as sns
from shapely.geometry import Point
import plotly.express as px


# Streamlit app title
st.title('Data Story with Streamlit')
st.info("Please note: Running the code and applying widgets can take some time.")

def main():
    st.title("Data Preprocessing")

    st.write("After an initial exploration of the data, it was clear that some preprocessing and filtering was necessary. For doing so, the following criteria were considered:")
    
    st.markdown("1. Incidents that occurred in roads not included in the NWD Road Network Data.")
    st.markdown("2. Incidents that had a duration of zero minutes or lasted longer than one day.")
    
    st.write("After applying these procedures, a total of 13,172 incidents were removed.")

if __name__ == "__main__":
    main()

st.title("Incident information")
st.write("First, the data is filtered. Subsequently, the results can be plotted, providing valuable information for distributing inspectors.")

# Load the shapefile and CSV data with caching
@st.cache_data(experimental_allow_widgets=True, persist="disk")  # Cache this function for faster data retrieval
def load_data():
    highway_shapefile = './Data/Shapefiles/Snelheid_Wegvakken.shp'
    network_temp = gpd.read_file(highway_shapefile)

    path = './Data/incidents19Q3Q4.csv'
    df_incident = pd.read_csv(path)
    df_incident['starttime_new'] = pd.to_datetime(df_incident['starttime_new'])
    df_incident['endtime_new'] = pd.to_datetime(df_incident['endtime_new'])
    # Filter the data
    df_incident = data_filtering.filter_out(df_incident, network_temp)
    return df_incident

df_incident = load_data()

# Widget for distributiontype
selected_distribution = st.selectbox('Select time distribution:', ['Hourly', 'Weekly', 'Monthly'])

# Widget for type accident
selected_accident_type = st.selectbox('Select type of incident:', df_incident['type'].unique())
@st.cache_data(experimental_allow_widgets=True, persist="disk") 
# Functie voor het weergeven van de distributie op basis van de selectie
def plot_distribution(selected_distribution, selected_accident_type):
    if selected_distribution == 'Hourly':
        distribution_in_day(df_incident, selected_accident_type)
    elif selected_distribution == 'Weekly':
        distribution_in_week(df_incident, selected_accident_type)
    else:
        distribution_in_year(df_incident, selected_accident_type)


@st.cache_data(experimental_allow_widgets=True, persist="disk") 
# Functie voor het weergeven van de hourly distributie (distribution in day)
def distribution_in_day(df_incident, selected_accident_type):
    df_incident['Hour'] = df_incident['starttime_new'].dt.hour

    hourly_counts = df_incident.groupby(['type', 'Hour']).size().unstack(fill_value=0)

    def plot_accidents(accident_type):
        if accident_type == 'all':
            filtered_df = df_incident
        else:
            filtered_df = df_incident[df_incident['type'] == accident_type]

        hourly_counts = filtered_df['Hour'].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(hourly_counts.index, hourly_counts.values)
        ax.set_title(f'Hourly Distribution of {accident_type} Incidents')
        ax.set_xlabel('Hour')
        ax.set_ylabel('Number of Incidents')
        ax.set_xticks(range(24))
        st.pyplot(fig)

    plot_accidents(selected_accident_type)


@st.cache_data(experimental_allow_widgets=True, persist="disk") 
# Function for weekly distribution (distribution in week)
def distribution_in_week(df_incident, selected_accident_type):
    df_incident['Day_of_Week'] = df_incident['starttime_new'].dt.day_name()

    def plot_accidents(accident_type):
        if accident_type == 'all':
            filtered_df = df_incident
        else:
            filtered_df = df_incident[df_incident['type'] == accident_type]

        daily_counts = filtered_df['Day_of_Week'].value_counts().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(daily_counts.index, daily_counts.values)
        ax.set_title(f'Daily Distribution of {accident_type} Incidents')
        ax.set_xlabel('Day of the Week')
        ax.set_ylabel('Number of Incidents')
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)

    plot_accidents(selected_accident_type)


@st.cache_data(experimental_allow_widgets=True, persist="disk") 
# Function for distribution in year
def distribution_in_year(df_incident, selected_accident_type):
    df_incident['Month'] = df_incident['starttime_new'].dt.month

    def plot_monthly_distribution(accident_type):
        if accident_type == 'all':
            filtered_df = df_incident
        else:
            filtered_df = df_incident[df_incident['type'] == accident_type]

        monthly_counts = filtered_df['Month'].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(monthly_counts.index, monthly_counts.values)
        ax.set_title(f'Monthly Distribution of {accident_type} Incidents')
        ax.set_xlabel('Month')
        ax.set_ylabel('Number of Incidents')
        ax.set_xticks(range(1, 13))
        st.pyplot(fig)

    plot_monthly_distribution(selected_accident_type)

# plot
plot_distribution(selected_distribution, selected_accident_type)


# Define the list of highways
highways = ['A1', 'A2', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A12', 'A13', 'A15', 'A16', 'A17', 'A18', 'A20', 'A22',
           'A27', 'A28', 'A29', 'A30', 'A31', 'A32', 'A35', 'A37', 'A38', 'A44', 'A50', 'A58', 'A59', 'A65', 'A67', 'A73',
           'A74', 'A76', 'A77', 'A79', 'A200', 'A208', 'A256', 'A270', 'A325', 'A326', 'A348', 'A783']

# Load your data or use sample data
# df_incident = pd.read_csv('your_data.csv')

# Count accidents on each highway
count = np.zeros(len(highways))
for i in range(len(df_incident['vild_primair_wegnummer'])):
    for j in range(len(count)):
        if df_incident['vild_primair_wegnummer'].iloc[i] == highways[j]:
            count[j] += 1

# Create a DataFrame to store the data
data_locations_df_accidents = {'Highway': highways, 'Amount of Accidents': count}
locations_df_accidents = pd.DataFrame(data=data_locations_df_accidents)


# Display a bar chart of accidents on highways
st.subheader('Location of Accidents')

st.write("You can see on which roads the most accidents occur (during Q3&Q4 of 2019) in the figure below.")

# Create a Matplotlib figure and axis

import plotly.figure_factory as ff



# Create a Plotly bar chart
fig = px.bar(
    locations_df_accidents,
    x=locations_df_accidents['Highway'],
    y= locations_df_accidents['Amount of Accidents'],
    title='Location of Accidents',
    labels={'Amount of Accidents': 'Amount of Accidents', 'Highway': 'Highways'}
)

st.plotly_chart(fig, use_container_width=True)



# Display the data table
st.subheader('Accident Data by Highway descending')
# Reorder locations_df_accidents based on the number of accidents in descending order
sorted_locations_df_accidents = locations_df_accidents.sort_values(by='Amount of Accidents', ascending=False)

# Display table
st.dataframe(sorted_locations_df_accidents, use_container_width=True)

st.subheader("Duration of incidents")
st.write("The duration of the different types of incidents can be seen below. Chose the incident type and click on generate plot. ")

selected_accident_type = st.selectbox('Select type of incident:', ['vehicle_obstruction', 'general_obstruction','accident', 'all'])

def plot_distribution(selected_accident_type):
    if selected_accident_type == 'vehicle_obstruction':
        distribution_duration(df_incident, selected_accident_type)
    elif selected_accident_type == 'general_obstruction':
        distribution_duration(df_incident, selected_accident_type)
    elif selected_accident_type == 'accident':
        distribution_duration(df_incident, selected_accident_type)
    elif selected_accident_type == 'all':
        distribution_duration(df_incident, selected_accident_type)



def distribution_duration(df_incident, selected_accident_type):
    df_incident['starttime_new'] = pd.to_datetime(df_incident['starttime_new'])
    df_incident['endtime_new'] = pd.to_datetime(df_incident['endtime_new'])

    df_incident['Duration_Minutes'] = (df_incident['endtime_new'] - df_incident['starttime_new']).dt.total_seconds() / 60
    df_incident = df_incident[(df_incident['Duration_Minutes'] <= 500) & (df_incident['Duration_Minutes'] > 0)]

    def plot_duration_distribution(accident_type):
        if accident_type == 'all':
            filtered_df = df_incident
        else:
            filtered_df = df_incident[df_incident['type'] == accident_type]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(filtered_df['Duration_Minutes'], bins=20, edgecolor='k')
        ax.set_title(f'Duration Distribution of {accident_type} Incidents')
        ax.set_xlabel('Duration (Minutes)')
        ax.set_ylabel('Number of Incidents')
        st.pyplot(fig)

    accident_types = ['all'] + list(df_incident['type'].unique())

    #dropdown = widgets.Dropdown(options=accident_types, description='Incident Type:')
    #selected_accident_type = st.selectbox('Select the incident type:', accident_types)

    if st.button('Generate Plot'):
        plot_duration_distribution(selected_accident_type)

plot_distribution(selected_accident_type)


#PIE CHART

data = {
    'Incident Type': ['Vehicle Obstruction', 'General Obstruction', 'Accident'],
    'Percentage': [80.9, 6.1, 13.0]
}


df = pd.DataFrame(data)

def main():
    st.subheader("Relative incident type distribution")

    # Table
    st.dataframe(df)

    # Piechart Plotly
    st.write("Pie Chart of Incident Types:")
    fig = px.pie(df, names='Incident Type', values='Percentage')
    st.plotly_chart(fig)

if __name__ == "__main__":
    main()

