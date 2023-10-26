import streamlit as st
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
from shapely.geometry import Point

# Streamlit app title
st.title('Spatio-Temporal Visualization of Incidents')
st.info("Please note: Running the code and applying widgets can take some time.")
st.write("The purpose of this page is mainly to analyze the spatiotemporal distribution of various types of incidents. It can visualize the specific location and type of incidents in the road network that occurred in each time window.")

# Load the shapefile and CSV data
@st.cache_data(experimental_allow_widgets=True) 
def load_data():
    highway_shapefile = './Data/Shapefiles/Snelheid_Wegvakken.shp'
    network_temp = gpd.read_file(highway_shapefile)

    path = './Data/incidents19Q3Q4.csv'
    df_incident = pd.read_csv(path)
    df_incident['starttime_new'] = pd.to_datetime(df_incident['starttime_new'])

    return df_incident, network_temp

df_incident, network_temp = load_data()

# Define a Streamlit slider for selecting the time window
timestep = st.slider('Select Time Window (minutes):', min_value=1, max_value=60, value=15)

# Transform GIS
network_temp = network_temp.to_crs("EPSG:4326")

# Convert starttime_new column to datetime object
df_incident['starttime_new'] = pd.to_datetime(df_incident['starttime_new'])

# Calculate the time window to which each time point belongs
time_interval = pd.Timedelta(minutes=timestep)
df_incident['time_window_start'] = df_incident['starttime_new'] - ((df_incident['starttime_new'] - df_incident['starttime_new'].min()) % time_interval)

# List of unique time_window_start values
time_window_options = df_incident['time_window_start'].unique()
time_window_options = sorted(time_window_options)

# Streamlit slider for selecting the time window using datetime
# Streamlit slider for selecting the time window using datetime
selected_time = st.slider('Select Time Window:', 
                          min_value=df_incident['time_window_start'].min().to_pydatetime(), 
                          max_value=df_incident['time_window_start'].max().to_pydatetime())
# Filter data based on selected time
selected_data = df_incident[df_incident['time_window_start'] == selected_time]

# Create a map
fig, ax = plt.subplots(figsize=(10, 8))
network_temp.plot(ax=ax, color='blue', linewidth=0.5, label='Network')

# Define colors for different accident types
colors = {'general_obstruction': 'red', 'vehicle_obstruction': 'green', 'accident': 'm'}

for acc_type, acc_data in selected_data.groupby('type'):
    plt.scatter(
        acc_data['primaire_locatie_lengtegraad'],
        acc_data['primaire_locatie_breedtegraad'],
        marker='x',
        s=50,
        color=colors.get(acc_type, 'black'),
        label=acc_type
    )

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title(f'Distribution of Incidents ({selected_time})')
plt.grid(True)
plt.legend()

# Display the Matplotlib figure using st.pyplot()
st.pyplot(fig)
