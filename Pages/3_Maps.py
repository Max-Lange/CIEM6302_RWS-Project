import streamlit as st
import numpy as np
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from datetime import datetime
import data_filtering
import folium
from folium.plugins import HeatMap
import json
import matplotlib.pyplot as plt
from shapely.geometry import Point
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
import seaborn as sns
import streamlit.components.v1 as components

st.markdown("# Maps")
st.sidebar.markdown("# Visualising results using maps")

# Load the shapefile and CSV data with caching
@st.cache_data(experimental_allow_widgets=True, persist="disk")
def load_data():
    highway_shapefile = 'Shapefiles/Snelheid_Wegvakken.shp'
    network_temp = gpd.read_file(highway_shapefile)

    path = 'incidents19Q3Q4.csv'
    df_incident = pd.read_csv(path)
    df_incident['starttime_new'] = pd.to_datetime(df_incident['starttime_new'])
    df_incident['endtime_new'] = pd.to_datetime(df_incident['endtime_new'])
    # Filter the data
    df_incident = data_filtering.filter_out(df_incident, network_temp)
    return df_incident, network_temp

df_incident, network_temp = load_data()

# Streamlit app title

st.write("# Clustering Incidents")

# Read the HTML file and cache the result
@st.cache_data(experimental_allow_widgets=True, persist="disk")
def read_html_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as HtmlFile:
        source_code = HtmlFile.read()
    return source_code

source_code = read_html_file("incidents_map_test.html")

# Display the HTML content in Streamlit
components.html(source_code, width=800, height=600)

@st.cache_data(experimental_allow_widgets=True, persist="disk")
# Function to create a simple density heatmap
def heatmap_simple(_df, _network_temp):
    _network_temp = _network_temp.to_crs("EPSG:4326")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.kdeplot(data=_df, x="primaire_locatie_lengtegraad", y="primaire_locatie_breedtegraad", fill=True, cmap="YlOrRd", thresh=0.5, ax=ax)
    plt.title("Traffic Accident Density Heatmap")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    _network_temp.plot(ax=ax, color='blue', linewidth=0.5, label='network')
    st.pyplot(fig)

# Display the simple density heatmap
st.subheader('Simple Density Heatmap of Traffic Accidents')
st.write('The simple density heatmap shows the density of traffic accidents.')
st.info("The results clearly show three high-incidents areas, the largest of which is the Hague-Rotterdam area, followed by the Amsterdam area and finally the Utrecht area. All three areas show a distinct red color. Then there are three yellow zones, Breda, Eindhoven and Arnhem.")
heatmap_simple(df_incident, network_temp)






