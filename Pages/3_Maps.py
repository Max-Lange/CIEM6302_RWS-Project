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
st.write("Information about maps can be found in this page")
st.write("The first one corresponds to an interactive clustering map showing the incident location.")
st.info("When the map is zoomed in or out, points within the view will be reclustered. At the same time, click on the cluster and it will be divided into smaller clusters until it becomes a point. In the results, when the view is zoomed in to the whole country, all incident points are divided into 8 clusters. Among them, the cluster in the Amsterdam-The Hague-Rotterdam area contains the vast majority of incident points, with as many as 49,732, more than half of the total. The cluster in the Eindhoven area has the second most points, with a total of 11,578. Finally, clusters in the northeast and southwest regions of the country contain relatively few points.")
st.sidebar.markdown("# Visualising results using maps")

# Load the shapefile and CSV data with caching
@st.cache_data(experimental_allow_widgets=True, persist="disk")
def load_data():
    highway_shapefile = 'Data/Shapefiles/Snelheid_Wegvakken.shp'
    network_temp = gpd.read_file(highway_shapefile)
    NL_map = gpd.read_file('Netherlands SHP/NL_Province/NLProvince.SHP')

    path = 'Data/incidents19Q3Q4.csv'
    df_incident = pd.read_csv(path)
    df_incident['starttime_new'] = pd.to_datetime(df_incident['starttime_new'])
    df_incident['endtime_new'] = pd.to_datetime(df_incident['endtime_new'])
    # Filter the data
    df_incident = data_filtering.filter_out(df_incident, network_temp)
    return df_incident, network_temp, NL_map

df_incident, network_temp, NL_map = load_data()

# Streamlit app title

st.subheader("Clustering Incidents")

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
    NL_map.plot(ax = ax, facecolor = "None")
    st.pyplot(fig)

# Display the simple density heatmap
st.subheader('Simple Density Heatmap of Traffic Accidents')
st.write('The simple density heatmap shows the density of traffic accidents.')
st.info("The results clearly show three high-incidents areas, the largest of which is the Hague-Rotterdam area, followed by the Amsterdam area and finally the Utrecht area. All three areas show a distinct red color. Then there are three yellow zones, Breda, Eindhoven and Arnhem.")
heatmap_simple(df_incident, network_temp)

st.subheader("Kernel density estimation.")
st.write("The kernel density estimation helps us to convert discrete incident data points into incident probability densities on the map.")
st.info("As can be seen, and corresponding with the findings from previous maps, the segments with the highest incident density are located in the Ranstad Area.")

st.image("./Images/pmap.png", caption="Segment Probability Map")




