import streamlit as st
from PIL import Image
import graphviz
import geopandas as gpd
import matplotlib.pyplot as plt
from datetime import datetime
import data_filtering
import folium
from folium.plugins import HeatMap
import json
import pandas as pd
from shapely.geometry import Point
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
import seaborn as sns
from scipy.spatial import Voronoi, voronoi_plot_2d
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
from shapely import Point, LineString, Polygon
import node_probability
from projection_conversions import DutchRDtoWGS84, WGS84toDutchRD


st.title("Algorithm - Variable Data")

st.write("In this window, it is possible for the user to create a subset of the input dataset according to multiple criteria: start and end date, start and end time, day of the week, and incident type. It was decided to use a separated window as the filtering process modifies the location of the selected incidents and thus the spatial probability density function. As such, please note that the running time of this page can be lengthy as the whole algorithm needs to be run from scratch at every change.")


# Load the shapefile and CSV data with caching
def load_data():
    highway_shapefile = 'Data/Shapefiles/Snelheid_Wegvakken.shp'
    network_temp = gpd.read_file(highway_shapefile)

    path = 'Data/incidents19Q3Q4.csv'
    df_incident = pd.read_csv(path)
    df_incident['starttime_new'] = pd.to_datetime(df_incident['starttime_new'])
    df_incident['endtime_new'] = pd.to_datetime(df_incident['endtime_new'])

    # Filter the data
    df_incident = data_filtering.filter_out(df_incident, network_temp)

    # Add new columns 'date' and 'time'
    df_incident['date'] = df_incident['starttime_new'].dt.date
    df_incident['time'] = df_incident['starttime_new'].dt.time

    return df_incident

df_incident = load_data()

# Assuming you have the 'df_incident' DataFrame

# Calculate default start and end dates based on the data
default_start_date = df_incident['starttime_new'].dt.date.min()
default_end_date = df_incident['starttime_new'].dt.date.max()

# Calculate default start and end times based on the data
default_start_time = df_incident['starttime_new'].dt.time.min()
default_end_time = df_incident['starttime_new'].dt.time.max()

# Create Streamlit widgets for user input
selected_date_range = st.date_input(
    'Select a date range',
    min_value=default_start_date,
    max_value=default_end_date,
    value=(default_start_date, default_end_date),
    key="date_range_key"
)

selected_start_time = st.time_input('Select a start time', value=default_start_time, key="start_time_key")
selected_end_time = st.time_input('Select an end time', value=default_end_time, key="end_time_key")

# Create checkboxes for selecting specific days of the week
selected_days = st.multiselect(
    'Select specific days of the week',
    ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
)

# Create a dropdown widget to select the incident type
incident_types = df_incident['type'].unique()
selected_accident_type = st.selectbox('Select type of incident:', ['all'] + list(incident_types))

# Filter the data based on user selections
filtered_data = df_incident.copy()  # Create a copy to avoid modifying the original data

if selected_date_range:
    start_date, end_date = selected_date_range
    filtered_data = filtered_data[(filtered_data['starttime_new'].dt.date >= start_date) & (filtered_data['starttime_new'].dt.date <= end_date)]

if selected_start_time and selected_end_time:
    start_time = selected_start_time
    end_time = selected_end_time
    filtered_data = filtered_data[(filtered_data['starttime_new'].dt.time >= start_time) & (filtered_data['starttime_new'].dt.time <= end_time)]

if selected_days:
    selected_day_names = [day for day in selected_days]
    filtered_data = filtered_data[filtered_data['starttime_new'].dt.day_name().isin(selected_day_names)]

# Filter by incident type
if selected_accident_type != 'all':
    filtered_data = filtered_data[filtered_data['type'] == selected_accident_type]

# Display the filtered data
st.dataframe(filtered_data)


highway_shapefile = 'Data/Shapefiles/Snelheid_Wegvakken.shp'
road_network = gpd.read_file(highway_shapefile)
road_network = road_network.to_crs("EPSG:4326")


df_incident = filtered_data
# Create points_gdf
points_gdf = node_probability.create_points_gdf(road_network, df_incident)

# Create edges_gdf
road_network = gpd.read_file(highway_shapefile)
edges = {'from': [], 'to': [], 'type': [], 'direction': [], 'name': [], 'geometry': []}

for _, row in road_network.iterrows():
    edges['from'].append(row['JTE_ID_BEG'])
    edges['to'].append(row['JTE_ID_END'])
    edges['type'].append(row['BST_CODE'])
    edges['direction'].append(row['RIJRICHTNG'])
    edges['name'].append(row['STT_NAAM'])
    edges['geometry'].append(row['geometry'])

edges_gdf = gpd.GeoDataFrame(edges)

st.write("v3")

# Use point_gdf and edges_gdf to construct the network
NWB_Graph = nx.MultiDiGraph()

for _, point in points_gdf.iterrows():
    NWB_Graph.add_node(point['name'], pos=point['geometry'].coords[0], probability=point['probability'])

already_connected = []
crossroads_id = 0
for index, edge in edges_gdf.iterrows():
    if edge['direction'] in ['H', 'B', 'O']:
        NWB_Graph.add_edge(edge['from'], edge['to'], geom=edge['geometry'], weight=edge['geometry'].length)
    
    if edge['direction'] in ['T', 'B', 'O']:
        NWB_Graph.add_edge(edge['to'], edge['from'], geom=edge['geometry'], weight=edge['geometry'].length)
    
    # Add a new exit node if edge is an exit ramp
    if edge['type'] in ['AFR', 'OPR'] and index not in already_connected:
        to_connect = edges_gdf[edges_gdf['name'] == edge['name']]
        to_connect = to_connect[to_connect['type'].isin(['AFR', 'OPR', 'PST'])]

        # Add an exit node in the middle of all exit sections
        x = []
        y = []
        crossroads_id += 1
        for _, i in to_connect.iterrows():
            x.append(i['geometry'].coords[-1][0])
            y.append(i['geometry'].coords[-1][1])
        x = np.sum(x) / len(to_connect)
        y = np.sum(y) / len(to_connect)
        NWB_Graph.add_node(crossroads_id, pos=(x, y), probability=0)

        # Connect all to nodes to new exit node
        for index, i in to_connect.iterrows():
            already_connected.append(index)
            # TODO: For now we assume that all ramps have direction 'H', need to check if this is a garuntee, if not extra code is needed
            NWB_Graph.add_edge(i['to'], crossroads_id, geom=None, weight=0)
            NWB_Graph.add_edge(crossroads_id, i['to'], geom=None, weight=0)
            NWB_Graph.add_edge(i['from'], crossroads_id, geom=None, weight=0)
            NWB_Graph.add_edge(crossroads_id, i['from'], geom=None, weight=0)


# Allocate all nodes to a single exit
nx.set_node_attributes(NWB_Graph, None, "closest_exit")
nx.set_node_attributes(NWB_Graph, 10000000000, "closest_exit_dist")

for exit_node in range(1, crossroads_id+1):
    distances = nx.single_source_dijkstra_path_length(NWB_Graph, exit_node)
    for node, dist in distances.items():
        if dist < NWB_Graph.nodes[node]['closest_exit_dist']:
            NWB_Graph.nodes[node]['closest_exit'] = exit_node
            NWB_Graph.nodes[node]['closest_exit_dist'] = dist
# TODO: What to do with the probability of the nodes who cannot be connected to an exit, use as the crow flies dist?
# Calculate total probability for the exit
for node in NWB_Graph.nodes:
    if NWB_Graph.nodes[node]['closest_exit'] is not None:
        NWB_Graph.nodes[NWB_Graph.nodes[node]['closest_exit']]['probability'] += NWB_Graph.nodes[node]['probability']



#@st.cache_data(experimental_allow_widgets=True, persist="disk")
def no_exit_nodes_plot(total_graph):
    fig, ax = plt.subplots(figsize=(8, 8))

    for edge in total_graph.edges.data():
        line = edge[2]['geom']
        if line is not None:
            x, y = line.coords.xy
            ax.plot(x, y, color='k')
    
    count = 0
    for node in total_graph.nodes():
        if total_graph.nodes[node]['closest_exit'] is None:
            position = total_graph.nodes[node]['pos']
            ax.plot(position[0], position[1], '.', color='r')
            count += 1

    ax.set_title("Network")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_aspect('equal')

    # Display the plot within the Streamlit app
    st.pyplot(fig)
    return count


#st.write(no_exit_nodes_plot(NWB_Graph))

#@st.cache_data(experimental_allow_widgets=True, persist="disk")
def algorithm_v3(total_graph, exit_id_max, desired_inspector_nr):
    desired_inspector_nr += 10
    inspector_placements = dict()

    for exit_node in range(1, exit_id_max + 1):
        inspector_placements[exit_node] = [total_graph.nodes[exit_node]['probability'], [exit_node]]

    # Main Loop
    removed = []
    while len(inspector_placements) > desired_inspector_nr:
        # Determine inspector network with lowest probability
        lowest_prob_inspector = None
        lowest_prob = np.inf
        for exit_node, inspector_data in inspector_placements.items():
            if inspector_data[0] < lowest_prob:
                lowest_prob_inspector = exit_node
                lowest_prob = inspector_data[0]

        print(len(inspector_placements), lowest_prob_inspector, lowest_prob)

        # Determine closest other inspector to lowest probability inspector
        closest_exit = None
        lowest_distance = np.inf
        for inspector_node in inspector_placements[lowest_prob_inspector][1]:
            distances = nx.single_source_dijkstra_path_length(total_graph, inspector_node)
            for node, distance in distances.items():
                if node <= exit_id_max and node not in inspector_placements[lowest_prob_inspector][1] and node not in removed:
                    if distance < lowest_distance:
                        closest_exit = node
                        lowest_distance = distance

        if closest_exit is None:
            # Special clause for if no neighbors can be found
            inspector_placements[lowest_prob_inspector][0] = np.inf
        else:
            for inspector, inspector_data in inspector_placements.items():
                if closest_exit in inspector_data[1]:
                    closest_inspector = inspector

            if closest_inspector is not None:
                # Combine inspector networks into lowest_prob_inspector and delete closest_inspector
                inspector_placements[lowest_prob_inspector][0] += inspector_placements[closest_inspector][0]
                for exit in inspector_placements[closest_inspector][1]:
                    inspector_placements[lowest_prob_inspector][1].append(exit)
                del inspector_placements[closest_inspector]
                removed.append(closest_inspector)

    return inspector_placements






# Add an input field for the number of road inspectors
inspector_count = st.number_input("Number of (useful) road inspectors (integer):", min_value=1, step=1)


placements = algorithm_v3(NWB_Graph, crossroads_id, inspector_count)



#@st.cache_data(experimental_allow_widgets=True, persist="disk")
def improve_inspector_placements(total_graph: nx.MultiDiGraph, inspector_placements: dict):
    new_placements = dict()
    for _, inspector_data in inspector_placements.items():
        best_exit_placement = None
        best_distance_sum = np.inf
        for exit in inspector_data[1]:
            count = 0
            distance_sum = 0
    	    # TODO: Add a check for if all nodes are reachable by an exit, otherwise take it out of consideration
            distances = nx.single_source_dijkstra_path_length(total_graph, exit)
            for node, distance in distances.items():
                if total_graph.nodes[node]['closest_exit'] in inspector_data[1]:
                    distance_sum += distance
                    count += 1

            if distance_sum < best_distance_sum:
                best_exit_placement = exit
                best_distance_sum = distance_sum

        avg_travel_time = (best_distance_sum / count) / 27.778  # 27.778 m/s == 100 km/h
        new_placements[best_exit_placement] = [inspector_data[0], inspector_data[1], avg_travel_time, best_distance_sum, count]

    return new_placements


new_placements = improve_inspector_placements(NWB_Graph, placements)


#@st.cache_data(experimental_allow_widgets=True, persist="disk")

def inspector_plotter(total_graph, inspector_placements):
    # Create a figure
    fig, ax = plt.subplots(figsize=(8, 8))

    for edge in total_graph.edges.data():
        line = edge[2]['geom']
        if line is not None:
            x, y = line.coords.xy
            ax.plot(x, y, color='k')

    for node in total_graph.nodes():
        if node in inspector_placements:
            position = total_graph.nodes[node]['pos']
            ax.plot(position[0], position[1], '.', color='r')

    ax.set_title("Network")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_aspect('equal')

    # Display the plot within the Streamlit app
    st.pyplot(fig)

# Streamlit app
st.title("Inspector Plotter")
st.write("Graph with inspectors:")

# Define inspector placements (replace with your data)
new_placements = improve_inspector_placements(NWB_Graph, placements)

# Create a button to generate the plot
#if st.button("Generate Plot"):
st.write("Graph with inspectors:")
inspector_plotter(NWB_Graph, new_placements)
