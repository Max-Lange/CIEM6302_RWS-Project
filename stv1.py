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
import folium
from scipy.spatial import Voronoi, voronoi_plot_2d
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
from shapely import Point, LineString, Polygon

from projection_conversions import DutchRDtoWGS84, WGS84toDutchRD

# Streamlit app title
st.title('Running algorithm v1')
st.info("Please note: Running the code and applying widgets can take some time.")
st.write("Version 1 of algorithm, select specific data to see the effect on the inspectors distribution.")

# Load the shapefile and CSV data with caching

 

# Load the shapefile and CSV data with caching
#@st.cache_data(experimental_allow_widgets=True, persist="disk")
def load_data():
    highway_shapefile = 'Shapefiles/Snelheid_Wegvakken.shp'
    network_temp = gpd.read_file(highway_shapefile)

    path = 'incidents19Q3Q4.csv'
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
    key="date_range_key"  # Set a unique key to ensure widget state consistency
)

selected_start_time = st.time_input('Select a start time', value=default_start_time, key="start_time_key")
selected_end_time = st.time_input('Select an end time', value=default_end_time, key="end_time_key")

# Create checkboxes for selecting specific days of the week
selected_days = st.multiselect(
    'Select specific days of the week',
    ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
)

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

# Display the filtered data
#st.dataframe(filtered_data)

#PART 2

road_network = gpd.read_file(r'Data\Shapefiles\Snelheid_Wegvakken.shp')

# Convert data to node and edge GeoDataFrames containing only needed info
points = {'name': [], 'geometry': []}
edges = {'from': [], 'to': [], 'type': [], 'direction': [], 'name': [], 'geometry': []}

for _, row in road_network.iterrows():
    if row['JTE_ID_BEG'] not in points['name']:
        points['name'].append(row['JTE_ID_BEG'])
        points['geometry'].append(Point(row['geometry'].coords[0]))
    
    if row['JTE_ID_END'] not in points['name']:
        points['name'].append(row['JTE_ID_END'])
        points['geometry'].append(Point(row['geometry'].coords[-1]))
    
    edges['from'].append(row['JTE_ID_BEG'])
    edges['to'].append(row['JTE_ID_END'])
    edges['type'].append(row['BST_CODE'])
    edges['direction'].append(row['RIJRICHTNG'])
    edges['name'].append(row['STT_NAAM'])
    edges['geometry'].append(row['geometry'])

points_gdf = gpd.GeoDataFrame(points)
edges_gdf = gpd.GeoDataFrame(edges)

#2

accident_data = filtered_data
st.dataframe(accident_data)
st.write(len(accident_data))

locations = []
for _, accident in accident_data.iterrows():
    location = Point(WGS84toDutchRD(accident['primaire_locatie_lengtegraad'],
                                    accident['primaire_locatie_breedtegraad']))
    locations.append(location)

accident_data["geometry"] = locations
accident_gdf = gpd.GeoDataFrame(accident_data, geometry=accident_data['geometry'])

total_amount = 5_000
points_gdf['accident_count'] = [0] * len(points_gdf)

for index, accident in accident_gdf[:total_amount].iterrows():
    if index % 500 == 0:
        print(index)

    distances = points_gdf.distance(accident['geometry'])
    index_i = distances.idxmin()
    points_gdf.loc[index_i, 'accident_count'] += 1

points_gdf['probability'] = points_gdf['accident_count'] / total_amount
points_gdf

NWB_Graph = nx.Graph()

for _, point in points_gdf.iterrows():
    NWB_Graph.add_node(point['name'], pos=point['geometry'].coords[0], probability=point['probability'])

already_connected = []
crossroads_id = 1
for index, edge in edges_gdf.iterrows():
    NWB_Graph.add_edge(edge['from'], edge['to'], geom=edge['geometry'], weight=edge['geometry'].length)

    if edge['type'] in ['AFR', 'OPR'] and index not in already_connected:
        to_connect = edges_gdf[edges_gdf['name'] == edge['name']]
        to_connect = to_connect[to_connect['type'].isin(['AFR', 'OPR'])]

        # Add a node
        x = []
        y = []
        # TODO: add check for if direction is 'H' / add functionality to deal with other directions than 'H'
        for _, i in to_connect.iterrows():
            x.append(i['geometry'].coords[-1][0])
            y.append(i['geometry'].coords[-1][1])
        x = np.sum(x) / len(to_connect)
        y = np.sum(y) / len(to_connect)
        NWB_Graph.add_node(crossroads_id, pos=(x, y), probability=0)

        # Connect all to nodes to new node
        for index, i in to_connect.iterrows():
            already_connected.append(index)
            NWB_Graph.add_edge(i['to'], crossroads_id, geom=None, weight=0)

        crossroads_id += 1

#st.write(crossroads_id)
#st.write(NWB_Graph.size())

max_dist = 30_000  # Max distance which can be travelled in 18 minutes while going 100 km/h
max_prob = 1/120  # Variable to configure the algorithm by

# TODO: For now this only works for undirected graph, create a system for directed ones (using successor and predeccesor for neighbors)
# TODO: Can we add edges and from one graph to another directly? Instead of this copying data
# TODO: Maybe look at starting at exit points, allowing only those to be legitimate placements for inspectors

def single_inspector_placement(name, total_graph: nx.Graph):
    # print(f"Starting placement at node {name}")
    inspector_graph = nx.Graph()
    inspector_graph.add_node(name,
                             pos=total_graph.nodes[name]['pos'],
                             probability=total_graph.nodes[name]['probability'])
    prob_sum = total_graph.nodes[name]['probability']

    # Node addition loop
    node_added = True
    possible_additions = {}
    while node_added is True:
        node_added = False

        # Construct list of all possible neighbours to add
        neighbours = [neighbour for neighbour in total_graph.neighbors(name)]
        for neighbour in neighbours:
            if neighbour not in inspector_graph.nodes:
                possible_additions[neighbour] = (name, total_graph.edges[name, neighbour]['weight'])
        # Sort value by lowest distance
        possible_additions = dict(sorted(possible_additions.items(), key=lambda item: item[1][1]))

        # Check if any of the possible additions meet the requirements
        for possible_add in possible_additions:
            # Check for maximum probability requirement
            if prob_sum + total_graph.nodes[possible_add]['probability'] < max_prob:

                # Add node to inspector graph (to perform distance check)
                inspector_graph.add_node(possible_add,
                                         pos=total_graph.nodes[possible_add]['pos'],
                                         probability=total_graph.nodes[possible_add]['probability'])
                edge_data = total_graph.edges[possible_additions[possible_add][0], possible_add]
                inspector_graph.add_edge(name, possible_add, geom=edge_data['geom'], weight=edge_data['weight'])

                # Check for maximum distance requirement
                below_max_dist = False
                for node in inspector_graph.nodes:
                    lengths, _ = nx.single_source_dijkstra(inspector_graph, node)
                    lengths_list = list(lengths.values())

                    if all([dist <= max_dist for dist in lengths_list]):
                        below_max_dist = True
                        break  # At least one node can reach all other nodes within max_dist
                
                if below_max_dist:  # possible add meets all checks and can be added to the inspector network
                    prob_sum += total_graph.nodes[possible_add]['probability']
                    name = possible_add
                    del possible_additions[possible_add]
                    node_added = True  # Makes node addition loop run again
                    break

                else:  # Remove the possible addition and move on to the next possible one
                    inspector_graph.remove_node(possible_add)

    # Inspector graph is finished, remove nodes from main graph to take them out of consideration for others
    for node in inspector_graph.nodes:
        total_graph.remove_node(node)

    return total_graph, inspector_graph


def marching_leaf_algorithm(total_graph: nx.Graph):
    inspector_graphs = []
    min_degree = 0
    while total_graph.size() > 0:
        min_degree += 1
        for name, degree in list(total_graph.degree):
            if degree == min_degree:
                total_graph, inspector_graph = single_inspector_placement(name, total_graph)
                inspector_graphs.append(inspector_graph)
                min_degree = 0
                print(len(inspector_graphs))
                if len(inspector_graphs) % 100 == 0:
                    print(len(inspector_graphs))

                break

    return inspector_graphs

def place_inspectors(inspector_graphs: list[nx.Graph]):
    """"Determine most central place for inspectors in their relative graphs
    by taking the node with the lowest total distance to all other nodes"""
    inspector_locations = []
    for inspector_graph in inspector_graphs:
        nodes = []
        distances = []
        for node in inspector_graph.nodes:
            lengths, _ = nx.single_source_dijkstra(inspector_graph, node)
            lengths_list = list(lengths.values())
            nodes.append(node)
            distances.append(np.sum(lengths_list))
        
        inspector_locations.append(nodes[np.array(distances).argmin()])

    return inspector_locations


inspector_graphs = marching_leaf_algorithm(NWB_Graph.copy())

locs = place_inspectors(inspector_graphs)

import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx

def inspector_plotter(total_graph, inspector_locations):
    # Create a figure
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot the edges
    for edge in total_graph.edges.data():
        line = edge[2]['geom']
        if line is not None:
            x, y = line.coords.xy
            ax.plot(x, y, color='k')

    # Plot the inspector locations
    for node in total_graph.nodes():
        if node in inspector_locations:
            position = total_graph.nodes[node]['pos']
            ax.plot(position[0], position[1], '.', color='r')

    # Set labels and axis properties
    ax.set_title("Network")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_aspect('equal')

    # Show the plot within the Streamlit app
    st.pyplot(fig)

# Create a sample graph and inspector locations
#total_graph = nx.Graph()
#inspector_locations = [1, 3]

# Streamlit app
st.title("Inspector Plotter")
if st.button("Generate Plot"):
    st.write("Graph with inspectors:")
    inspector_plotter(NWB_Graph, locs)
