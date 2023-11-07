import data_filtering
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from datetime import datetime
import geopandas as gpd
import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.stats import gaussian_kde
import networkx as nx
import node_probability
import random
from matplotlib.lines import Line2D
from projection_conversions import DutchRDtoWGS84, WGS84toDutchRD


def create_edges_dataframe(road_network):
    edges = {
        'from': [],
        'to': [],
        'type': [],
        'direction': [],
        'name': [],
        'geometry': []
    }

    for _, row in road_network.iterrows():
        edges['from'].append(row['JTE_ID_BEG'])
        edges['to'].append(row['JTE_ID_END'])
        edges['type'].append(row['BST_CODE'])
        edges['direction'].append(row['RIJRICHTNG'])
        edges['name'].append(row['STT_NAAM'])
        edges['geometry'].append(row['geometry'])

    # Create a GeoDataFrame from the edges dictionary
    edges_gdf = gpd.GeoDataFrame(edges)
    return edges_gdf

# Example usage:
# edges_gdf = create_edges_dataframe(road_network)



def create_NWB_Graph(points_gdf, edges_gdf):
    NWB_Graph = nx.Graph()
    # Add nodes to NWB_Graph
    for _, point in points_gdf.iterrows():
        NWB_Graph.add_node(point['name'], pos=point['geometry'].coords[0], probability=point['probability'])

    already_connected = []
    crossroads_id = 1
    # Iterate through edges and connect them in NWB_Graph
    for index, edge in edges_gdf.iterrows():
        NWB_Graph.add_edge(edge['from'], edge['to'], geom=edge['geometry'], weight=edge['geometry'].length)

        if edge['type'] in ['AFR', 'OPR'] and index not in already_connected:
            to_connect = edges_gdf[edges_gdf['name'] == edge['name']]
            to_connect = to_connect[to_connect['type'].isin(['AFR', 'OPR'])]
            # Add a node
            x = []
            y = []
            for _, i in to_connect.iterrows():
                x.append(i['geometry'].coords[-1][0])
                y.append(i['geometry'].coords[-1][1])
            x = np.sum(x) / len(to_connect)
            y = np.sum(y) / len(to_connect)
            NWB_Graph.add_node(crossroads_id, pos=(x, y), probability=0)
            # Connect all nodes to the new node
            for index, i in to_connect.iterrows():
                already_connected.append(index)
                NWB_Graph.add_edge(i['to'], crossroads_id, geom=None, weight=0)

            crossroads_id += 1

    return NWB_Graph

# Example usage:
# NWB_Graph = create_NWB_Graph(points_gdf, edges_gdf)



def get_inspector_locations(inspector_graphs):
    """
    Get inspector locations from a list of inspector graphs.

    Args:
        inspector_graphs (list of networkx.Graph): A list of inspector graphs.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing inspector locations with attributes.
    """
    inspector_coordinates = []
    for inspector_graph in inspector_graphs:
        inspector_node = list(inspector_graph.nodes)[0]
        inspector_position = inspector_graph.nodes[inspector_node]['pos']
        inspector_coordinates.append({'inspector_name': inspector_node, 'x_coordinate': inspector_position[0], 'y_coordinate': inspector_position[1]})
    inspector_df = pd.DataFrame(inspector_coordinates)
    inspector_gdf = gpd.GeoDataFrame(inspector_df, 
                       geometry=gpd.points_from_xy(inspector_df['x_coordinate'], inspector_df['y_coordinate']))

    return inspector_gdf

# Example usage:
# inspector_gdf = get_inspector_locations(inspector_graphs)



def get_inspector_locations_v3(new_placements, NWB_Graph):
    """
    Create a GeoDataFrame with inspector locations based on the given new_placements data.

    Args:
    new_placements (dict): A dictionary containing inspector placement data.
    NWB_Graph (nx.MultiDiGraph): The road network graph containing node coordinates.

    Returns:
    gpd.GeoDataFrame: A GeoDataFrame with inspector locations, including their properties.
    """
    # Define the columns for the GeoDataFrame
    columns = ['inspector_name', 'probability', 'avg_travel_time', 'geometry']
    inspector_gdf = gpd.GeoDataFrame(columns=columns)

    # Iterate through new_placements data and add rows to the GeoDataFrame
    for inspector_id, inspector_data in new_placements.items():
        probability, avg_travel_time = inspector_data[0], inspector_data[2]

        # Create a Point geometry representing the inspector's location
        geometry = Point(NWB_Graph.nodes[inspector_id]['pos'])

        # Add a row to the GeoDataFrame
        inspector_gdf.loc[len(inspector_gdf)] = [inspector_id, probability, avg_travel_time, geometry]

    return inspector_gdf

# Usage example:
# inspector_gdf = get_inspector_locations_v3(new_placements, NWB_Graph)




def filter_accidents_in_time_range(accident_data, start_time, end_time):
    """
    Filter accidents within a specified time range and convert coordinates.

    Args:
        accident_data (gpd.GeoDataFrame): GeoDataFrame containing accident data.
        start_time (pd.Timestamp): Start time of the time range.
        end_time (pd.Timestamp): End time of the time range.

    Returns:
        gpd.GeoDataFrame: Filtered GeoDataFrame containing accidents within the time range.
    """
    locations = []
    # Iterate through each accident record and convert WGS84 to Dutch RD coordinates
    for _, accident in accident_data.iterrows():
        location = Point(WGS84toDutchRD(accident['primaire_locatie_lengtegraad'],
                                        accident['primaire_locatie_breedtegraad']))
        locations.append(location)
    accident_data["geometry"] = locations
    accident_data['starttime_new'] = pd.to_datetime(accident_data['starttime_new'])
    filtered_data = accident_data[(accident_data['starttime_new'] >= start_time) & (accident_data['starttime_new'] <= end_time)]
    filtered_data = gpd.GeoDataFrame(filtered_data, geometry=filtered_data['geometry'])
    return filtered_data

# Example usage:
# test_incident_gdf = filter_accidents_in_time_range(df_incident, '2019-10-1', '2019-10-2')



def find_nearest_inspector(test_incident_gdf, inspector_gdf):
    """
    Find the nearest inspector to each incident point.

    Args:
        test_incident_gdf (gpd.GeoDataFrame): GeoDataFrame containing incident data.
        inspector_gdf (gpd.GeoDataFrame): GeoDataFrame containing inspector locations.

    Returns:
        gpd.GeoDataFrame: Modified incident GeoDataFrame with 'nearest_inspector_name' column.
    """
    test_incident_gdf['nearest_inspector_name'] = ""

    for index, incident in test_incident_gdf.iterrows():
        incident_point = incident['geometry']

        min_distance = float('inf')
        nearest_point_name = None

        for _, point in inspector_gdf.iterrows():
            point_name = point['inspector_name']
            point_geom = point['geometry']
            distance = incident_point.distance(point_geom)

            if distance < min_distance:
                min_distance = distance
                nearest_point_name = point_name

        test_incident_gdf.at[index, 'nearest_inspector_name'] = nearest_point_name

    return test_incident_gdf

def find_closest_edge(test_incident_gdf, NWB_Graph):
    """
    Find the closest edge to each incident point.

    Args:
        test_incident_gdf (gpd.GeoDataFrame): GeoDataFrame containing incident data.
        NWB_Graph (networkx.Graph): Graph representing road network.

    Returns:
        gpd.GeoDataFrame: Modified incident GeoDataFrame with 'close_edge_node_1' and 'close_edge_node_2' columns.
    """
    test_incident_gdf['close_edge_node_1'] = ""
    test_incident_gdf['close_edge_node_2'] = ""

    for index, incident in test_incident_gdf.iterrows():
        incident_point = incident['geometry']

        min_distance = float('inf')
        closest_edge = None
        closest_edge_node_1 = None
        closest_edge_node_2 = None

        for edge in NWB_Graph.edges(data=True):
            edge_geom = edge[2]['geom']

            if edge_geom is not None:
                distance = incident_point.distance(edge_geom)
                if distance < min_distance:
                    min_distance = distance
                    closest_edge = edge
                    closest_edge_node_1, closest_edge_node_2 = closest_edge[:2]

        test_incident_gdf.at[index, 'close_edge_node_1'] = closest_edge_node_1
        test_incident_gdf.at[index, 'close_edge_node_2'] = closest_edge_node_2

    return test_incident_gdf

# Example usage:
# updated_test_incident_gdf = find_nearest_inspector(test_incident_gdf, inspector_gdf)
# updated_test_incident_gdf = find_closest_edge(updated_test_incident_gdf, NWB_Graph)



def shortest_path_distance(graph, start_node_name, end_node_name):
    """
    Calculate the shortest route distance between two end points in the network.
    
     Reference number:
     - graph: NetworkX image
     - start_node: start node name
     - end_node: end point name
    
     Reply:
     - distance: length of the shortest path, whether or not the path returns None
    """
    try:
        distance = nx.shortest_path_length(graph, source=start_node_name, target=end_node_name, weight='weight')
        return distance
    except nx.NetworkXNoPath:    
        return float('inf')

# Example usage:
# distance_364061076_600012888 = shortest_path_distance(NWB_Graph, 364061076, 600012888)



def distance_inspector_to_incident(graph, one_incident_gdf):
    """
    Calculate the distance from the inspector to the incident point.

    Args:
        graph (networkx.Graph): Graph representing the road network.
        one_incident_gdf (gpd.GeoDataFrame): GeoDataFrame containing one incident data point.

    Returns:
        float: The distance from the inspector to the incident point.
    """
    edge_node_1 = graph.nodes[one_incident_gdf['close_edge_node_1']]
    edge_node_2 = graph.nodes[one_incident_gdf['close_edge_node_2']]
    pos1 = edge_node_1['pos']
    pos2 = edge_node_2['pos']
    
    path_1 = shortest_path_distance(graph, one_incident_gdf['close_edge_node_1'], one_incident_gdf['nearest_inspector_name']) + \
             ((one_incident_gdf['geometry'].x - pos1[0]) ** 2 + (one_incident_gdf['geometry'].y - pos1[1]) ** 2) ** 0.5
    path_2 = shortest_path_distance(graph, one_incident_gdf['close_edge_node_2'], one_incident_gdf['nearest_inspector_name']) + \
             ((one_incident_gdf['geometry'].x - pos2[0]) ** 2 + (one_incident_gdf['geometry'].y - pos2[1]) ** 2) ** 0.5
    
    # Return the minimum distance between two possible paths
    return min(path_1, path_2)

# Example usage:
# distance_39 = distance_inspector_to_incident(NWB_Graph, test_incident_gdf.iloc[39])



def calculate_total_distance(graph, incident_gdf):
    """
    Calculate total distance, average distance, and update distances in incident data.

    Args:
        graph (networkx.Graph): Graph representing the road network.
        incident_gdf (gpd.GeoDataFrame): GeoDataFrame containing incident data.

    Returns:
        Tuple[gpd.GeoDataFrame, float, float]: 
            - Modified incident GeoDataFrame with updated 'Travel distance' column.
            - Total distance traveled by inspectors to reach incidents.
            - Average distance traveled by inspectors to reach reachable incidents.
    """
    sum_distance = 0
    unable_reach_number = 0

    for index, row in incident_gdf.iterrows():
        distance = distance_inspector_to_incident(graph, row)
        incident_gdf.loc[index, 'Travel distance'] = distance
        if distance > 10000000000000:
            unable_reach_number += 1
        else:
            sum_distance += distance

    # Calculate the average distance for reachable incidents
    if len(incident_gdf) - unable_reach_number > 0:
        average_distance = sum_distance / (len(incident_gdf) - unable_reach_number)
    else:
        average_distance = 0.0  # Avoid division by zero

    return incident_gdf, sum_distance, average_distance

# Example usage:
# test_incident_gdf_distance, test_total_distance, test_average_distance = calculate_total_distance(NWB_Graph, test_incident_gdf)



def plot_travel_distance_distribution(test_incident_gdf, test_average_distance):
    """
    Plot the travel distance distribution of incidents and show average travel distance.

    Args:
        test_incident_gdf (gpd.GeoDataFrame): GeoDataFrame containing incident data and 'Travel distance' column.
        test_average_distance (float): The average travel distance of inspectors to reach incidents.
    """
    # Filter out infinite distance values
    filtered_incidents = test_incident_gdf[np.isfinite(test_incident_gdf['Travel distance'])]
    filtered_incidents['Travel distance (KM)'] = filtered_incidents['Travel distance']/1000

    plt.figure(figsize=(8, 6))
    plt.hist(filtered_incidents['Travel distance (KM)'], bins=50, density=False, color='blue', edgecolor='black', alpha=0.7)
    plt.xlabel('Travel distance')
    plt.ylabel('Number of incidents')
    plt.legend([f'Average travel distance: {test_average_distance/1000:.2f} km'])
    plt.title('Travel distance distribution')
    plt.grid(True)
    plt.show()

# Example usage:
# plot_travel_distance_distribution(test_incident_gdf_distance, test_average_distance)



def visualize_inspector_coverage(inspector_gdf, NWB_Graph, road_network, NL_map):
    # Create a dictionary to store the nodes within 30000 distance for each inspector
    inspector_coverage = {}

    # Define the maximum distance
    max_distance = 30000

    # Iterate through each inspector
    for _, inspector in inspector_gdf.iterrows():
        inspector_name = inspector['inspector_name']
        inspector_pos = inspector['geometry'].coords[0]

        # Find nodes within the maximum distance using network analysis
        reachable_nodes = nx.single_source_dijkstra_path_length(NWB_Graph, source=inspector_name, cutoff=max_distance)

        # Store the reachable nodes for this inspector
        inspector_coverage[inspector_name] = list(reachable_nodes.keys())

    # Add the 'inspector_coverage' information to the 'inspector_gdf' GeoDataFrame
    inspector_gdf['inspector_coverage'] = inspector_gdf['inspector_name'].map(inspector_coverage)

    # Create a color mapping for covered and uncovered road segments
    covered_color = 'green'  # Color for covered road segments
    uncovered_color = 'gray'  # Color for uncovered road segments

    fig, ax = plt.subplots(figsize=(12, 12))

    # Plot the road network
    road_network.plot(ax=ax, color=uncovered_color, linewidth=0.5)
    NL_map.plot(ax = ax, facecolor = "None")
    
    # Plot inspectors' points
    #inspector_gdf.plot(ax=ax, color='red', markersize=20, label='Inspectors')

    legend_labels =  ['Within 15 min', 'More than 15 min']

    for index, row in inspector_gdf.iterrows():
        inspector_name = row['inspector_name']
        coverage_nodes = row['inspector_coverage']
        coverage_edges = []

        # Collect edges based on coverage nodes
        for node in coverage_nodes:
            if NWB_Graph.has_node(node):
                neighbors = NWB_Graph.neighbors(node)
                for neighbor in neighbors:
                    if NWB_Graph.has_edge(node, neighbor):
                        coverage_edges.append((node, neighbor))

        coverage_edges_gdf = gpd.GeoDataFrame({'geometry': [NWB_Graph[edge[0]][edge[1]]['geom'] for edge in coverage_edges]})
        coverage_edges_gdf.plot(ax=ax, color=covered_color, linewidth=2)

    ax.set_title("Inspector Coverage Map")

    # Add legend with labels
    legend_elements = [Line2D([0], [0], color=covered_color, lw=4, label=legend_labels[0]),
                       Line2D([0], [0], color=uncovered_color, lw=4, label=legend_labels[1])]

    ax.legend(handles=legend_elements)
    plt.show()
    
    return inspector_gdf

# Example usage:
# updated_inspector_gdf = visualize_inspector_coverage(inspector_gdf, NWB_Graph, road_network, NL_map)



def analyze_and_visualize_coverage_v3(inspector_gdf, NWB_Graph, NL_map, road_network):
    unique_edges = set()
    edge_coverage_count = defaultdict(int)

    # Calculate coverage
    for _, inspector in inspector_gdf.iterrows():
        distances = nx.single_source_dijkstra_path_length(NWB_Graph, inspector['inspector_name'], cutoff=30000)
        for node, distance in distances.items():
            for edge in NWB_Graph.edges(node, data=True):
                edge_data = edge[2]
                edge_id = (edge[0], edge[1], edge_data['geom'])  
                unique_edges.add(edge_id)  
                edge_coverage_count[edge_id] += 1

    # Compute covered and total lengths
    covered_length = sum(NWB_Graph[edge[0]][edge[1]][0]['geom'].length
                         for edge in unique_edges
                         if NWB_Graph[edge[0]][edge[1]][0]['geom'] is not None)

    total_length = sum(edge_data['geom'].length
                       for _, _, edge_data in NWB_Graph.edges(data=True)
                       if 'geom' in edge_data and edge_data['geom'] is not None)

    coverage_ratio = covered_length / total_length

    # Compute redundancy
    repeated_length = sum(
        NWB_Graph[edge[0]][edge[1]][0]['geom'].length
        for edge, count in edge_coverage_count.items()
        if count > 1 and 'geom' in NWB_Graph[edge[0]][edge[1]][0] and NWB_Graph[edge[0]][edge[1]][0]['geom'] is not None
    )
    redundancy_rate = repeated_length / covered_length if covered_length > 0 else 0

    # Printing coverage information
    print(f"Total covered length: {covered_length}")
    print(f"Total network length: {total_length}")
    print(f"Coverage ratio: {coverage_ratio}")
    print(f"Total repeated coverage length: {repeated_length}")
    print(f"Redundancy rate: {redundancy_rate}")

    # Visualization
    fig, ax = plt.subplots(figsize=(12, 12))

    road_network.plot(ax=ax, color='red', linewidth=2, label='Uncovered')
    NL_map.plot(ax=ax, facecolor="None")

    covered_edges_geom = [NWB_Graph[edge[0]][edge[1]][0]['geom'] for edge in unique_edges
                          if 'geom' in NWB_Graph[edge[0]][edge[1]][0] and NWB_Graph[edge[0]][edge[1]][0]['geom'] is not None]
    covered_edges_gdf = gpd.GeoDataFrame({'geometry': covered_edges_geom})
    covered_edges_gdf.plot(ax=ax, color='green', linewidth=2, label='Covered')

    # Add legend and text
    legend_elements = [Line2D([0], [0], color='green', lw=2, label='Covered'),
                       Line2D([0], [0], color='red', lw=2, label='Uncovered')]
    ax.legend(handles=legend_elements, loc='upper right')

    ax.set_title('Road Network Coverage')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.text(
        x=0.01,  
        y=0.99,  
        s=f"Total covered length: {covered_length / 1000:.2f} km\n"
          f"Total network length: {total_length / 1000:.2f} km\n"
          f"Coverage ratio: {coverage_ratio:.2%}\n"
          f"Total repeated coverage length: {repeated_length / 1000:.2f} km\n"
          f"Redundancy rate: {redundancy_rate:.2%}",
        transform=ax.transAxes,  
        fontsize=10,
        verticalalignment='top',  
        horizontalalignment='left',  
        bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white', alpha=0.9)  
    )

    plt.show()