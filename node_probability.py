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

from projection_conversions import DutchRDtoWGS84, WGS84toDutchRD

def create_points_gdf(road_network, df_incident):
    bandwidth = np.mean(road_network['geometry'].length)
    accident_coords = df_incident[['primaire_locatie_lengtegraad', 'primaire_locatie_breedtegraad']].values.T

    kde = gaussian_kde(accident_coords, bw_method=bandwidth)

    road_network['Accident_Density'] = np.nan

    for idx, row in road_network.iterrows():
        density = kde([row.geometry.centroid.x, row.geometry.centroid.y])
        road_network.at[idx, 'Accident_Density'] = density[0]

    road_network['Unnormalized_Probability'] = road_network['Accident_Density'] * road_network['geometry'].length
    road_network['Normalized_Probability'] = road_network['Unnormalized_Probability'] / sum(road_network['Unnormalized_Probability'])

    G_NL2 = nx.Graph()

    for index, row in road_network.iterrows():
        line = row.geometry
        start_node = line.coords[0]
        end_node = line.coords[-1]

        G_NL2.add_node(start_node)
        G_NL2.add_node(end_node)

        G_NL2.add_edge(start_node, end_node, geometry=line, weight=row['Normalized_Probability'])

    amatrix = nx.adjacency_matrix(G_NL2).todense()
    copy_amatrix = amatrix.copy()

    zeros_pos = []
    node_weights = []

    for i, row in enumerate(copy_amatrix):
        nonzero = np.count_nonzero(row)
        if nonzero == 1:
            zeros_pos.append(i)

    for i in zeros_pos:
        amatrix[:, i] *= 2

    for i, row in enumerate(copy_amatrix):
        if i in zeros_pos:
            node_weights.append(0)
        else:
            node_weights.append(np.sum(row) / 2)

    points_gdf = gpd.GeoDataFrame(columns=['name', 'geometry', 'probability'])

    for i, node in enumerate(list(G_NL2.nodes())):
        name = i
        probability = node_weights[i]
        geometry = Point(node)

        points_gdf.loc[i] = [name, geometry, probability]
    
    points_gdf['probability'] = points_gdf['probability'] * 1 / sum(points_gdf['probability'])

    points = {'name': [], 'geometry': []}
    for _, row in road_network.iterrows():
        if row['JTE_ID_BEG'] not in points['name']:
            points['name'].append(row['JTE_ID_BEG'])
            points['geometry'].append(Point(row['geometry'].coords[0]))

        if row['JTE_ID_END'] not in points['name']:
            points['name'].append(row['JTE_ID_END'])
            points['geometry'].append(Point(row['geometry'].coords[-1]))

    name_to_geometry = dict(zip(points['geometry'], points['name']))

    for index, row in points_gdf.iterrows():
        if row['geometry'] in name_to_geometry:
            points_gdf.at[index, 'name'] = name_to_geometry[row['geometry']]

    for index, row in points_gdf.iterrows():
        wgs84_point = row['geometry']
        wgs84_coords = wgs84_point.coords[0]
        dutch_rd_coords = WGS84toDutchRD(wgs84_coords[0], wgs84_coords[1])
        dutch_rd_point = Point(dutch_rd_coords)
        points_gdf.at[index, 'geometry'] = dutch_rd_point

    return points_gdf