import geopandas as gpd
import folium
from folium.plugins import MarkerCluster
import json
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from datetime import datetime
from IPython.display import display
import ipywidgets as widgets
from folium.plugins import HeatMap
import seaborn as sns

import data_filtering
def distribution_in_day(df_incident):
    df_incident['starttime_new'] = pd.to_datetime(df_incident['starttime_new'])
    df_incident['endtime_new'] = pd.to_datetime(df_incident['endtime_new'])
    df_incident['Hour'] = df_incident['starttime_new'].dt.hour
    hourly_counts = df_incident.groupby(['type', 'Hour']).size().unstack(fill_value=0)
    def plot_accidents(accident_type):
        if accident_type == 'all':
            filtered_df = df_incident
        else:
            filtered_df = df_incident[df_incident['type'] == accident_type]

        hourly_counts = filtered_df['Hour'].value_counts().sort_index()
        plt.figure(figsize=(10, 6))
        plt.bar(hourly_counts.index, hourly_counts.values)
        plt.title(f'Hourly Distribution of {accident_type} Incidents')
        plt.xlabel('Hour')
        plt.ylabel('Number of Incidents')
        plt.xticks(range(24))
        plt.show()
    # Get all different incident types, including "all incidents"
    accident_types = ['all'] + list(df_incident['type'].unique())
    # Create drop-down menu
    dropdown = widgets.Dropdown(options=accident_types, description='Incident Type:')
    output = widgets.interactive(plot_accidents, accident_type=dropdown)
    # show the figure
    display(output)

def distribution_in_week(df_incident):
    df_incident['starttime_new'] = pd.to_datetime(df_incident['starttime_new'])
    df_incident['endtime_new'] = pd.to_datetime(df_incident['endtime_new'])
    df_incident['Day_of_Week'] = df_incident['starttime_new'].dt.dayofweek
    def plot_accidents(accident_type):
        if accident_type == 'all':
            filtered_df = df_incident
        else:
            filtered_df = df_incident[df_incident['type'] == accident_type]
        daily_counts = filtered_df['Day_of_Week'].value_counts().sort_index()
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_counts.index = days[:len(daily_counts)] 
        plt.figure(figsize=(10, 6))
        plt.bar(daily_counts.index, daily_counts.values)
        plt.title(f'Daily Distribution of {accident_type} Incidents')
        plt.xlabel('Day of the Week')
        plt.ylabel('Number of Incidents')
        plt.xticks(rotation=45)
        plt.show()
    accident_types = ['all'] + list(df_incident['type'].unique())
    dropdown = widgets.Dropdown(options=accident_types, description='Incident Type:')
    output = widgets.interactive(plot_accidents, accident_type=dropdown)
    display(output)

def distribution_in_year(df_incident):
    df_incident['starttime_new'] = pd.to_datetime(df_incident['starttime_new'])
    df_incident['endtime_new'] = pd.to_datetime(df_incident['endtime_new'])
    df_incident['Month'] = df_incident['starttime_new'].dt.month
    def plot_monthly_distribution(accident_type):
        if accident_type == 'all':
            filtered_df = df_incident
        else:
            filtered_df = df_incident[df_incident['type'] == accident_type]
        monthly_counts = filtered_df['Month'].value_counts().sort_index()
        plt.figure(figsize=(10, 6))
        plt.bar(monthly_counts.index, monthly_counts.values)
        plt.title(f'Monthly Distribution of {accident_type} Incidents')
        plt.xlabel('Month')
        plt.ylabel('Number of Incidents')
        plt.xticks(range(7, 13)) 
        plt.show()
    accident_types = ['all'] + list(df_incident['type'].unique())
    dropdown = widgets.Dropdown(options=accident_types, description='Incident Type:')
    output = widgets.interactive(plot_monthly_distribution, accident_type=dropdown)
    display(output)

def distribution_time(df, time):
    if time == 'day':
        distribution_in_day(df)
    if time == 'week':
        distribution_in_week(df)
    if time == 'year':
        distribution_in_year(df)


def distribution_duration(df_incident):
    df_incident['starttime_new'] = pd.to_datetime(df_incident['starttime_new'])
    df_incident['endtime_new'] = pd.to_datetime(df_incident['endtime_new'])
    df_incident['Duration_Minutes'] = (df_incident['endtime_new'] - df_incident['starttime_new']).dt.total_seconds() / 60
    df_incident = df_incident[(df_incident['Duration_Minutes'] <= 500) & (df_incident['Duration_Minutes'] > 0)]
    def plot_duration_distribution(accident_type):
        if accident_type == 'all':
            filtered_df = df_incident
        else:
            filtered_df = df_incident[df_incident['type'] == accident_type]
        plt.figure(figsize=(10, 6))
        plt.hist(filtered_df['Duration_Minutes'], bins=20, edgecolor='k')
        plt.title(f'Duration Distribution of {accident_type} Incidents')
        plt.xlabel('Duration (Minutes)')
        plt.ylabel('Number of Incidents')
        plt.show()
    accident_types = ['all'] + list(df_incident['type'].unique())
    dropdown = widgets.Dropdown(options=accident_types, description='Incident Type:')
    output = widgets.interactive(plot_duration_distribution, accident_type=dropdown)
    display(output)

def type_distribution(df):
    incident_counts = df['type'].value_counts()
    plt.figure(figsize=(10, 6))
    plt.pie(incident_counts, labels=incident_counts.index, autopct='%1.1f%%', startangle=140)
    plt.xlabel('Incident Type')
    #plt.ylabel('Number of Incidents')
    plt.title('Incident Type Distribution')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def simultaneous_accidents(df):
    df['starttime_new'] = pd.to_datetime(df['starttime_new'])
    # Sort data to ensure correct chronological order
    df = df.sort_values(by='starttime_new')
    # Use a one-hour window to count the number of accidents in every 15 minutes
    counts = df.resample('15T', on='starttime_new').size()
    # Find the time with the most accidents
    max_time = counts.idxmax()
    max_incidents = counts.max()
    plt.figure(figsize=(40, 6))
    plt.plot(counts.index, counts.values, marker='o', linestyle='-')
    plt.xlabel('Time')
    plt.ylabel('Number of Incidents')
    plt.title('Incident Count by every 15 minutes')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.show()
    return max_time, max_incidents

def import_network_to_map(file_path):
    network_temp = gpd.read_file(file_path)
    #transform  DutchRD to WGS84
    network_temp = network_temp.to_crs("EPSG:4326")
    map = folium.Map(location=[52.399190, 4.893658])
    gjson = folium.features.GeoJson(
        network_temp,
    ).add_to(map)
    return map

def add_point_to_map(df_incident, map):
    # use MarkerCluster to show the points and cluster them
    marker_cluster = MarkerCluster().add_to(map)
    for _, row in df_incident.iterrows():
        popup_content = f"ID: {row['id']}<br>Type: {row['type']}<br>Start time: {row['starttime_new']}"
        folium.Marker(
            location=[row['primaire_locatie_breedtegraad'], row['primaire_locatie_lengtegraad']],
            popup=popup_content,
            icon=folium.Icon(color='red')
            ).add_to(marker_cluster)
    # save the map
    map.save('incidents_map_test.html')
    return map

def create_heatmap(df_incident, map):
    # Create a list of points from the DataFrame
    locations = df_incident[['primaire_locatie_breedtegraad', 'primaire_locatie_lengtegraad']].values.tolist()
    # Create the HeatMap layer
    heatmap = HeatMap(locations)
    # Add the HeatMap layer to the map
    heatmap.add_to(map)
    # Save the map
    map.save('heatmap.html')
    return map

def heatmap_simple(df, network_temp):
    network_temp = network_temp.to_crs("EPSG:4326")
    plt.figure(figsize=(10, 8))
    sns.kdeplot(data=df, x="primaire_locatie_lengtegraad", y="primaire_locatie_breedtegraad", fill=True, cmap="YlOrRd", thresh=0.5)
    plt.title("Traffic Incident Heatmap")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    ax = plt.gca()
    network_temp.plot(ax=ax, color='blue', linewidth=0.5, label='network')
    plt.show()

def spatio_temporal(df_incident, network_temp, timestep):
    #df_incident: dataframe of incident
    #network_temp
    #timestep: time window (min)
    # transform GIS
    network_temp = network_temp.to_crs("EPSG:4326")
    # Convert starttime_new column to datetime object
    df_incident['starttime_new'] = pd.to_datetime(df_incident['starttime_new'])
    # Calculate the time window to which each time point belongs
    time_interval = pd.Timedelta(minutes=timestep)
    df_incident['time_window_start'] = df_incident['starttime_new'] - ((df_incident['starttime_new'] - df_incident['starttime_new'].min()) % time_interval)
    # List of options for creating a slider timeline
    time_window_options = df_incident['time_window_start'].unique()
    time_window_options = sorted(time_window_options)
    # creat the slider
    time_window_slider = widgets.SelectionSlider(
        options=time_window_options,
        description='time window:',
        continuous_update=False,
        step=0.1
    )
    def plot_accidents(time_window_start):
        selected_data = df_incident[df_incident['time_window_start'] == time_window_start]

        fig, ax = plt.subplots(figsize=(10, 8))
        network_temp.plot(ax=ax, color='blue', linewidth=0.5, label='network')
        
        # Define colors for different accident types
        colors = {'general_obstruction': 'red', 'vehicle_obstruction': 'green', 'accident': 'm'}  # Replace with your actual types and corresponding colors
        
        for acc_type, acc_data in selected_data.groupby('type'):
            plt.scatter(
                acc_data['primaire_locatie_lengtegraad'], 
                acc_data['primaire_locatie_breedtegraad'], 
                marker='x', 
                s=50, 
                color=colors.get(acc_type, 'black'),  # Use the defined colors or black if type is not found
                label=acc_type
            )
        plt.xlabel('longitude')
        plt.ylabel('latitude')
        plt.title(f'Distribution of Incidents ({time_window_start})')
        plt.grid(True)
        plt.legend()
        plt.show()
    interactive_plot = widgets.interactive(plot_accidents, time_window_start=time_window_slider)
    display(interactive_plot)
    return interactive_plot