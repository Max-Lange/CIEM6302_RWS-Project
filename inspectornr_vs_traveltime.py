import numpy as np
import networkx as nx
from matplotlib import pyplot as plt

def initial_avg_travel_time(total_graph: nx.MultiDiGraph, inspector_placements: dict):
    """Calculates the initial travel times for if inspectors are placed at all exits"""
    new_placements = dict()

    for _, inspector_data in inspector_placements.items():
        best_exit_placement = None
        total_distance_sum = np.inf
        prob_scaled_distance_sum = np.inf
        for exit in inspector_data[1]:
            distance_sum_i = 0
            count = 0
            prob_scaled_distance_i = 0
            prob_sum = 0

            # Go through all nodes and add the travel time info if they are part of the exit
            distances = nx.single_source_dijkstra_path_length(total_graph, exit)
            for node, distance in distances.items():
                if total_graph.nodes[node]['closest_exit'] in inspector_data[1]:
                    distance_sum_i += distance
                    prob_scaled_distance_i += distance * total_graph.nodes[node]['probability']
                    count += 1
                    prob_sum += total_graph.nodes[node]['probability']

            # Keep travel time data if this is the best exit placement
            if prob_scaled_distance_i < prob_scaled_distance_sum:
                best_exit_placement = exit
                prob_scaled_distance_sum = prob_scaled_distance_i
                total_distance_sum = distance_sum_i

        new_placements[best_exit_placement] = [inspector_data[0], inspector_data[1],
                                               prob_scaled_distance_sum, prob_sum,
                                               total_distance_sum, count]

    return new_placements


def update_travel_times(total_graph: nx.MultiDiGraph, inspector_placements: dict, to_update: int):
    """Updates the travel times if two inspectors are combined into one. 
    Does the same as the 'initial_travel_time' function but only for one inspector, saving computing time"""
    best_exit_placement = None
    total_distance_sum = np.inf
    prob_scaled_distance_sum = np.inf
    to_update_data = inspector_placements[to_update]
    for exit in to_update_data[1]:
        distance_sum_i = 0
        count = 0
        prob_scaled_distance_i = 0
        prob_sum = 0
        
        # Go through all nodes and add the travel time info if they are part of the exit
        distances = nx.single_source_dijkstra_path_length(total_graph, exit)
        for node, distance in distances.items():
            if total_graph.nodes[node]['closest_exit'] in to_update_data[1]:
                distance_sum_i += distance
                prob_scaled_distance_i += distance * total_graph.nodes[node]['probability']
                count += 1
                prob_sum += total_graph.nodes[node]['probability']

        # Keep travel time data if this is the best exit placement
        if prob_scaled_distance_i < prob_scaled_distance_sum:
            best_exit_placement = exit
            prob_scaled_distance_sum = prob_scaled_distance_i
            total_distance_sum = distance_sum_i

        inspector_placements[to_update] = [to_update_data[0], to_update_data[1],
                                           prob_scaled_distance_sum, prob_sum, total_distance_sum,
                                           count, best_exit_placement]

    return inspector_placements

def inspector_nr_vs_travel_time(total_graph: nx.MultiDiGraph, exit_id_max: int, inspector_speed: int):
    """Calculates the different kind of travel times for all possible numbers of inspectors"""
    inspector_placements = dict()
    inspector_speed = inspector_speed / 3.6

    for exit_node in range(1, exit_id_max+1):
        inspector_placements[exit_node] = [total_graph.nodes[exit_node]['probability'], [exit_node]]

    inspector_placements = initial_avg_travel_time(total_graph, inspector_placements)
    total_dist_sum = 0
    count_sum = 0
    prob_dist_sum = 0
    prob_sum = 0
    for inspector, data in inspector_placements.items():
        prob_dist_sum += data[2]
        prob_sum += data[3]
        total_dist_sum += data[4]
        count_sum += data[5]

    inspector_counts = [len(inspector_placements)]
    count_tts = [(total_dist_sum / count_sum) / inspector_speed / 60]
    prob_tts = [(prob_dist_sum / prob_sum) / inspector_speed / 60]

    # Main Loop
    removed = []
    while len(inspector_placements) > 20:
        # Determine inspecotr network with lowest probability
        lowest_prob_inspector = None
        lowest_prob = np.inf
        for exit_node, inspector_data in inspector_placements.items():
            if inspector_data[0] < lowest_prob:
                lowest_prob_inspector = exit_node
                lowest_prob = inspector_data[0]

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

        if closest_exit == None:
            # Special clause for if no neighbours can be found
            inspector_placements[lowest_prob_inspector][0] = np.inf

        else:
            for inspector, inspector_data in inspector_placements.items():
                if closest_exit in inspector_data[1]:
                    closest_inspector = inspector
            
            # Combine inspector networks into lowest_prob_inspector and delete closest_inspector
            inspector_placements[lowest_prob_inspector][0] += inspector_placements[closest_inspector][0]
            for exit in inspector_placements[closest_inspector][1]:
                inspector_placements[lowest_prob_inspector][1].append(exit)
            del inspector_placements[closest_inspector]
            removed.append(closest_inspector)

        inspector_placements = update_travel_times(total_graph, inspector_placements, lowest_prob_inspector)
        total_dist_sum = 0
        count_sum = 0
        prob_dist_sum = 0
        prob_sum = 0
        for inspector, data in inspector_placements.items():
            prob_dist_sum += data[2]
            prob_sum += data[3]
            total_dist_sum += data[4]
            count_sum += data[5]

        inspector_counts.append(len(inspector_placements))
        count_tts.append((total_dist_sum / count_sum) / inspector_speed / 60)
        prob_tts.append((prob_dist_sum / prob_sum) / inspector_speed / 60)

    return inspector_counts, count_tts, prob_tts

def travel_time_vs_nr_plotter(counts, nodal_tts, prob_tts):
    fig = plt.figure()
    fig.set_figheight(6)
    fig.set_figwidth(6)

    plt.plot(counts, prob_tts, color='g', label='Probability Corrected')
    plt.plot(counts, nodal_tts, color='b', label='Node Count Corrected')

    plt.title("Average Travel Time")
    plt.xlabel("Number of Inspectors [-]")
    plt.ylabel("Travel Time [minutes]")
    plt.legend()
    plt.grid()
    plt.show()
