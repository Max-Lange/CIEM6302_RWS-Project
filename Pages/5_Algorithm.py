import streamlit as st
from PIL import Image
import streamlit as st
import graphviz


st.title("Algorithm")

st.title("The Algorithm: Marching Leaf Nodes")


# Create a graphlib graph object
graph = graphviz.Digraph()
#st.info("Use the button in the upper right corner of the figure to view full screen")
st.subheader("Algorithm explanation")
st.write("With this placement loop, we can go through the entire network. Once a inspector placement has been made, the nodes which that inspector covers are removed from the main network. This way once a new inspector placement is made it cannot consider already covered nodes. It also means that leaf nodes are constantly created, as removing nodes and edges from the network in this way always creates new leafes. Once going through all the existing leaf-nodes, we repeat the loop again with the newly created leaf nodes, and repeat this cycle until the entire network is covered!")
st.write("At the step finish placement in the flowchart an inspector has been placed.")

# Add nodes and edges
graph.node('start at leafnode', shape='ellipse', style='filled', fillcolor='green', label='Start at Leaf Node')
graph.node('Look at all possible neighbours of the inspector network, select the nearest one', shape='ellipse', label='Look at all possible neighbours of the inspector network, select the nearest one')
graph.node('add node to list of one inspector', shape='ellipse', label='Add Node to Inspector List')
graph.node('check sum of p', shape='ellipse', label='Check Sum of Probabilities')
graph.node('finish placement(1)*', shape='ellipse', style='filled', fillcolor='red', label='Finish Placement')
graph.node('finish placement(2)**', shape='ellipse', style='filled', fillcolor='red', label='Finish Placement')
graph.node('check time limit (18m) using Dijkstra algorithm', shape='ellipse', label='Check Time Limit (18m) using Dijkstra Algorithm')

# Add edges
graph.edge('start at leafnode', 'Look at all possible neighbours of the inspector network, select the nearest one')
graph.edge('Look at all possible neighbours of the inspector network, select the nearest one', 'add node to list of one inspector')
graph.edge('add node to list of one inspector', 'check sum of p')
graph.edge('check sum of p', 'finish placement(1)*', label='p > pmax')

graph.edge('check sum of p', 'check time limit (18m) using Dijkstra algorithm', label='p < pmax')
graph.edge('check time limit (18m) using Dijkstra algorithm', 'finish placement(2)**', label='if above 18m')
graph.edge('check time limit (18m) using Dijkstra algorithm', 'Network is not at max capacity, so add more nodes',label ='if below 18m')
graph.edge('add the neighbour node to the inspectors nodes', "start at leafnode", label="remove covered nodes so new leafnodes will emerge")
graph.edge("finish placement(1)*","select inspector's node with most central placement to place inspector at")
graph.edge("finish placement(2)**","select inspector's node with most central placement to place inspector at")
graph.edge("select inspector's node with most central placement to place inspector at", 'add the neighbour node to the inspectors nodes')
graph.edge('Network is not at max capacity, so add more nodes', "Look at all possible neighbours of the inspector network, select the nearest one", label = "add for more nodes to list" )

# Add graph to st

st.info("Use the button in the upper right corner of the figure to view the flowchart in full screen")
st.graphviz_chart(graph)

# Load the PNG images
image1 = Image.open('./Images/nodes.png')
image2 = Image.open('./Images/v1.png')
image3 = Image.open('./Images/v2.png')
image4 = Image.open('./Images/output.png')

# Add a title or text above the images
st.subheader("****All nodes in the network****")
st.write("These are the nodes in the network. All accidents are allocated to the nodes. The nodes are possible locations for road inspectors.")

# Display the first image ('nodes.png') with a caption
st.image(image1, caption='Nodes in the network', use_column_width=True)


# Streamlit-applicatie instellen
st.title("Results")


image1_path = "./Images/v1.png"
image2_path = "./Images/v2.png"

# 
col1, col2 = st.columns(2)

# IMAGE 1
with col1:
    st.image(image1_path, caption="V1", use_column_width=True)
    st.write(
"""
**V1**
- Network as is: 253
- Duration: 1m 10s
""")


# IMAGE2
with col2:
    st.image(image2_path, caption="With exit switching", use_column_width=True)
    st.write(
"""
**V2**
- With exit switching: 139
- Duration: 3m 40s
""")


# Add a title or text above the images
st.subheader("Add more info about this (latest result)....")
# Display with a caption
st.image(image4, caption='Image output', use_column_width=True)



