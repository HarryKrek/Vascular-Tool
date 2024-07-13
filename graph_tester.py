import networkx as nx
import matplotlib.pyplot as plt
import yaml
import numpy as np  
from vascular_tool import consolidate_internal_graph_edges

#Config
configPath = ".\\config.yml"
with open(configPath, "r") as file:
    config = yaml.safe_load(file)

#Create labelled graph
G = nx.MultiGraph()

i = 0
while i <= 9:
    G.add_node(i)
    i+=1

#Make connections
G.add_edge(1,3, weight = 3, pts = np.array([[1,2]]))
G.add_edge(2,3, weight = 3, pts = np.array([[1,2]]))
G.add_edge(3,4,weight = 1, pts = np.array([[1,2],[3,4]]))
G.add_edge(4,5,weight = 1, pts = np.array([[1,2],[3,4]]))
G.add_edge(5,6,weight = 1, pts = np.array([[1,2],[3,4]]))
G.add_edge(9,4, weight = 3, pts = np.array([[1,2]]))
G.add_edge(7,6, weight = 3, pts = np.array([[1,2]]))
G.add_edge(8,6, weight = 3, pts = np.array([[1,2]]))


G_cleaned = consolidate_internal_graph_edges(G, config)

nx.draw(G_cleaned)


plt.show()