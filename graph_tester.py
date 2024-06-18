import networkx as nx
import matplotlib.pyplot as plt
import string
from vascular_tool import consolidate_internal_graph_edges

#Create labelled graph
G = nx.Graph()

alphabet = list(string.ascii_uppercase)
for letter in alphabet:
    if letter <= 'I':
        G.add_node(letter)
#Make connections
G.add_edge('A','C', weight = 3)
G.add_edge('B','C', weight = 3)
G.add_edge('C','D',weight = 1)
G.add_edge('D','E',weight = 1)
G.add_edge('E','F',weight = 1)
G.add_edge('D','I', weight = 3)
G.add_edge('F','G', weight = 3)
G.add_edge('F','H', weight = 3)


pos = nx.spring_layout(G, seed=7)  # positions for all nodes - seed for reproducibility

# nodes
nx.draw_networkx_nodes(G, pos, node_size=700)

# edges
nx.draw_networkx_edges(G, pos, width=6)


# node labels
nx.draw_networkx_labels(G, pos, font_size=20, font_family="sans-serif")
# edge weight labels
edge_labels = nx.get_edge_attributes(G, "weight")
nx.draw_networkx_edge_labels(G, pos, edge_labels)

plt.show()