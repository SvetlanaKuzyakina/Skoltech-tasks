import numpy as np
import numpy.ma as ma
from itertools import combinations, chain
import matplotlib.pyplot as plt
import pandas as pd
import random

class Vertex:
    def __init__(self, label):
        self.node = label # node name
        self.neighbours = {} # dictionary {Vertex: edge weight}

    def get_node(self):
        return self.node

    def add_neighbour(self, v, weight = 0):
        self.neighbours[v] = weight # each neighbour carries weight of an edge 

    def get_neighbours(self):
        return self.neighbours.keys()  

    def get_weight(self, v):
        return self.neighbours[v]

    def __str__(self):
        return str(self.node) + ', adjacent vertices: ' + str([x.node for x in self.neighbours])    
    
class Graph:
    def __init__(self):
        self._vertices = {} # dictionary {node name: Vertex}
        self._num_vertices = 0

    def __iter__(self):
        return iter(self._vertices.values())

    def add_vertex(self, node):
        v = Vertex(node)
        self._vertices[node] = v 
        self._num_vertices = self._num_vertices + 1

    def get_vertex(self, node):
        if node in self._vertices:
            return self._vertices[node]
        else:
            return None

    def add_edge(self, start, end, weight = 0):
        if start not in self._vertices:
            self.add_vertex(start)
        if end not in self._vertices:
            self.add_vertex(end)

        self._vertices[start].add_neighbour(self._vertices[end], weight)
        self._vertices[end].add_neighbour(self._vertices[start], weight)

    def get_all(self):
        return self._vertices.keys()
    
    def __str__(self):
        return str(self._num_vertices) + ' vertices : \n\n' + "\n".join([ self._vertices[v].__str__() for v in self._vertices])  
    
    def draw(self, image_size = 10): # doesn`t draw weight values
        plt.figure(figsize = (image_size, image_size)) 
        node_names = list(self._vertices.keys())
        
        angles = np.linspace(0,2 * np.pi,self._num_vertices + 1)[:-1]
        xx = np.cos(angles)
        yy = np.sin(angles)
        node_dict = dict(zip(node_names, zip(xx,yy)))
        
        combs = []
        lone = []
        for node, x1, y1 in zip(node_names, xx, yy):
            v = self._vertices[node]
            v_neighs = list(v.get_neighbours())
            if len(v_neighs) != 0:
                v_edges = [(node_dict[node], node_dict[neigh.get_node()]) for neigh in v_neighs]
                combs = combs + v_edges
            else:
                lone.append([[x1], [y1]])
        combs = list(set(tuple(sorted(c)) for c in combs))  
        
        for c1, c2 in combs:
            plt.plot([c1[0],c2[0]], [c1[1],c2[1]], linewidth = 1, marker = 'o', color = "indigo", ms = 13, mfc = 'r', mec = 'b')
        
        for l in lone:
            plt.scatter(l[0], l[1], marker = 'o', color = 'r', edgecolor = 'b', s = 150)
        
        for label, x1, y1 in zip(node_names, xx, yy):
            plt.annotate(label, xy = (x1, y1), textcoords = "offset points", xytext = (3,7), fontsize = 13, ha = 'center')

        plt.axis('off')
        plt.xlim(-1.5, 1.5) 
        plt.ylim(-1.5, 1.5)     
        plt.show()
        
    def opposite_graph(self):
        node_names = list(self._vertices.keys())
        g = Graph()
        for node in node_names:
            g.add_vertex(node)
        for v1 in node_names:
            v1_neighs = [i.get_node() for i in self._vertices[v1].get_neighbours()]
            for v2 in node_names:
                if v2 not in v1_neighs and v1 != v2:
                    g.add_edge(v1, v2)
       
        return g
            
    def max_clique(self): # finding max complete subgraph (clique) of a graph
        list_clique = []
        list_clique_max = [[i] for i in list(self._vertices.keys())]
        node_names = list(self._vertices.keys())
        flag = 0
        while flag == 0:
            list_clique = list_clique_max
            list_clique_max = []
            for l in list_clique:
                for add in node_names:
                    add_neighs = set([i.get_node() for i in self._vertices[add].get_neighbours()])
                    if (set(l).issubset(add_neighs)) and (add not in l) and (len(add_neighs) != 0):
                        list_clique_max.append(l + [add])
            if list_clique_max == []:
                list_clique_max = list_clique
                flag = 1
                    
        return set(tuple(sorted(set(l))) for l in list_clique_max)
    
    def picnic_company(self):
        return self.opposite_graph().max_clique()
        
def graph_from_matrix(adj_matrix, node_names): # we suppose undirected graph with no loops 
    if adj_matrix.shape[0] != adj_matrix.shape[1]:
        print("Incorrect adjacency matrix dimension")
        return None
    for i in range(adj_matrix.shape[0]):
        for j in range(i + 1, adj_matrix.shape[1]):
            if adj_matrix[i,j] != adj_matrix[j,i]:
                print("Asymmetrical adjacency matrix")
                return None
    for i in range(adj_matrix.shape[0]):
        if adj_matrix[i,i] != np.inf:
            print("Graph with loops")
            return None
    g = Graph()
    n_nodes = adj_matrix.shape[0]
    for node in node_names:
        g.add_vertex(node)
    for node_id in range(n_nodes):
        for neigh_id in range(node_id + 1, n_nodes):
            w = adj_matrix[node_id, neigh_id]
            if w != np.inf: # inf corresponds to no edge
                g.add_edge(node_names[node_id], node_names[neigh_id], weight = w)
    return g    

def generate_graph(n_nodes, node_names = []): # generate adjacency matrix and build graph based on it
    if node_names != []:
        if n_nodes != len(node_names):
            print("The number of graph vertices is incorrectly specified")
            return None
    else:    
        node_names = np.array(range(n_nodes))
    adj_matrix = ma.masked_array(np.random.rand(n_nodes, n_nodes), mask = np.random.randint(0, 2, (n_nodes, n_nodes)))
    adj_matrix = adj_matrix.filled(np.inf)
    for i in range(adj_matrix.shape[0]):
        for j in range(i, adj_matrix.shape[1]):
            adj_matrix[i,j] = adj_matrix[j,i]
    for i in range(adj_matrix.shape[0]):
        adj_matrix[i,i] = np.inf
    g = graph_from_matrix(adj_matrix, node_names)
    
    return g