import matplotlib.pyplot as plt
import networkx as nx
from itertools import combinations
from numpy import random
import time

N = 300
p = 0.7


class Hamilton:
    pathFound = False
    hamiltonPath = []

    def __init__(self, graph):
        self.graph = graph

    def findPath(self, v, visited=None):
        if visited is None:
            visited = []
        if v in visited:
            return None
        visited.append(v)
        neighbours = [n for n in G.neighbors(v)]
        if not self.pathFound:
            if len(visited) == N:
                for neighbour in neighbours:
                    if neighbour == visited[0]:
                        self.pathFound = True
                        self.hamiltonPath = visited
                        return visited
                return None
            else:
                for neighbour in neighbours:
                    if self.pathFound:
                        return self.hamiltonPath
                    self.findPath(neighbour, visited.copy())
        else:
            return self.hamiltonPath
        return None


class NearestNeighbour:
    def __init__(self, graph):
        self.graph = graph

    def findPath(self, graph):
        visited = [0]
        graph.edges(data=True)
        edges = graph.edges()
        weights = []
        U = []
        V = []
        for x, y in edges:
            U.append(x)
            V.append(y)
            weights.append(graph[x][y]['weight'])
        path = [0]
        print(f"Wagi: {weights}")
        for element in range(N-1):
            lowestWeiVal = 1000
            lowestWeiInd = 1000

            for weight in weights:
                if (weight < lowestWeiVal) and (weights.index(weight) not in visited):
                    lowestWeiVal = weight
                    lowestWeiInd = weights.index(weight)

            if lowestWeiInd % 2 == 0:
                visited.append(lowestWeiInd)
                visited.append(lowestWeiInd+1)
                number = lowestWeiInd / 2
                path.append(int(number))
            else:
                visited.append(lowestWeiInd)
                visited.append(lowestWeiInd - 1)
                number = (lowestWeiInd-1)/2
                path.append(int(number))

        return path


class Graph:
    def createGraph(self, n, p):
        V = set([v for v in range(n)])
        E = set()
        for combination in combinations(V, 2):
            a = random.rand()
            if a < p:
                E.add(combination)

        g = nx.Graph()
        g.add_nodes_from(V)
        g.add_edges_from(E)

        return g

    def createCustomGraph(self):
        g = nx.Graph()
        g.add_nodes_from([0, 1, 2, 3, 4])
        g.add_edges_from([(0, 1), (0, 2), (0, 3), (1, 2), (1,4), (1,3), (2,4), (4,3)])
        return g

    def createGraphWithWeights(self, n):
        g = nx.Graph()
        V = set([v for v in range(n)])
        E = set()
        g.add_nodes_from(V)
        for combination in combinations(V, 2):
            g.add_edge(combination[0],combination[1], weight=random.randint(100))

        g.add_edges_from(E)

        return g


if __name__ == '__main__':
    start_time = time.time()
    graph = Graph()

    # Graph 1
    G = graph.createGraph(N, p)
    hamilton = Hamilton(G)
    for counter in range(N):
        result = hamilton.findPath(0)
        if result is not None:
            break

    if result is not None:
        print(f"Odnaleziony cykl Hamiltona: {result}")
    else:
        print("Brak sciezki Hamiltona")


    pos = nx.spring_layout(G)
    # nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)
    print(f"--- Czas wykonania programu dla N = {N}, i p = {p}, jest równy: {(time.time() - start_time)} sekund ---")
    nx.draw_networkx(G)
    plt.title("Cykl Hamiltona")
    plt.show()


    # Graph 2
    # wg = graph.createGraphWithWeights(N)
    # nei = NearestNeighbour(wg)
    # result2 = nei.findPath(wg)
    # print(f"Odnaleziona droga: {result2}")
    # pos = nx.spring_layout(wg)  # pos = nx.nx_agraph.graphviz_layout(G)
    # nx.draw_networkx(wg, pos)
    # labels = nx.get_edge_attributes(wg, 'weight')
    # nx.draw_networkx_edge_labels(wg, pos, edge_labels=labels)

    # plt.title("Najbliższy sąsiad")
    # plt.show()
