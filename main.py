import matplotlib.pyplot as plt
import networkx as nx
from itertools import combinations
from numpy import random
import time

N = 5
p = 0.7


class Hamilton:
    pathFound = False
    hamiltonPath = []

    def __init__(self, graph):
        self.G = graph

    def findPath(self, v, visited=None):
        if visited is None:
            visited = []
        if v in visited:
            return None
        visited.append(v)

        neighbours = [n for n in self.G.neighbors(v)]
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


class Point:
    def __init__(self, v):
        self.V = v


class NearestNeighbour:
    def __init__(self, graph):
        self.graph = graph
        self.visited = []
        self.weights = []

    def findPath(self, start):
        self.visited.append(start)

        neighbours = [n for n in self.graph.neighbors(start)]
        minD = 1000
        n = None
        for neighbour in neighbours:
            if neighbour not in self.visited:
                d = (wg.edges[start, neighbour].get("weight"))
                if d < minD:
                    minD = d
                    n = neighbour
        if n:
            self.weights.append(minD)
            self.findPath(n)
        else:

            print(f"Znaleziona droga = {self.visited}")
            print(f"Znalezione wagi = {self.weights}")
            return self.visited

        return None


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
        g.add_edges_from([(0, 1), (0, 2), (0, 3), (1, 2), (1, 4), (1, 3), (2, 4), (4, 3)])
        return g

    def createGraphWithWeights(self, n):
        g = nx.Graph()

        V = set([v for v in range(n)])
        E = set()
        g.add_nodes_from(V)
        for combination in combinations(V, 2):
            g.add_edge(combination[0], combination[1], weight=random.randint(100))

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
    print(f"--- Czas wykonania programu dla N = {N}, i p = {p}, jest równy: {(time.time() - start_time)} sekund ---")
    nx.draw_networkx(G)
    plt.title("Cykl Hamiltona")
    plt.show()

    # Graph 2
    wg = graph.createGraphWithWeights(N)

    print(f" Wszystkie punkty z wagami: {wg.edges(data='weight')}")

    nei = NearestNeighbour(wg)
    result2 = nei.findPath(0)
    pos = nx.spring_layout(wg)
    nx.draw_networkx(wg, pos)
    labels = nx.get_edge_attributes(wg, 'weight')
    nx.draw_networkx_edge_labels(wg, pos, edge_labels=labels)

    plt.title("Najbliższy sąsiad")
    plt.show()
