#Search strategies implementation taken from AIMA code
from search import *

##Overriding the GraphProblem class of AIMA code
class Road_Problem(Problem):
    """The problem of searching a graph from one node to another."""

    def __init__(self, initial, goal, graph):
        super().__init__(initial, goal)
        self.graph = graph
        self.dist = dict(
            Austin = 182,
            Charlotte = 929,
            SanFransisco = 1230,
            LosAngeles = 1100,
            NewYork = 1368,
            Chicago = 800,
            Seattle = 1670,
            SantaFe = 560,
            Bakersville = 1282,
            Boston = 1551,
            Dallas = 0
        )

    def actions(self, A):
        """The actions at a graph node are just its neighbors."""
        return list(self.graph.get(A).keys())

    def result(self, state, action):
        """The result of going to a neighbor is just that neighbor."""
        return action

    def path_cost(self, cost_so_far, A, action, B):
        #consistant heuristics test by triangle inequality
        #h(n') <= cost(n,n') + h(n)
        print("Is consistent heuristic", self.graph.get(A, B) + self.dist[B] >= self.dist[A])
        return cost_so_far + (self.graph.get(A, B) or np.inf)

    def find_min_edge(self):
        """Find minimum value of edges."""
        m = np.inf
        for d in self.graph.graph_dict.values():
            local_min = min(d.values())
            m = min(m, local_min)
        return m

    def h(self, node):
        """h function is straight-line distance from a node's state to goal."""
        return self.dist[node.state]

#Initial graph
road_map = UndirectedGraph(dict(
    LosAngeles=dict(SanFransisco=383, Austin=1377, Bakersville=153),
    SanFransisco=dict(Bakersville=283, Seattle=807),
    Boston=dict(Austin=1963, Chicago=983, SanFransisco=3095),
    Seattle=dict(SantaFe=1463, Chicago=2064),
    Bakersville=dict(SantaFe=864),
    Austin=dict(Dallas=195, Charlotte=1200),
    SantaFe=dict(Dallas=640),
    Dallas=dict(NewYork=1548),
    Chicago=dict(SantaFe=1272),
    NewYork=dict(Boston=225),
    Charlotte=dict(NewYork=634)))

# print(road_map.nodes())

seattle_to_dallas=Road_Problem('Seattle', 'Dallas', road_map)

print("A* search")
s=astar_search(seattle_to_dallas,seattle_to_dallas.h)
print(s.path())


print("\nRecursive best-first search")
s=recursive_best_first_search(seattle_to_dallas,seattle_to_dallas.h)
print(s.path())