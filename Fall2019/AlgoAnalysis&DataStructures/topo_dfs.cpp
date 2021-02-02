#include<iostream>
#include <list>
#include <stack>
using namespace std;
/*
Implement the DFS topological ordering.
Your directed graph must have at least 10 vertices and 15 edges.
You must run the algorithm on two sets of graphs.
1. The graph does not have a cycle.  It generates a correct topological ordering.
2. The graph has a cycle. It attempts to generate an order. But discovers the cycle and exits (some nodes will already be ordered.
Submit the code.
Submit screen shots of each execution.
You can print each of the graphs as adjacency matrix or other representation.
*/
class Graph
{
	int vertices;
	list<int> *adj;    //STL library to create a adjacency list for a vertex
	bool DFS(int v, bool visited[], bool completed[], stack<int> &order);

public:
	Graph(int vertices); // Constructor
	void addEdge(int v, int w);
    void print_graph(int vertices);
	void topologicalSort();
};

//Constructor for the Graph class
Graph::Graph(int vertices)
{
	this->vertices = vertices;
	adj = new list<int>[vertices];
}

//This functions adds the edge to the given vertex
void Graph::addEdge(int v, int w)
{
	adj[v].push_back(w); // Add w to v’s list.
}

//This function prints the graph with its adjacency matrix
void Graph::print_graph(int vertices){
    list<int>::iterator i;
    for (int v = 0; v < vertices; v++) {
        if (adj[v].size() > 0) {
            cout << "Adjacency list for " << v ;
            for (i = adj[v].begin(); i != adj[v].end(); ++i){
                cout << " -> " << *i ;
            }
            cout << endl;
        }
    }
}

//Depth first search function for exploring the nodes
//If the node is already visited but not completed, then a cycle exists in the given DAG
//If the node is already visited and completed too, go to the next node in adjacency list
bool Graph::DFS(int v, bool visited[], bool completed[], stack<int> &order)
{
	// Mark the current node as visited.
    bool cycle = false;
	visited[v] = true;

	// DFS for all the vertices adjacent to this vertex
	list<int>::iterator i;
	for (i = adj[v].begin(); i != adj[v].end(); ++i) {
        if (visited[*i] && completed[*i] == false)
            return true;
		if (!visited[*i] && cycle == false)
			cycle = DFS(*i, visited, completed, order);
    }

	// Push current vertex to stack and mark the node to completed
    completed[v] = true;
	order.push(v);
    return cycle;
}

void Graph::topologicalSort()
{
	stack<int> order;
	bool cycle = false;
	bool *visited = new bool[vertices];
    bool *completed = new bool[vertices];
    //Initialise the visited and completed boolean arrays
	for (int i = 0; i < vertices; i++) {
		visited[i] = false;
        completed[i] = false;
    }

	// Visit all the nodes in the graph
	for (int i = 0; i < vertices; i++)
	    if (visited[i] == false && cycle == false)
		    cycle = DFS(i, visited, completed, order);

    cout << endl;
    // Print contents of order stack untill now
    while (order.empty() == false)
    {
        cout << order.top() << " ";
        order.pop();
    }

    if(cycle){
        cout  << "\nCycle detected, Exiting Program\n\n" ;
    } else{
        cout  << "\nAbove is a Topological Sort of the given graph\n\n" ;
    }
}

// Driver program to test above functions
int main()
{
	// Creating a graph with no cycles
    int v = 10;
	Graph g1(v);
	g1.addEdge(0, 1);
	g1.addEdge(0, 6);
	g1.addEdge(0, 5);
	g1.addEdge(0, 7);
	g1.addEdge(1, 2);
	g1.addEdge(3, 1);
    g1.addEdge(3, 4);
    g1.addEdge(4, 1);
    g1.addEdge(4, 8);
    g1.addEdge(5, 3);
    g1.addEdge(5, 4);
    g1.addEdge(6, 1);
    g1.addEdge(7, 4);
    g1.addEdge(7, 9);
    g1.addEdge(9, 4);
    g1.print_graph(v);
	g1.topologicalSort();

    cout << "----------------------------------------------------" << endl;
    //Creating a graph having cycle
    //2 -> 5 -> 11 -> 9 -> 6 -> 2 forms a cycle
    v = 12;
    Graph g2(v);
	g2.addEdge(0, 1);
	g2.addEdge(0, 2);
	g2.addEdge(0, 3);
	g2.addEdge(1, 2);
	g2.addEdge(2, 5);
    g2.addEdge(3, 4);
    g2.addEdge(3, 5);
    g2.addEdge(5, 10);
    g2.addEdge(5, 11);
    g2.addEdge(6, 2);
    g2.addEdge(6, 10);
    g2.addEdge(7, 6);
    g2.addEdge(8, 6);
    g2.addEdge(8, 9);
    g2.addEdge(9, 6);
    g2.addEdge(11, 9);
    g2.addEdge(11, 10);
    g2.print_graph(v);
	g2.topologicalSort();

	return 0;
}
