#include<iostream>
#define MAX 999999
using namespace std;
//This function finds the vertex with the minimum distance in the
//distances array updated till date and not visited yet
int findMinVertexDistance(int *dist, bool *vis, int n){
    int minVertex = -1;
    for(int i = 0; i < n; i++)
        if(!vis[i] && (minVertex == -1 || dist[i] < dist[minVertex]))
            minVertex = i;
    return minVertex;
}


void dijkstra(int **matrix,int vertices){
//    for(int i=0;i<vertices;i++){
//        for(int j=0;j<vertices;j++)
//            cout<<matrix[i][j]<< " ";
//        cout<<endl;
//    }

    bool *visited = new bool[vertices];
    int *distances = new int[vertices];
    for(int i = 0; i < vertices; i++){
        visited[i] = false;
        distances[i] = MAX;
    }
    distances[0] = 0;
    //Considering 0th vertex as the source vertex

    for(int i = 0; i < vertices - 1; i++){
        int minVertex = findMinVertexDistance(distances, visited, vertices);
        //cout<< i << " " << minVertex << endl;
        //Once the vertex is visited set the visited matrix
        visited[minVertex] = true;
//        cout<< "Visited"<<endl;
//        for(int i=0;i<vertices;i++)
//            cout<<i << " " << visited[i] << endl;
//        cout<<"Visited end"<<endl;
        for(int j=0; j<vertices; j++)
            if(matrix[minVertex][j] != 0 && !visited[j]){
                //Relaxation of edges to the minVertex
                int new_distance = matrix[minVertex][j] + distances[minVertex];
                if(distances[j] > new_distance)
                    distances[j] = new_distance;
            }
    }

    cout << endl<< "The minimum distances from source (0th) are:" << endl;
    cout << "Vertex\tDistance from Source" << endl;
    for(int i=0; i<vertices; i++)
        cout << "   " << i << " \t\t" << distances[i] << endl;
}


int main(){
    int num_vertices = 10;
    int num_edges = 20;
    //We are taking input of the graph as two connecting vertices and the edge weight between them
    int input[num_edges][3] = {
        {0,1,4},
        {0,2,8},
        {0,7,7},
        {0,8,3},
        {0,9,4},
        {1,2,2},
        {1,3,5},
        {1,4,1},
        {1,5,12},
        {1,7,2},
        {2,3,5},
        {2,4,9},
        {3,4,4},
        {3,5,3},
        {3,6,2},
        {3,9,15},
        {4,6,5},
        {5,6,1},
        {7,8,2},
        {8,9,5},
    };

    cout << "The given graph is" <<endl;
    cout << "Vertex 1   Vertex 2   Connecting Edge Weight" << endl;
    for(int i = 0; i < num_edges; i++){
        cout << "   " << input[i][0] << "\t---\t" << input[i][1] << "\t\t" << input[i][2] << endl;
    }


    int **ad_matrix = new int *[num_vertices];
    for(int i = 0; i < num_vertices; i++){
        ad_matrix[i] = new int[num_vertices];
        for(int j = 0;j < num_vertices; j++)
            ad_matrix[i][j] = 0;
    }

    for(int i = 0;i < num_edges; i++){
        int row = input[i][0];
        int col = input[i][1];
        int weight = input[i][2];
        ad_matrix[row][col] = weight;
        ad_matrix[col][row] = weight;
    }

    dijkstra(ad_matrix,num_vertices);
}
