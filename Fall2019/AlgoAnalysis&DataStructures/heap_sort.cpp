#include<iostream>
using namespace std;

void print_array(int *arr, int len){

for(int i=0;i < len;i++)
    cout << arr[i] << " ";
cout<<endl;
}

void heapify(int *arr, int len, int index){

int biggest = index;
int lchild = 2*index;
int rchild = 2*index + 1;

if(lchild<len && arr[biggest] < arr[lchild])
    biggest = lchild;
if(rchild<len && arr[biggest] < arr[rchild])
    biggest = rchild;

if(biggest != index){
    int temp = arr[biggest];
    arr[biggest] = arr[index];
    arr[index] = temp;

    heapify(arr,len,biggest);
}
}

void buildHeap(int *arr,int len){
    for(int i=len/2-1; i >= 0; i--)
        heapify(arr,len,i);
}

int heapSort(int *arr, int len){
    // Bring the min element one by one to heap
    for (int i=len-1; i>=0; i--)
    {
        int temp = arr[0];
        arr[0] = arr[i];
        arr[i] = temp;

        heapify(arr, i, 0);
    }
}



int main(){
//You are given an array of N positive numbers, in a random order.
int A[] = {22,3,32,5,21,5,24,54,2,6,3,30,32,52,12,100,76,34,87,12,53};
int length = sizeof(A)/sizeof(A[0]);
//cout<<length<<endl;
cout<<"Initial Array: ";
print_array(A,length);
buildHeap(A,length);
cout<<"Array to Heap: ";
print_array(A,length);
heapSort(A,length);
cout<<"After Heap Sort: ";
print_array(A,length);
return 0;
}
