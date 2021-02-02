#include<iostream>
using namespace std;

typedef struct s{
    int val;
    s* next;
} node;

/* Function to insert a new node */
void insert_node(node **tmp, int val){
    node* new_node = new node();
    new_node->val = val;
    new_node->next = *tmp;
    *tmp = new_node;
}

/* Function to traverse the linked list */
void print_list(node* tmp){
    while(tmp) {
        cout << tmp->val << " ";
        tmp = tmp->next;
    }
    cout << endl;
}

/* Function to insert multiple nodes
 Assuming the nodes values are integers only
 For new node addition we are adding to the head of list*/
void insert_multiple_nodes(node ** tmp){
    insert_node(tmp,7);
    insert_node(tmp,1);
    insert_node(tmp,2);
    insert_node(tmp,10);
    insert_node(tmp,2);
    insert_node(tmp,31);
    insert_node(tmp,13);
    insert_node(tmp,1);
    insert_node(tmp,24);
    insert_node(tmp,-12);
    insert_node(tmp,32);
    insert_node(tmp,0);
    insert_node(tmp,-14);
    insert_node(tmp,10);
    insert_node(tmp,222);
    insert_node(tmp,-222);
    insert_node(tmp,-12);
    insert_node(tmp,32);
    insert_node(tmp,22);
    insert_node(tmp,-1234);
    insert_node(tmp,234);
    insert_node(tmp,34);
}

/* Function to sort the list in ascending order using Selection Sort
    min_node -> The node with minimum value in a traversal
    itr1,itr2 -> The iterating pointers for traversals
    prev_min_node -> The node previous to the node having minimum value in a traversal
    prev_itr1_node -> The node previous to the itr1 node in a traversal */
void sort_list(node ** start){
    node* itr1 = *start;

    while(itr1){            //Iterate through all elements
        node* min_node = itr1;
        node* itr2 = itr1->next;
        node* prev_min_node = itr1;

        while(itr2){
            if(itr2->val < min_node->val)
                min_node = itr2;            //Find the minimum from the list
            itr2 = itr2->next;
        }

        if(itr1->val != min_node->val){
            while(prev_min_node->next->val != min_node->val)
                prev_min_node = prev_min_node->next;        //Get the node previous to minimum node

            if(itr1->val == (*start)->val){         //In case the swap of nodes happen on first node.
                prev_min_node->next = itr1;
                node* temp = itr1->next;
                itr1->next = min_node->next;
                min_node->next = temp;
                itr1 = min_node;
                *start = itr1;
            } else{
                node* prev_itr1_node = *start;
                while(prev_itr1_node->next->val != itr1->val)
                    prev_itr1_node = prev_itr1_node->next;
                prev_min_node->next = itr1;
                prev_itr1_node->next = min_node;
                node* temp = itr1->next;
                itr1->next = min_node->next;
                min_node->next = temp;
                itr1 = min_node;
            }
        }
        itr1 = itr1->next;
    }

}

/* Main function */
int main(){
node * head = new node();
head->val = 5;
head->next = 0;

cout<< "Before Sort" << endl;
insert_multiple_nodes(&head);
print_list(head);

cout<< "After Sort" << endl;
sort_list(&head);
print_list(head);

return 0;
}
