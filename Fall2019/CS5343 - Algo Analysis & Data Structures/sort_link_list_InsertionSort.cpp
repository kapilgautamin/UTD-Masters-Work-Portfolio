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

/* Function to sort the list in ascending order using Insertion Sort
    itr -> The pointer to move through the list
    temp -> The double pointer pointing to the head of the list
    swap_node -> The node in the internal node of algorithm which needs to be 'inserted' in correct order
    check -> The pointer pointing head of the list with which the swap element is compared */
void sort_list(node ** temp){
    node* itr = *temp;
    while(itr->next){
        if(itr->next->val <= itr->val){
            node * check = *temp;
            node * swap_node = itr->next;
            if(swap_node->val < check->val) {        // If the swap_node needs to be inserted at the head of list
                itr->next = swap_node->next;
                swap_node->next = check;
                *temp = swap_node;                  // Make the new first element head of the list
            } else {
                while(check->next->val <= swap_node->val)       // Reach to the point of insertion starting from head of list
                    check = check->next;
                itr->next = swap_node->next;
                swap_node->next = check->next;
                check->next = swap_node;
                if(itr->next && itr->next->val >= itr->val)           // Proceed to the next element only if the next element is in sorted order
                    itr = itr->next;
                if(!itr->next)                          // If the next element is the last element, exit the loop
                    break;
            }
        } else {
            itr = itr->next;
        }
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
