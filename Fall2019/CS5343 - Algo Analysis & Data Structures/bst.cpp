#include<iostream>
using namespace std;

typedef struct t{
    t* left;
    t* right;
    int val;
}tree;

/*
This function searches for a node with the given value and returns the node.
*/
tree* searchNode(tree* temp, int val){
    if(temp == 0)
        return 0;
    if(temp->val == val) {
        return temp;
    } else if(temp->val < val)
        searchNode(temp->right, val);
    else searchNode(temp->left, val);
}

/*
This function finds the predecessor(Maximum node value on the left side) of the given node.
*/
tree* findPredecessor(tree *temp){

tree* parent;
if(temp->left){
    parent = temp;
    temp = temp->left;
}

while(temp->right){
        parent = temp;
        temp = temp->right;
}

if(parent->right)   //actual node will only have a left child
    parent->right = temp->left;

return temp;
}

/*
This function performs a inOrderTraversal of the BST.
*/
void inorderTraversal(tree *temp){
if(temp != 0){
    inorderTraversal(temp->left);
    cout<<temp->val<< " ";
    inorderTraversal(temp->right);
}
}

/*
This function inserts a new node with the given value in BST.
*/
tree* insertBST(tree* temp,int val){
if(temp == 0){
    tree * new_node = new tree();
    new_node->val = val;
    new_node->left = 0;
    new_node->right = 0;
    //cout<<"came here " << new_node->val<<endl;
    return new_node;
} else{
    if(temp->val > val)
        temp->left = insertBST(temp->left, val);
    else
        temp->right = insertBST(temp->right, val);
}
return temp;
}

/*
This function inserts multiple nodes using the insertBST function.
*/
void insert_multiple_values(tree* main){
insertBST(main, 50);
insertBST(main, 200);
insertBST(main, 150);
insertBST(main, 300);
insertBST(main, 25);
insertBST(main, 75);
insertBST(main, 12);
insertBST(main, 37);
insertBST(main, 125);
insertBST(main, 175);
insertBST(main, 250);
insertBST(main, 320);
insertBST(main, 67);
insertBST(main, 87);
insertBST(main, 94);
insertBST(main, 89);
insertBST(main, 92);
insertBST(main, 88);
}

int main(){
tree* root = 0;
root = insertBST(root, 100);
insert_multiple_values(root);
cout<< "Inorder traversal after insertion:" << endl;
inorderTraversal(root);
cout<<endl;

int val_to_delete = 100;
tree* deleteNode = searchNode(root, val_to_delete);

tree* replaceNode;
replaceNode = findPredecessor(deleteNode);

//cout << replaceNode->val << endl;
// node child changed in findPredecessor function
deleteNode->val = replaceNode->val;
cout<< "Inorder traversal after deletion of 100:" << endl;
inorderTraversal(root);
cout<<endl;

return 0;
}
