#include<iostream>
#include<fstream>
#include<cstring>
using namespace std;
//Iniital size is 53 words
int table_size = 53;
string *hasht;
bool *check;
/*
The program first computes the position of the strings from a file using linear probing 
and then quadratic probing and lists the number of collisions each time.
It then asks for a word to be found in the hash table.
To compile the program :
> g++ hash_words.cpp -o hashing
> ./hashing
*/

//A simple hash function which is used to compute the position of a string in the hash table
int hash_function(string str){
    int position = 0;
    int len = 0;
    while(str[len] != '\0'){
        position += (str[len]-'a')%table_size;
        len++;
    }
    //cout << "Position for " << str << " is " << position % table_size << endl;
    return position % table_size;
}

//This function finds the word in a hash table using the quadratic probing.
bool find_word(string search){
    int loc = hash_function(search);
    bool found = false;
    //cout << search << " " << loc << " " << hasht[loc] <<endl;
    if(strcmp(search.c_str(),hasht[loc].c_str()) == 0){
        found = true;
    }
    if(!found){
        int quad_probe_iter = 0;
        while(check[loc]){
            quad_probe_iter++;
            loc = (loc+quad_probe_iter * quad_probe_iter) % table_size;
            if(strcmp(search.c_str(),hasht[loc].c_str()) == 0){
                found = true;
                break;
            }
            //cout << search << " " << loc << " " << hasht[loc] <<endl;
        }
    }
    return found;
}

//This function is used to rehash the previous values of the hash table using the new table size
int rehash(bool quad_probe){
    //cout << "Rehashing now with table size: " << table_size << endl;
    int rehash_collisions = 0;
    string *copy_hash = new string[table_size/2];
    bool *copy_check = new bool[table_size/2];
    for(int i=0;i<table_size/2;i++){
        copy_hash[i] = hasht[i];
        copy_check[i] = check[i];
    }
    hasht = new string[table_size];
    check = new bool[table_size];
    for(int i=0;i<table_size;i++){
        hasht[i] = "";
        check[i] = false;
    }
    int new_loc;

    for(int i=0;i<table_size/2;i++){
        if(copy_check[i]){
            new_loc = hash_function(copy_hash[i]);
            int quad_probe_iter = 0;
            while(check[new_loc]){
                //cout << "Collision during rehashing " << copy_hash[i] << " at loc: " << new_loc << endl;
                if(quad_probe){
                    quad_probe_iter++;
                    new_loc = (new_loc + quad_probe_iter*quad_probe_iter) % table_size;
                }else
                    new_loc = (new_loc+1)%table_size;
                rehash_collisions++;
            }
            hasht[new_loc] = copy_hash[i];
            check[new_loc] = true;
        }
    }
    return rehash_collisions;
}


//This function is used to hash the strings inputted from a file using a hash function
//using either the linear probing or the quadratic probing.
int hash_words(ifstream& file,bool quad_probe){
    int collisions = 0;
    int loc;
    int count_rec = 0;
    float load;
    string line;

    while(getline(file,line)){
        //cout << line << endl;
        loc = hash_function(line);
        int quad_probe_iter = 0;
        while(check[loc]){
            //cout << "Collision for " << line << " at loc: " << loc << endl;
            if(quad_probe){
                quad_probe_iter++;
                loc = (loc + quad_probe_iter*quad_probe_iter) % table_size;
            } else
                loc = (loc + 1) % table_size;
            collisions++;
        }
        check[loc] = true;
        hasht[loc] = line;
        count_rec++;
        load = (float)count_rec / table_size;
        if(load > 0.5){
            cout << "Load factor reached for tablesize: " << table_size << " at " << collisions << " collisions" << endl;
            table_size *= 2;
            cout << "Increasing Table size to " << table_size << endl;
            collisions += rehash(quad_probe);
        }
    }
    return collisions;
}

//The program first gives the number of collisions from linear and quadratic probing and
//then interactively asks the user to input a string , which is then a yes/no depending
//if the string is available in the hash table.
int main(){
    ifstream file;
    file.open("random_words.txt");
    hasht = new string[table_size];
    //true is for filled, false is for empty
    check = new bool[table_size];
    for(int i=0;i<table_size;i++){
        hasht[i] = "";
        check[i] = false;
    }
    bool quadratic_probe = false;
    int collision = hash_words(file,quadratic_probe);
    cout << "Final table size: " << table_size << " and linear probing collisions: " << collision << endl;
    file.close();

    cout << endl;
    table_size = 53;
    hasht = new string[table_size];
    //true is for filled, false is for empty
    check = new bool[table_size];
    for(int i=0;i<table_size;i++){
        hasht[i] = "";
        check[i] = false;
    }
    quadratic_probe = true;
    file.open("random_words.txt");
    collision = hash_words(file,quadratic_probe);
    cout << "Final table size: " << table_size << " and quadratic probing collisions: " << collision << endl;
    file.close();

    while(1){
        string word;
        cout << endl << "Enter a word:" << endl;
        cin >> word;
        if(word.length() == 0)
            break;
        if(find_word(word))
            cout << "Found " <<  word << " in hash table" << endl;
        else
            cout << word << " not found in hash table" << endl;

    }
    return 0;
}