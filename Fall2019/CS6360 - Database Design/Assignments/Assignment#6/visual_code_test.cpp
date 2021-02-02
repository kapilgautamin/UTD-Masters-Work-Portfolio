#include<iostream>
#include<fstream>
#include<map>
#include<cstring>
using namespace std;

int main(int argc, char** argv){
    // map<string,int> index;
    // index.insert(make_pair("AAAA",15));
    // index.insert(make_pair("ZZZZ",39));
    // index.insert(make_pair("CCCC",12));
    // index.insert(make_pair("MMMM",43));

    // map<string,int> :: iterator it;
    // for(it=index.begin();it!=index.end();it++){
    //     cout << (*it).first << " " << (*it).second << endl;
    // }
    cout << "You have entered " << argc 
         << " arguments:" << "\n"; 
  
    for (int i = 0; i < argc; ++i) 
        cout << argv[i] << "\n"; 

    fstream f;
    char buffer[20];
    f.open("testing.txt", ios::out);
    strcpy(buffer,"hello");
    f.write(buffer,strlen("hello"));
    strcpy(buffer,"this");
    f.write(buffer,strlen("this"));
    strcpy(buffer,"is for testing purpose");
    f.write(buffer,strlen("is for testing purpose"));
    f.close();


    return 0;
}