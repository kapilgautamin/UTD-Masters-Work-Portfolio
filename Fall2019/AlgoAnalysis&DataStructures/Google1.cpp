#include<iostream>
using namespace std;

int out(int val){
if(val>=26)
    return val-26;
else
    return val;

}

int main(){
char *str="sqzuq";


for(int i=1;i<=26;i++){
    for(int j=0;j<5;j++)
    cout<<(char)((((str[j]-'a')+i)%26)+'a');
    cout<<endl;
}
return 0;
}
