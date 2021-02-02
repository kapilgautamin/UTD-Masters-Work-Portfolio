#include<iostream>
#include<fstream>
#include<cstring>
#include<map>
#include<windows.h>
using namespace std;

/*
Created by Kapil Gautam
kapil.gautam@utdallas.edu
This program is to be used as CLI as it requires command line arguments.
Compile using : g++ .\index_single.cpp -o INDEX
Use by :
For creating index: 
> ./INDEX -c CS6360Asg5TestDataA.txt DataA.idx 15
To output sorted index from index file to console:
> ./INDEX -l CS6360Asg5TestDataA.txt DataA.idx 15
(We assume the files are being used and created in the same directory of the execution of program)
*/

//This union is used to read/write interchangeably between a character array and long long.
union charLong
  {
  long long lpart;
  char cpart[8];
  };
charLong cl;

//This function is being used to create a index file based on a keylength and input file.
//It also appends the keylength and input file name in index file for verification purposes,
// so that the records are shown for the correct input file
// Additionaly, we are making the index file, only readable, so that it cannot be edited.
void create_index(string input_file_name, string output_file_name,int keylength){
	
	fstream file_in;
	string line;
	streampos beg,ending;
	file_in.open(input_file_name, ios::binary | ios::in);
	beg = file_in.tellg();
	file_in.seekg(0,ios::end);
	ending = file_in.tellg();
	cout << "Size of file is "  << ending - beg << endl;
    file_in.seekg(0, ios::beg);
    
	int pointer_size = 8;

	multimap<string,int> index;
	//Read each line of the file and store it in a mult-map, which allows dupicates and stores them
	//in sorted order
	int last_starting_index = file_in.tellp();
	while(getline(file_in,line)){
		index.insert(make_pair(line.substr(0,keylength),last_starting_index));
		last_starting_index = (file_in.tellp());
        if(file_in.eof())
            cout<<"Reached end"<<endl;
	}

	file_in.close();
	//Since we are setting the Read only attribute to make the file non-editable 
	//the first time, we need to remove it, if we are trying to re-write the same file again,
	//and then re-create the same file again.
	SetFileAttributes( output_file_name.c_str(),  
                   GetFileAttributes(output_file_name.c_str()) & ~FILE_ATTRIBUTE_READONLY);
	remove(output_file_name.c_str());

	fstream file;
    file.open(output_file_name,ios::out | ios::binary);
	char buffer[keylength+8];
	int buffer_ptr = 0;
	//Store the keylength in the index file
	buffer[0] = keylength;
	file.write(buffer,1);
	//Store the input file name in the index file
	strcpy(buffer,input_file_name.c_str());
	file.write(buffer,input_file_name.length());

	//Iterate through the multi-map in memory and output them to the index file using binary file I/O.
	multimap<string,int> :: iterator it;
	int count = 0;
    for(it=index.begin();it!=index.end();it++){
        //cout << (*it).first << "[" << (*it).second << "]" << endl;
		buffer_ptr = 0;
		strcpy(buffer,(*it).first.c_str());
		buffer_ptr += keylength;
		//Store it in long long part of union
		cl.lpart = (*it).second;
		//Write in file using the character part of the union
		memcpy(&buffer[buffer_ptr],cl.cpart,pointer_size);
		file.write(buffer,keylength+8);
		//Just counting the number of records for addtional statistics
		count++;
    }
	file.close();
	//Make the file non-editable, only readable
	
	string read_only = "attrib +R " + output_file_name;
	system(read_only.c_str());
	cout << "Processed " << count << " records" << endl;
	cout << "Created index file " << output_file_name <<" with keylength: " << keylength << endl;
}

//This function is being used to output on console the list of records in sorted order, using the 
//index file from the create_index file. It confirms the keylength and inputfile name
//while giving the output, so that the program does'nt behave weirdly, and act on the correct
//index file only.
void create_list(string input_file_name, string index_file_name, int keylength){
	int pointer_size = 8;
	char buffer_dense[1024];
	char buffer[keylength+8];
	string line;
	fstream file_in;
	fstream file_read;

	file_in.open(input_file_name, ios::binary | ios::in);
	file_read.open(index_file_name, ios::binary | ios::in);
	memset(buffer,0,keylength+pointer_size);
	//Read from the index file if the keylength of given command is same as the index file keylength.
	file_read.read(buffer,1);
	if(buffer[0] != keylength){
		cout << endl << "The index file keylength is different from current entered keylength"<< endl;
		cout << "Please use the same keylength used for indexing" << endl;
		return;
	} 
	//Read from the index file if the inputfile for creating the index file was same as the one given
	//in the current command inputfile name.
	file_read.read(buffer_dense,input_file_name.length());
	buffer_dense[input_file_name.length()] = '\0';
	if (buffer_dense != input_file_name){
		cout<< buffer_dense << " " << input_file_name <<endl;
		cout << "Please use the same text file which was used for indexing" << endl;
		return;
	}

	//Now one by one read 1024bytes from the file each time and display them using the index from the 
	//index file name.
	int count = 0;
	while(file_read.read(buffer,keylength+pointer_size)){
		if(file_read.eof())
			break;
		memcpy(cl.cpart, &buffer[keylength], pointer_size);
		file_in.seekg(cl.lpart, ios::beg);
		getline(file_in,line);
		strcpy(buffer_dense,line.c_str());
		buffer_dense[line.length()]='\0';
		cout<< buffer_dense <<endl;
		//Count the records outputted from the console for additional statistics.
		count++;
	}

	file_read.close();
	file_in.close();
	cout << "All " << count << " records processed" << endl;
}

int main(int argc, char** argv) {

	string index_type;
	index_type = argv[1];
	bool display_sorted = false;
	//Check if we have to create the index or list the records
	cout<<"Index type is "<< index_type << endl;
	if(index_type == "-c")
		cout << "Need to do Indexing" << endl;
	else if(index_type ==  "-l"){
		display_sorted = true;
		cout << "Need to do list records based on the indexed data" << endl; 
		}

	string input_file_name = argv[2];
	string output_file_name = argv[3];
	//input_file_name = "CS6360Asg5TestDataA.txt";
	//output_file_name = "output.txt";
	//display_sorted = true;
	int keylength = stoi(argv[4]);
	cout << "Input file: " << input_file_name << " Output file: " << output_file_name ;
	cout << " Keylength: " << keylength << endl;
	
	//According to the program assignment, the user can enter from 1 to 24 keylength.
	if(keylength < 1 || keylength >24){
		cout << "Please enter a keylength between 1 and 24 inclusive";
		return 0;
	}

	if(display_sorted) {
		create_list(input_file_name,output_file_name,keylength);
	} else{
		create_index(input_file_name,output_file_name,keylength);
	}

	return 0;
}