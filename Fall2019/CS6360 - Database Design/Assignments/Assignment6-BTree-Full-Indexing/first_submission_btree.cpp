#include<iostream>
#include<fstream>
#include<cstring>
#include<string>
#define BLOCK_SIZE 1024
#define POINTER_SIZE 8
#define BLOCK_META_DATA_LEN 12
using namespace std;

union charLong
  {
  long long lpart;
  char cpart[8];
  };
charLong cl;

void print_buffer(char *buf){
	cout << "The buffer is: \n";

	for(int i=0;i<BLOCK_SIZE;i++){
		if(buf[i] == '\0' || buf[i] == '\r')
			cout<< ' ';
		else
			cout << buf[i];
	}
	cout<< endl;
}

void padding(char* buf,int filledUpto) {
	for(int i=filledUpto; i<BLOCK_SIZE; i++)
		buf[i] = ' ';
}

void toFile(fstream& file, char* buffer) {
	long before = file.tellp();
	file.write(buffer,BLOCK_SIZE);
	long after = file.tellp();
	cout << /*std::hex <<*/ "Before: " << before << " After: " << after << " " << endl;
}

void shift(char *buf,int from,int keyl){
	int size = BLOCK_SIZE-POINTER_SIZE-keyl-from;
	char buff_copy[size];
	memcpy(buff_copy, &buf[from], size);
	memcpy(&buf[from+POINTER_SIZE+keyl], buff_copy, size);
}

void make_key_pair(char *key_pair,int keylength,string line,int buffer_ptr,int last_starting){
	string key = line.substr(0,keylength);
	//store the location of record in input file
	cl.lpart = last_starting;
	//key_pair is our new record to be inserted
	int key_pair_size = keylength + POINTER_SIZE;
	//char *key_pair = (char*) malloc(sizeof(key_pair_size));
	strcpy(&key_pair[0], key.c_str());
	buffer_ptr += keylength;
	memcpy(&key_pair[buffer_ptr],cl.cpart,POINTER_SIZE);
	cout << "key : " << key_pair << " record address: " << cl.lpart << endl;
	//Key pair created with location, now we can use the cl.lpart to find the position
	//in the bplus tree
}

long long get_root_address(fstream& file_out, int root_location){
	//Read the meta data to get the root address
	file_out.seekg(root_location,ios::beg);
	//cout << "File pointer at: " << file_out.tellp() << endl;
	file_out.read(cl.cpart,POINTER_SIZE);
	return cl.lpart;
}

void init_block_meta_data(char *buffer,fstream& file_out,bool isLeafNode, int block_number){
	//Inititalise meta data for a block
	file_out.seekg(block_number*BLOCK_SIZE,ios::beg);

	if(isLeafNode)
		buffer[0] = 'L';
	else
		buffer[0] = 'N';
	//Initially curr_records are 0
	snprintf(&buffer[1],4,"%3d",0);
	//init block has no next pointer
	cl.lpart = 0;
	memcpy(&buffer[4],cl.cpart,POINTER_SIZE);
	padding(buffer,BLOCK_META_DATA_LEN);
	file_out.write(buffer,BLOCK_SIZE);
	
	//Reinitialise the file pointer to first record location of the current block 
	file_out.seekg(block_number*BLOCK_SIZE+BLOCK_META_DATA_LEN,ios::beg);
	
}

void init_file_meta_data(char *buffer,int keylength, string input_file_name, fstream& file_out){
	char meta_data[BLOCK_SIZE];
	//First meta data block is of 1024 size and then the first node block starts
	int buffer_ptr = 0;
	// Store the keylength in the index file
	// and Root node of bplus tree
	snprintf(&meta_data[0],3 ,"%2d", keylength);
	//meta_data.append(to_string(keylength));
	buffer_ptr += 2;
	
	//Store the input file name in the index file
	strcpy(&meta_data[buffer_ptr],input_file_name.c_str());
	buffer_ptr += input_file_name.length();
	//Initially let the root points to first block at address 1024
	cl.lpart = 1024;
	memcpy(&meta_data[buffer_ptr],cl.cpart,POINTER_SIZE);
	cout << " Meta: " << meta_data << endl;
	//memcpy(cl.cpart,&meta_data[buffer_ptr],8);
	//cout << "root: " << cl.lpart << endl;
	buffer_ptr += POINTER_SIZE;
	file_out.write(meta_data,BLOCK_SIZE);
	padding(meta_data,buffer_ptr);

	//The root is a leaf in the beginning, and its block number is 1
	init_block_meta_data(buffer,file_out,true, 1);
}

bool insert_in_block(char *key_pair,int keylength, char *buffer, int buffer_ptr, int curr_records){
	int key_pair_size = keylength + POINTER_SIZE;
	//We want to check each record in the block and fit our new record in increasing order
	if(curr_records == 0){
		//no record in this leaf now, insert that one directly
		memcpy(&buffer[buffer_ptr],&key_pair[0],key_pair_size);
		cout << "Inserted first node in block" << endl;
		buffer_ptr += key_pair_size;
	} else{
		//should have 1 record to compare and insert
		int i;
		char record[key_pair_size];
		bool inserted = false;
		for(i=1; i <= curr_records && !inserted;i++){
			memcpy(record,&buffer[buffer_ptr],key_pair_size);
			char rec[keylength];
			char pair[keylength];
			int j;
			for(j=0;j<keylength;j++){
				rec[j] = record[j];
				pair[j] = key_pair[j];
			}
			rec[j] = '\0';
			pair[j] = '\0';
			int compare = strcmp(rec, pair);
			cout << "Comparing " << rec << " " << pair << " " << compare << endl;

			if(compare > 0){
				//record is greater, need to insert pair before that
				shift(buffer,BLOCK_META_DATA_LEN + (i-1)*key_pair_size, keylength);
				memcpy(&buffer[buffer_ptr],key_pair,key_pair_size);
				//print_buffer(buffer);
				inserted = true;
			} else if(compare < 0){
				//Check for the next element and insert at the last
			} else {
				//record is same as keypair in leaf node, report a duplicate
				cout << "A Duplicate was found, Skip" << endl;
				//inserted = true;
				return true;
			}
			buffer_ptr += key_pair_size;
		}
		if(inserted == false){
			//record is smaller, need to insert pair after that
			shift(buffer,BLOCK_META_DATA_LEN + i*key_pair_size, keylength);
			memcpy(&buffer[buffer_ptr],key_pair,key_pair_size);
			buffer_ptr += key_pair_size;
			//print_buffer(buffer);
		}
	}
	return false;
}

//INDEX -create <input file> <output file> <key size>
void create_index() {
	fstream file_in,file_out;
	string line;
	streampos beg,ending;
    int keylength = 4;
    string input_file_name = "CS6360Asg6TestDataC.txt";
    string output_file_name = "output.txt";

	file_in.open(input_file_name, ios::binary | ios::in);
	file_out.open(output_file_name,ios::out | ios::in | ios::binary);

	beg = file_in.tellg();
	file_in.seekg(0,ios::end);
	ending = file_in.tellg();
	file_in.seekg(0, ios::beg);

	int file_size = ending - beg;

	// minus 12 => 1 for leaf/non-leaf node, 3 for curr_#_records, 8 for next pointer
	int node_size = (BLOCK_SIZE - 12);
	int max_records = node_size / (keylength + 8);
	int blplus_order = max_records + 1;
	
	//for now change bplus_order to 3, so max 2 keys in it
	blplus_order = 3;
	
	cout << "Size of input file is "  << file_size << " bytes" <<endl;
	cout << "Max records / node: "  << max_records << endl;
    char buffer[BLOCK_SIZE];
	int buffer_ptr = 0;
	int curr_records = 0;
	int block_number = 0;
	int last_starting = file_in.tellp();
	bool isLeafNode = false;
	block_number++;
	init_file_meta_data(buffer,keylength,input_file_name,file_out);

	int root_location = 2 + input_file_name.length();
	int next_pointer_loc = 4;
	while(getline(file_in,line)){
		buffer_ptr = 0;
		isLeafNode = false;
		int key_pair_size = keylength + POINTER_SIZE;
		char key_pair[key_pair_size];
		make_key_pair(key_pair,keylength,line,buffer_ptr,last_starting);
		
		//Assuming root adrress gets updated here during split operation
		int root_address = get_root_address(file_out, root_location);
		cout << "Root at: " << root_address << endl;

		//We read the 1k block
		file_out.seekg(root_address,ios::beg);
		file_out.read(buffer,BLOCK_SIZE);
		//print_buffer(buffer);
		
		//Each block has a info:
		//1 for leaf/non-leaf node, 3 for curr_#_records, 8 for next pointer, 8 for root address
		//If a key is duplicate of another key, don't insert it into b+ tree and output a warning message
		//Keep inserting in a block until it is full,once full find middle of the block, 1024/2 = 512 ,
		// so 512/(keylength+8) both before and after it,Than split it.
		//Also while splitting, we need to store if the block is a leaf or a non-leaf node.

		// Now we need to check where the key pair will go:
		// 1.) Initially, root node will be a leaf and it will store it in increasing order
		// OR 2.) Once the node becomes a parent, then compare which position will it go,
		// and insert it in the leaf block by storing it in increasing order
		if (buffer[0] == 'L' || buffer[0] == 'N'){
			if(buffer[0] == 'L')
				isLeafNode = true;
			else
				isLeafNode = false;
		}


		//Reevaluate the curr_records in this block
		char rec[4];
		rec[0] = buffer[1];
		rec[1] = buffer[2];
		rec[2] = buffer[3];
		rec[3] = '\0';
		curr_records = stoi(rec);
		cout << "Records in this block: " << curr_records << endl;

		//the write address will change to go to the leaf node		
		int write_address = root_address;


		if(!isLeafNode){
			//This is a root node, so we need to go using the key pointers
			cout << "Encountered a non-leaf node" << endl;			
			//We need to decide which leaf node the the incoming pair would go
			char record_key[key_pair_size];
			char pair[keylength];
			buffer_ptr = BLOCK_META_DATA_LEN;
			for(int i=1;i<=curr_records;i++){
				int j;
				memcpy(record_key,&buffer[buffer_ptr],key_pair_size);
				char parent_key[keylength];
				for(j=0;j<keylength;j++) {
					parent_key[j] = record_key[j];
					pair[j] = key_pair[j];
				}
				parent_key[j] = '\0';
				pair[j] = '\0';
				
				int compare = strcmp(parent_key, pair);
				cout << "Comparing " << parent_key << " " << pair << " " << compare << endl;
				//Since the keys are ordered, We will get out position in one pass in that block
				if(compare > 0) {
					//Keypair would be in the left
					cout << "Go left" << endl;
					memcpy(cl.cpart,&record_key[keylength],POINTER_SIZE);
					file_out.seekg(cl.lpart,ios::beg);
					file_out.read(buffer,BLOCK_SIZE);
					//Need to implement above
				} else if(compare < 0){
					//Keypair would be in the right
					cout << "Go Right" << endl;
				} else{
					//Got the keypair, follow the pointer
					cout << "Found the key" << endl;
				}
				buffer_ptr += key_pair_size;
			}



		}


		


		if (curr_records + 1 == blplus_order){
			cout << "Order maxed, Need to split on incoming node" << endl;
			//Make a new node for the half nodes
			char split_node[BLOCK_SIZE];
			int initial_node_blck_num = block_number;
			block_number++;
			int split_node_blck_num = block_number;
			int mid_pt = curr_records / 2;
			int mid_block_record_loc = BLOCK_META_DATA_LEN + mid_pt * key_pair_size;
			init_block_meta_data(split_node,file_out,true,block_number);

			//We need to decide where the incoming node will go, then the order will be figured
			//out by the insert_in_block function
			char record_before_mid[key_pair_size];
			char record_after_mid[key_pair_size];
			memcpy(record_before_mid,&buffer[mid_block_record_loc-key_pair_size],key_pair_size);
			memcpy(record_after_mid,&buffer[mid_block_record_loc],key_pair_size);
			char record_before_mid_key[keylength];
			char record_after_mid_key[keylength];
			char pair[keylength];
			int j;
			for(j=0;j<keylength;j++) {
				record_before_mid_key[j] = record_before_mid[j];
				record_after_mid_key[j] = record_after_mid[j];
				pair[j] = key_pair[j];
			}
			record_before_mid_key[j] = '\0';
			record_after_mid_key[j] = '\0';
			pair[j] = '\0';
			bool isDuplicate = false;
			int left_compare = strcmp(record_before_mid_key, pair);
			int right_compare = strcmp(record_after_mid_key,pair);
			cout << "Comparing left " << record_before_mid_key << " " << pair << " " << left_compare << endl;
			cout << "Comparing right " << record_after_mid_key << " " << pair << " " << right_compare << endl;
			int new_records = mid_pt + 1;
			if(left_compare < 0 && right_compare > 0) {
				//Incoming node in between left and right,Need to insert after mid-point
				cout << "You are given the Center house - Gryffindor" << endl;
				snprintf(&buffer[1],4,"%3d",mid_pt);
				memcpy(&split_node[BLOCK_META_DATA_LEN],&buffer[mid_block_record_loc],BLOCK_SIZE - mid_block_record_loc);
				
				isDuplicate = insert_in_block(key_pair,keylength,split_node,BLOCK_META_DATA_LEN, 1);
				padding(buffer, BLOCK_META_DATA_LEN + mid_pt*key_pair_size);
				snprintf(&split_node[1],4,"%3d",new_records);
			} else if (left_compare > 0 && right_compare > 0) {
				//Incoming node less than left and right,Need to insert before mid-point
				cout << "You are given the First house - Slytherine" << endl;
				//snprintf(&buffer[1],4,"%3d",new_records);
				memcpy(&split_node[BLOCK_META_DATA_LEN],key_pair,key_pair_size);
				
				//isDuplicate = insert_in_block(key_pair,keylength,buffer,BLOCK_META_DATA_LEN, 1);
				padding(split_node, BLOCK_META_DATA_LEN + key_pair_size);
				//padding(buffer,mid_block_record_loc+key_pair_size);
				snprintf(&split_node[1],4,"%3d",1);
			} else if(left_compare < 0 && right_compare < 0){
				//Incoming node is greater than left and right,Need to insert after mid-point
				cout << "You are given the Third house - HufflePuff" << endl;
				snprintf(&buffer[1],4,"%3d",mid_pt);
				memcpy(&split_node[BLOCK_META_DATA_LEN],&buffer[mid_block_record_loc],BLOCK_SIZE - mid_block_record_loc);
				
				isDuplicate = insert_in_block(key_pair,keylength,split_node,BLOCK_META_DATA_LEN, 1);
				padding(buffer, BLOCK_META_DATA_LEN + mid_pt*key_pair_size);
				snprintf(&split_node[1],4,"%3d",new_records);
			}

			//Decide the parent for the current 2 nodes
			if(!isDuplicate){
				char parent_node[BLOCK_SIZE];
				block_number++;
				int parent_node_blck_num = block_number;
				init_block_meta_data(parent_node,file_out,false,block_number);
				if(left_compare < 0 && right_compare > 0){
					//Incoming node in between left and right,Need to insert after mid-point
					//insert_in_block(&split_node[BLOCK_META_DATA_LEN],keylength,&parent_node[BLOCK_META_DATA_LEN],1);
					memcpy(&parent_node[BLOCK_META_DATA_LEN],&split_node[BLOCK_META_DATA_LEN],key_pair_size);
					padding(parent_node,BLOCK_META_DATA_LEN+key_pair_size);
					snprintf(&parent_node[1],4,"%3d",mid_pt);
					
					//Now assign the next pointers
					//The parent has next pointer in meta data, and prev pointer with the key
					cl.lpart = split_node_blck_num*BLOCK_SIZE;
					memcpy(&parent_node[4],cl.cpart,POINTER_SIZE);
					cl.lpart = initial_node_blck_num*BLOCK_SIZE;
					memcpy(&parent_node[BLOCK_META_DATA_LEN+keylength],cl.cpart,POINTER_SIZE);
					//Now assign the next for leaf nodes
					cl.lpart = split_node_blck_num*BLOCK_SIZE;
					memcpy(&buffer[4],cl.cpart,POINTER_SIZE);
				} else if(left_compare > 0 && right_compare > 0){
					//Incoming node less than left and right,Need to insert before mid-point
					//insert_in_block(&split_node[BLOCK_META_DATA_LEN],keylength,&parent_node[BLOCK_META_DATA_LEN],1);
					memcpy(&parent_node[BLOCK_META_DATA_LEN],&buffer[BLOCK_META_DATA_LEN],key_pair_size);
					padding(parent_node,BLOCK_META_DATA_LEN+key_pair_size);
					snprintf(&parent_node[1],4,"%3d",1);
					
					//Now assign the next pointers
					//The parent has next pointer in meta data, and prev pointer with the key
					cl.lpart = split_node_blck_num*BLOCK_SIZE;
					memcpy(&parent_node[4],cl.cpart,POINTER_SIZE);
					cl.lpart = initial_node_blck_num*BLOCK_SIZE;
					memcpy(&parent_node[BLOCK_META_DATA_LEN+keylength],cl.cpart,POINTER_SIZE);
					//Now assign the next for leaf nodes
					cl.lpart = initial_node_blck_num*BLOCK_SIZE;
					memcpy(&split_node[4],cl.cpart,POINTER_SIZE);
				} else if(left_compare < 0 && right_compare < 0){
					//Incoming node is greater than left and right,Need to insert after mid-point
					//insert_in_block(&split_node[BLOCK_META_DATA_LEN],keylength,&parent_node[BLOCK_META_DATA_LEN],1);
					memcpy(&parent_node[BLOCK_META_DATA_LEN],&split_node[BLOCK_META_DATA_LEN],key_pair_size);
					padding(parent_node,BLOCK_META_DATA_LEN+key_pair_size);
					snprintf(&parent_node[1],4,"%3d",mid_pt);
					
					//Now assign the next pointers
					//The parent has next pointer in meta data, and prev pointer with the key
					cl.lpart = split_node_blck_num*BLOCK_SIZE;
					memcpy(&parent_node[4],cl.cpart,POINTER_SIZE);
					cl.lpart = initial_node_blck_num*BLOCK_SIZE;
					memcpy(&parent_node[BLOCK_META_DATA_LEN+keylength],cl.cpart,POINTER_SIZE);
					//Now assign the next for leaf nodes
					cl.lpart = split_node_blck_num*BLOCK_SIZE;
					memcpy(&buffer[4],cl.cpart,POINTER_SIZE);
				}
				print_buffer(buffer);
				print_buffer(split_node);
				print_buffer(parent_node);


				//Change the root pointer location
				file_out.seekg(root_location,ios::beg);
				cl.lpart = block_number*BLOCK_SIZE;
				file_out.write(cl.cpart,POINTER_SIZE);

				file_out.seekg(initial_node_blck_num*BLOCK_SIZE,ios::beg);
				file_out.write(buffer,BLOCK_SIZE);
				file_out.seekg(split_node_blck_num*BLOCK_SIZE,ios::beg);
				file_out.write(split_node,BLOCK_SIZE);
				file_out.seekg(parent_node_blck_num*BLOCK_SIZE,ios::beg);
				file_out.write(parent_node,BLOCK_SIZE);
			} else{
				cout << "Found a duplicate, skip" << endl;
			}
			continue;
		}

		bool isDuplicate = false;
		if(isLeafNode) {
			// char key[keylength];
			// strcpy(key,key_pair);
			isDuplicate = insert_in_block(key_pair,keylength,buffer,BLOCK_META_DATA_LEN,curr_records);
			//If there is a duplicate key_pair then just skip insertion for that key_pair
			if(isDuplicate)
				continue;
		} else{
			//this is a parent node need to go to a leaf node
			int offset = block_number * 1024 + 12 + curr_records*(8);
		}

		//memcpy(&buffer[buffer_ptr],key_pair,key_pair_size);
		curr_records++;
		snprintf(&buffer[1],4,"%3d",curr_records);

		padding(buffer, BLOCK_META_DATA_LEN + curr_records*key_pair_size);

		file_out.seekg(write_address,ios::beg);
		long before = file_out.tellp();
		file_out.write(buffer,BLOCK_SIZE);
		long after = file_out.tellp();
		cout << "Before: " << before << " After: " << after << endl;
		print_buffer(buffer);
		
		last_starting = file_in.tellp();
	}

	file_in.close();
	file_out.close();
	
}
//INDEX -find <index filename> <key>
void find_record() {

}
//INDEX -insert <index filename> "new text line to be inserted." 
void insert_record() {

}
//INDEX -list <index filename> <starting key> <count>
void list_record() {

}


//INDEX -create <input file> <output file> <key size>
//INDEX -find <index filename> <key>
//INDEX -insert <index filename> "new text line to be inserted." 
//INDEX -list <index filename> <starting key> <count>
int main(int argc, char** argv) {

	string index_type;
	index_type = argv[1];
	
	//Check if we have to create the index or list the records
	cout << "Index type is "<< index_type << endl;
	if(index_type == "-create") {
		cout << "Need to do Indexing" << endl;
		string input_file_name = argv[2];
		string output_file_name = argv[3];
		int keylength = stoi(argv[4]);
		cout << "Input file: " << input_file_name << " Output file: " << output_file_name ;
		cout << " Keylength: " << keylength << endl;
		
		//According to the program assignment, the user can enter from 1 to 24 keylength.
		if(keylength < 1 || keylength > 40) {
			cout << "Please enter a keylength between 1 and 40 inclusive" << endl;
			return 0;
		}
		create_index();
	} else if(index_type ==  "-find") {
		cout << "Need to search records based on the indexed data" << endl; 
		string index_file_name = argv[2];
		string key = argv[3];
		cout << "Index file: " << index_file_name << " Key: " << key ;
		find_record();
	} else if(index_type ==  "-insert") {
		cout << "Need to insert records in the indexed data" << endl; 
		string index_file_name = argv[2];
		string insertion_line = argv[3];
		cout << "Index file: " << index_file_name << " Insertion line: " << insertion_line ;
		insert_record();
	} else if(index_type ==  "-list") {
		cout << "Need to list records based on the indexed data" << endl; 
		string index_file_name = argv[2];
		string starting_key = argv[3];
		int count_afterwards = stoi(argv[4]);
		cout << "Index file: " << index_file_name << "Starting Key: " << starting_key << " Count: " << count_afterwards;
		list_record();
	}

	//default
	create_index();
    return 0;
}