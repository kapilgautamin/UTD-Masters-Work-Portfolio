#include<iostream>
#include<fstream>
#include<cstring>
#include<string>
#include<stdio.h>
#include <sstream>
#define BLOCK_SIZE 1024
#define POINTER_SIZE 8
#define BLOCK_META_DATA_LEN 23
using namespace std;
/*
Created by Kapil Gautam - KXG180032 - Assignment#6 - CS6360 - Database Design
The program uses Binary Plus tree to store the records given in the file from the text input file.
The program works dynamically with the given keylength and the input text file.
It uses meta data which is stored in the starting of a 1K block to store important
information such as keylength, inputfilename and the global root address.
The BPlus tree can also split and grow as required by the incoming records,
Also, the program detects the duplicate values and outputs on the console, and we can insert
a new record too in the bplus tree.
The program at a time uses upto 3 1K blocks to split the data when needed,
to write the blocks in the file.
It can be used as following CLI input/output

Compile the program by:
> g++ index_btree.cpp -o INDEX

Create the index file by the following command:
>./INDEX -create CS6360Asg6TestData.txt index.indx 15
The program accepts keylength from 1 to 40 length

To find a record in btree with a given key:
>./INDEX -find index.indx 45526813100142A
>./INDEX -find index.indx 93288157045562A
>./INDEX -find index.indx 12222222222222C
It will return the key if it is found in the bplus_tree

To insert a record in the file:
>./INDEX -insert index.indx "12222222222222C test data I added"
If the record is already in bplus_tree it is not inserted in text file, neither in index file.
If it is not present in bplus_tree than it is added in the text file as well as the index file.
If we insert the same record it will return that it is a duplicate record

To list the number of records after the given key(including that) 
>./INDEX -list index.indx 38417813544394A 12
If the key is not found in the bplus_tree it lists the next key for the given count

The script given in the assignment works perfectly with the program.
*/


union charLong
  {
  long long lpart;
  char cpart[8];
  };
charLong cl;
int block_number;
bool DEBUG = false;

void print_buffer(char *buf, string name){
	cout << "The " << name << " buffer is: " << endl;

	for(int i=0;i<BLOCK_SIZE;i++){
		if(buf[i] == '\0' || buf[i] == '\r')
			cout<< ' ';
		else
			cout << buf[i];
	}
	cout<< endl;
}

void print_blocks(fstream& file_out){
	cout << "Printing the blocks " << endl;
	char buffer[1024];
	file_out.seekg(0,ios::beg);

	for(int i=0;i<=block_number;i++){
		cout<< "Block " << i << endl;
		file_out.read(buffer,BLOCK_SIZE);
		print_buffer(buffer, "i");
		cout<< endl<<endl;
	}
	cout<< endl;
}

int file_length;

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
	//cout << from << " " << keyl << " " << size << endl;
	char buff_copy[size];
	memcpy(buff_copy, &buf[from], size);
	memcpy(&buf[from+POINTER_SIZE+keyl], buff_copy, size);
}

int get_curr_records(char *buffer){
	//Reevaluate the curr_records in this block
	char rec[4];
	rec[0] = buffer[1];
	rec[1] = buffer[2];
	rec[2] = buffer[3];
	rec[3] = '\0';
	return stoi(rec);
}

int get_block_number(char *buffer){
	//Reevaluate the block number of this block
	char rec[3];
	rec[0] = buffer[20];
	rec[1] = buffer[21];
	rec[3] = '\0';
	return stoi(rec);
}

void form_key(char *to, char *from,int keyl){
	int j = 0;
	for(j=0; j < keyl;j++){
		to[j] = from[j];
	}
	to[j] = '\0';
}

int read_keylength_filename(char* input_file_name,string index_file_name){
	fstream file_out;
	char key[3];
	file_out.open(index_file_name,ios::in | ios::binary);
	char buffer[BLOCK_SIZE];
	file_out.read(buffer,BLOCK_SIZE);
	if(DEBUG){
		cout << "Index filename is: " << index_file_name << endl;
		//print_buffer(buffer,"meta data");
	}

	key[0] = buffer[0];
	key[1] = buffer[1];
	key[2] = '\0';
	int keylength = stoi(key);

	key[0] = buffer[2];
	key[1] = buffer[3];
	key[2] = '\0';
	file_length = stoi(key);
	
	//memcpy((void*)keylength,buffer,2);
	//cout << "Keylength is: " << keylength << endl;

	
	int l = 0;
	while(l<file_length){
		input_file_name[l] = buffer[4+l];
		l++;
	}
	input_file_name[l] = '\0';
	//cout << "input file name is: " << input_file_name << endl;
	file_out.close();
	return keylength;
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
	//cout << "key : " << key_pair << " record address: " << cl.lpart << endl;
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

long long get_leaf_node(char* buffer, char* key_pair, int keylength, fstream& file_out, long long block_address){

	if(buffer[0] == 'L')
		return block_address;

	int curr_records = get_curr_records(buffer);
	if(DEBUG)
		cout << "Records in this block: " << curr_records << endl;

	int key_pair_size = keylength + POINTER_SIZE;
	char last_record[key_pair_size];
	char record[key_pair_size];
	char pair[keylength];
	int buffer_ptr = BLOCK_META_DATA_LEN;
	bool foundLeaf = false;
	if(DEBUG)
		print_buffer(buffer,"current buffer");
	for(int i=1;i<=curr_records;i++){
		memcpy(last_record,record,key_pair_size);
		memcpy(record,&buffer[buffer_ptr],key_pair_size);
		char record_key[keylength];

		form_key(record_key,record,keylength);
		form_key(pair,key_pair,keylength);
		
		int compare = strcmp(record_key, pair);
		if(DEBUG)
			cout << "Comparing " << record_key << " " << pair << " " << compare << endl;
		//Since the keys are ordered, We will get out position in one pass in that block
		if(compare > 0) {
			//Keypair would be in the left
			if(DEBUG)
				cout << "Go left" << endl;
			if(i == 1)
				memcpy(cl.cpart,&buffer[4],POINTER_SIZE);
			else
				memcpy(cl.cpart,&last_record[keylength],POINTER_SIZE);
			
			long long address_to_go = cl.lpart;
			file_out.seekg(cl.lpart,ios::beg);
			file_out.read(buffer,BLOCK_SIZE);
			//Set the parent address of the leaf node too
			cl.lpart = block_address;
			memcpy(&buffer[12],cl.cpart,POINTER_SIZE);
			if(DEBUG)
				cout << "Curr Address: " << block_address << " Child address: " << address_to_go << " Child parent address: " << cl.lpart << endl;
			file_out.seekg(address_to_go,ios::beg);
			file_out.write(buffer,BLOCK_SIZE);
			block_address = get_leaf_node(buffer,key_pair,keylength,file_out,address_to_go);
			foundLeaf = true;
			break;
		} else if(compare < 0){
			//Keypair would be in the right
			if(DEBUG)
				cout << "Go Right" << endl;
		} else{
			//Got the keypair, follow the pointer, similar to go right
			if(DEBUG)
				cout << "Found the key" << endl;
			memcpy(cl.cpart,&record[keylength],POINTER_SIZE);

			long long address_to_go = cl.lpart;
			file_out.seekg(cl.lpart,ios::beg);
			file_out.read(buffer,BLOCK_SIZE);
			//Set the parent address of the leaf node too
			cl.lpart = block_address;
			memcpy(&buffer[12],cl.cpart,POINTER_SIZE);
			if(DEBUG)
				cout << "Curr Address: " << block_address << " Child address: " << address_to_go << " Child parent address: " << cl.lpart << endl;
			file_out.seekg(address_to_go,ios::beg);
			file_out.write(buffer,BLOCK_SIZE);
			block_address = get_leaf_node(buffer,key_pair,keylength,file_out,address_to_go);
			foundLeaf = true;
			break;
		}
		buffer_ptr += key_pair_size;
	}
	if(!foundLeaf){
		memcpy(cl.cpart,&record[keylength],POINTER_SIZE);
		file_out.seekg(cl.lpart,ios::beg);
		long long address_to_go = cl.lpart;
		file_out.read(buffer,BLOCK_SIZE);
		//Set the parent address of the leaf node too
		cl.lpart = block_address;
		memcpy(&buffer[12],cl.cpart,POINTER_SIZE);
		if(DEBUG)
			cout << "Curr Address: " << block_address << " Child address: " << address_to_go << " Child parent address: " << cl.lpart << endl;
		file_out.seekg(address_to_go,ios::beg);
		file_out.write(buffer,BLOCK_SIZE);
		block_address = get_leaf_node(buffer,key_pair,keylength,file_out,address_to_go);
		foundLeaf = true;
	}

	return block_address;
}

void init_block_meta_data(char *buffer,fstream& file_out,bool isLeafNode, int block_number){
	//Inititalise meta data for a block
	file_out.seekg(block_number*BLOCK_SIZE,ios::beg);
	int buffer_ptr = 0;
	if(isLeafNode)
		buffer[0] = 'L';
	else
		buffer[0] = 'N';
	buffer_ptr++;
	//Initially curr_records are 0
	sprintf(&buffer[buffer_ptr],"%d",0);
	buffer_ptr += 3;
	//init block has no prev pointer
	cl.lpart = 99999999;
	memcpy(&buffer[buffer_ptr],cl.cpart,POINTER_SIZE);
	buffer_ptr += POINTER_SIZE;
	//init block is its own parent
	cl.lpart = block_number*BLOCK_SIZE;
	memcpy(&buffer[buffer_ptr],cl.cpart,POINTER_SIZE);
	buffer_ptr += POINTER_SIZE;
	//store the block number
	sprintf(&buffer[buffer_ptr],"%d",block_number);
	buffer_ptr += 3;
	padding(buffer,BLOCK_META_DATA_LEN);
	file_out.write(buffer,BLOCK_SIZE);
	
	//Reinitialise the file pointer to first record location of the current block 
	//file_out.seekg(block_number*BLOCK_SIZE+BLOCK_META_DATA_LEN,ios::beg);
}

//IT finds the possible place for a record to be inserted
int find_loc_in_block(char*buffer,char key_pair[],int keylength){
	
	int curr_records = get_curr_records(buffer);
	int key_pair_size = keylength + POINTER_SIZE;
	char record[key_pair_size];
	char pair[keylength];
	int buffer_ptr = BLOCK_META_DATA_LEN;
	int i;
	for(i=1;i<=curr_records;i++){
		memcpy(record,&buffer[buffer_ptr],key_pair_size);
		char record_key[keylength];

		form_key(record_key,record,keylength);
		form_key(pair,key_pair,keylength);
		
		int compare = strcmp(record_key, pair);
		if(DEBUG)
			cout << "Comparing " << record_key << " " << pair << " " << compare << endl;
		//Since the keys are ordered, We will get out position in one pass in that block
		if(compare > 0)
			break;
		buffer_ptr += key_pair_size;
	}
	return i-1;
}

void init_file_meta_data(int keylength, string input_file_name, fstream& file_out){
	char meta_data[BLOCK_SIZE];
	//First meta data block is of 1024 size and then the first node block starts
	int buffer_ptr = 0;
	// Store the keylength in the index file
	// and Root node of bplus tree
	sprintf(&meta_data[0], "%d", keylength);
	//meta_data.append(to_string(keylength));
	buffer_ptr += 2;
	
	sprintf(&meta_data[buffer_ptr], "%d", input_file_name.length());
	//meta_data.append(to_string(keylength));
	buffer_ptr += 2;

	//Store the input file name in the index file
	strcpy(&meta_data[buffer_ptr],input_file_name.c_str());
	buffer_ptr += input_file_name.length();
	//Initially let the root points to first block at address 1024
	cl.lpart = 1024;
	memcpy(&meta_data[buffer_ptr],cl.cpart,POINTER_SIZE);
	if(DEBUG)
		cout << "Meta: " << meta_data << endl;
	//memcpy(cl.cpart,&meta_data[buffer_ptr],8);
	//cout << "root: " << cl.lpart << endl;
	buffer_ptr += POINTER_SIZE;
	file_out.write(meta_data,BLOCK_SIZE);
	padding(meta_data,buffer_ptr);
	
	char buffer[BLOCK_SIZE];
	//The root is a leaf in the beginning, and its block number is 1
	init_block_meta_data(buffer,file_out,true, 1);
}

bool insert_in_block(char *key_pair,int keylength, char *buffer, int buffer_ptr, int curr_records){
	int key_pair_size = keylength + POINTER_SIZE;
	//We want to check each record in the block and fit our new record in increasing order
	if(curr_records == 0){
		//no record in this leaf now, insert that one directly
		memcpy(&buffer[buffer_ptr],&key_pair[0],key_pair_size);
		//cout << "Inserted first node in block" << endl;
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
			form_key(rec,record,keylength);
			form_key(pair,key_pair,keylength);
			
			int compare = strcmp(rec, pair);
			if(DEBUG)
				cout << "Comparing " << rec << " " << pair << " " << compare << endl;

			if(compare > 0) {
				//record is greater, need to insert pair before that
				shift(buffer,BLOCK_META_DATA_LEN + (i-1)*key_pair_size, keylength);
				memcpy(&buffer[buffer_ptr],key_pair,key_pair_size);
				inserted = true;
			} else if(compare < 0){
				//Check for the next element and insert at the last
			} else {
				//record is same as keypair in leaf node, report a duplicate
				cout << "A Duplicate was found, Skip " << pair <<endl;
				//inserted = true;
				return true;
			}
			buffer_ptr += key_pair_size;
		}
		if(inserted == false){
			//record is smaller, need to insert pair after that
			shift(buffer,BLOCK_META_DATA_LEN + (i-1)*key_pair_size, keylength);
			memcpy(&buffer[buffer_ptr],key_pair,key_pair_size);
			buffer_ptr += key_pair_size;
		}
	}
	return false;
}

bool domino_effect(char *buffer,long long write_address,fstream& file_out,char* key_pair,int keylength,long long root_location,int bplus_order,string index_file_name){
	memcpy(cl.cpart,&buffer[12],POINTER_SIZE);
	if(DEBUG)
		cout << "Curr node: " << write_address << " Parent: " << cl.lpart << endl; 
	long long parent_loc = cl.lpart;
	//Make a new node for the half nodes
	char split_node[BLOCK_SIZE];
	char parent_node[BLOCK_SIZE];
	
	int curr_records = get_curr_records(buffer);
	int key_pair_size = keylength + POINTER_SIZE;
	int mid_pt = curr_records / 2;
	block_number++;
	init_block_meta_data(split_node,file_out,true,block_number);

	//We need to decide where the incoming node will go, then the order will be figured
	//out by the insert_in_block function
	int mid_block_record_loc = BLOCK_META_DATA_LEN + mid_pt * key_pair_size;

	char record_before_mid[key_pair_size],record_after_mid[key_pair_size];
	memcpy(record_before_mid,&buffer[mid_block_record_loc-key_pair_size],key_pair_size);
	memcpy(record_after_mid,&buffer[mid_block_record_loc],key_pair_size);
	char record_before_mid_key[keylength],record_after_mid_key[keylength],pair[keylength];

	form_key(record_before_mid_key,record_before_mid,keylength);
	form_key(record_after_mid_key,record_after_mid,keylength);
	form_key(pair,key_pair,keylength);

	bool isDuplicate = false;
	int left_compare = strcmp(record_before_mid_key, pair);
	int right_compare = strcmp(record_after_mid_key,pair);
	if(DEBUG){
		cout << "Comparing left " << record_before_mid_key << " " << pair << " " << left_compare << endl;
		cout << "Comparing right " << record_after_mid_key << " " << pair << " " << right_compare << endl;
	}
	//domino variable false means leaf to parent, when its true it is parent to parent
	bool domino = false;
	if(left_compare < 0 && right_compare > 0) {
		//Incoming node in between left and right,Need to insert after mid-point
		if(DEBUG)
			cout << "You are given the Center house - Gryffindor" << endl;
		
		memcpy(&split_node[BLOCK_META_DATA_LEN],&buffer[mid_block_record_loc],BLOCK_SIZE - mid_block_record_loc);
		//If leaf then copy all after the mid point
		//If non leaf then just copy one after the mid point

		if(buffer[0] == 'L'){
			isDuplicate = insert_in_block(key_pair,keylength,split_node,BLOCK_META_DATA_LEN, 1);
			if(!isDuplicate){
				sprintf(&split_node[1],"%d",curr_records - mid_pt + 1);
				padding(split_node, BLOCK_META_DATA_LEN + (curr_records-mid_pt+1)*key_pair_size);
			} else{
				sprintf(&split_node[1],"%d",curr_records - mid_pt);
				padding(split_node, BLOCK_META_DATA_LEN + (curr_records-mid_pt)*key_pair_size);
			}
		} else{
			//If it is non leaf and in between, it will become the parent and will be inserted in parent		
			insert_in_block(key_pair,keylength,parent_node,BLOCK_META_DATA_LEN, 0);
			padding(parent_node,BLOCK_META_DATA_LEN+key_pair_size);
			sprintf(&parent_node[1],"%d",1);
			sprintf(&split_node[1],"%d",curr_records - mid_pt);
			padding(split_node, BLOCK_META_DATA_LEN + (curr_records-mid_pt)*key_pair_size);
			domino = true;
			split_node[0] = 'N';
			//Now assign the prev and next pointers
			memcpy(&split_node[4],&key_pair[keylength],POINTER_SIZE);
					
			//need to also change the parent of the new split_node left most child
			char left_child[BLOCK_SIZE];
			long long curr_loc = file_out.tellp();
			memcpy(cl.cpart,&split_node[4],POINTER_SIZE);
			file_out.seekg(cl.lpart,ios::beg);
			long long child_loc = cl.lpart;
			file_out.read(left_child,BLOCK_SIZE);
			cl.lpart = get_block_number(split_node)*BLOCK_SIZE;
			//updating the parent for previous left child
			memcpy(&left_child[12],cl.cpart,POINTER_SIZE);
			file_out.seekg(child_loc,ios::beg);
			file_out.write(left_child,BLOCK_SIZE);
			file_out.seekg(curr_loc,ios::beg);	
			if(DEBUG){
				print_buffer(buffer,"mid buffer");
				print_buffer(split_node,"mid split node");
				print_buffer(parent_node,"mid parent_node");
			}
		}
		sprintf(&buffer[1],"%d",mid_pt);
		padding(buffer, BLOCK_META_DATA_LEN + mid_pt*key_pair_size);
	} else if (left_compare < 0 && right_compare < 0) {
		//Incoming node is greater than left and right,Need to insert after mid-point
		if(DEBUG)
			cout << "You are given the Third house - HufflePuff" << endl;
		
		//If leaf then copy all after the mid point
		//If non leaf then just copy one after the mid point
		if(buffer[0] == 'L'){
			memcpy(&split_node[BLOCK_META_DATA_LEN],&buffer[mid_block_record_loc],BLOCK_SIZE - mid_block_record_loc);
			isDuplicate = insert_in_block(key_pair,keylength,split_node,BLOCK_META_DATA_LEN, curr_records - mid_pt);
			if(!isDuplicate){
				sprintf(&split_node[1],"%d",curr_records - mid_pt + 1);
				padding(split_node, BLOCK_META_DATA_LEN + (curr_records-mid_pt+1)*key_pair_size);
			} else{
				sprintf(&split_node[1],"%d",curr_records - mid_pt);
				padding(split_node, BLOCK_META_DATA_LEN + (curr_records-mid_pt)*key_pair_size);
			}
		} else{
			if(DEBUG){
				print_buffer(buffer,"just before change buffer");
				print_buffer(split_node,"just before change split");
				print_buffer(parent_node,"just before change parent");
				cout << "During changing Incoming key_pair is: " << key_pair << endl;
			}
			memcpy(&split_node[BLOCK_META_DATA_LEN],&buffer[mid_block_record_loc+key_pair_size],BLOCK_SIZE - mid_block_record_loc - key_pair_size);
			isDuplicate = insert_in_block(key_pair,keylength,split_node,BLOCK_META_DATA_LEN, curr_records - mid_pt - 1);
			memcpy(&parent_node[BLOCK_META_DATA_LEN],&buffer[mid_block_record_loc], key_pair_size);
			padding(parent_node,BLOCK_META_DATA_LEN+key_pair_size);
			sprintf(&parent_node[1],"%d",1);
			if(!isDuplicate){
				sprintf(&split_node[1],"%d",curr_records - mid_pt);
				padding(split_node, BLOCK_META_DATA_LEN + (curr_records-mid_pt)*key_pair_size);
			} else{
				sprintf(&split_node[1],"%d",curr_records - mid_pt - 1);
				padding(split_node, BLOCK_META_DATA_LEN + (curr_records-mid_pt-1)*key_pair_size);
			}
			domino = true;
			split_node[0] = 'N';
			//Now assign the prev and next pointers
			long long loc = BLOCK_META_DATA_LEN + ((get_curr_records(buffer)/2)+1)*key_pair_size - POINTER_SIZE;
			memcpy(&split_node[4],&buffer[loc],POINTER_SIZE);
			
			//need to also change the parent of the new split_node left most child
			char left_child[BLOCK_SIZE];
			long long curr_loc = file_out.tellp();
			memcpy(cl.cpart,&split_node[4],POINTER_SIZE);
			file_out.seekg(cl.lpart,ios::beg);
			long long child_loc = cl.lpart;
			file_out.read(left_child,BLOCK_SIZE);
			cl.lpart = get_block_number(split_node)*BLOCK_SIZE;
			//updating the parent for previous left child
			memcpy(&left_child[12],cl.cpart,POINTER_SIZE);
			file_out.seekg(child_loc,ios::beg);
			file_out.write(left_child,BLOCK_SIZE);
			file_out.seekg(curr_loc,ios::beg);	
			memcpy(key_pair,&parent_node[BLOCK_META_DATA_LEN],keylength);

			if(DEBUG){
				print_buffer(buffer,"mid buffer");
				print_buffer(split_node,"mid split node");
				print_buffer(parent_node,"mid parent_node");
			}
		}
		sprintf(&buffer[1],"%d",mid_pt);
		padding(buffer, BLOCK_META_DATA_LEN + mid_pt*key_pair_size);
		
	} else if (left_compare > 0 && right_compare > 0) {
		//Incoming node less than left and right,Need to insert in first node, no change in second_node
		if(DEBUG)
			cout << "You are given the First house - Slytherine" << endl;
		memcpy(&split_node[BLOCK_META_DATA_LEN],&buffer[BLOCK_META_DATA_LEN],(mid_pt-1)*key_pair_size);
		isDuplicate = insert_in_block(key_pair,keylength,split_node,BLOCK_META_DATA_LEN,curr_records-mid_pt-1);
		if(!isDuplicate){
			padding(split_node, BLOCK_META_DATA_LEN +(curr_records-mid_pt)*key_pair_size);
			sprintf(&split_node[1],"%d",curr_records-mid_pt);
		} else{
			padding(split_node, BLOCK_META_DATA_LEN +(curr_records-mid_pt-1)*key_pair_size);
			sprintf(&split_node[1],"%d",curr_records-mid_pt-1);
		}
		memcpy(key_pair,&buffer[BLOCK_META_DATA_LEN],keylength);
		if(buffer[0] != 'L'){
			memcpy(&split_node[4],&key_pair[keylength],POINTER_SIZE);
			memcpy(&split_node[BLOCK_META_DATA_LEN+keylength],&buffer[4],POINTER_SIZE);

			memcpy(cl.cpart,&buffer[BLOCK_META_DATA_LEN+keylength],POINTER_SIZE);
			memcpy(&buffer[4],cl.cpart,POINTER_SIZE);			
	
			domino = true;
			split_node[0] = 'N';
			if(DEBUG){
				cout << "Incoming key_pair is: " << key_pair << endl;
				cout << "left child of buffer is "<< cl.lpart << endl;
				print_buffer(buffer,"mid buffer");
				print_buffer(split_node,"mid split node");
				print_buffer(parent_node,"mid parent_node");
				print_blocks(file_out);
			}
		}
	} else{
		cout << "Duplicate value found,skip" << endl;
		block_number--;
		return true;
	}

	//Decide the parent for the current 2 nodes
	if(!isDuplicate && parent_loc == write_address){
		block_number++;
		if(!domino)
			init_block_meta_data(parent_node,file_out,false,block_number);
		else{
			sprintf(&parent_node[20],"%d",block_number);
			parent_node[0] = 'N';
			//init block is its own parent
			cl.lpart = block_number*BLOCK_SIZE;
			memcpy(&parent_node[12],cl.cpart,POINTER_SIZE);
		}
		if(DEBUG){
			cout << "This node does not have a parent yet"<< endl;
			print_buffer(buffer,"mid buffer");
			print_buffer(split_node,"mid split node");
			print_buffer(parent_node,"mid parent_node");
		}

		if((left_compare < 0 && right_compare > 0) || (left_compare < 0 && right_compare < 0)){
			//Incoming node in between left and right,Need to insert after mid-point OR
			//Incoming node is greater than left and right,Need to insert after mid-point

			if(!domino){
				memcpy(&parent_node[BLOCK_META_DATA_LEN],&split_node[BLOCK_META_DATA_LEN],key_pair_size);
				padding(parent_node,BLOCK_META_DATA_LEN+key_pair_size);
				sprintf(&parent_node[1],"%d",1);
			}
			//Now assign the prev and next pointers
			//The parent has prev pointer in meta data, and next pointer with the key
			cl.lpart = get_block_number(buffer)*BLOCK_SIZE;
			memcpy(&parent_node[4],cl.cpart,POINTER_SIZE);
			cl.lpart = get_block_number(split_node)*BLOCK_SIZE;
			memcpy(&parent_node[BLOCK_META_DATA_LEN+keylength],cl.cpart,POINTER_SIZE);

			if(buffer[0] == 'L'){
				//Now assign the next for leaf nodes
				cl.lpart = get_block_number(split_node)*BLOCK_SIZE;
				memcpy(&buffer[4],cl.cpart,POINTER_SIZE);
				if(DEBUG)
					cout << "Set " << get_block_number(buffer) << " next to " << cl.lpart << endl;
			}
		} else if(left_compare > 0 && right_compare > 0){
			//Incoming node less than left and right,Need to insert before mid-point
			
			memmove(&buffer[BLOCK_META_DATA_LEN],&buffer[BLOCK_META_DATA_LEN+key_pair_size],BLOCK_SIZE-key_pair_size);
			sprintf(&buffer[1],"%d",get_curr_records(buffer)-1);

			memcpy(&parent_node[BLOCK_META_DATA_LEN],&buffer[BLOCK_META_DATA_LEN],key_pair_size);
			padding(parent_node,BLOCK_META_DATA_LEN+key_pair_size);
			sprintf(&parent_node[1],"%d",1);
			
			
			//Now assign the next pointers
			//The parent has next pointer in meta data, and prev pointer with the key
			cl.lpart = get_block_number(split_node)*BLOCK_SIZE;
			memcpy(&parent_node[4],cl.cpart,POINTER_SIZE);
			cl.lpart = get_block_number(buffer)*BLOCK_SIZE;
			memcpy(&parent_node[BLOCK_META_DATA_LEN+keylength],cl.cpart,POINTER_SIZE);
			
			if(split_node[0] == 'L'){
				//Now assign the next for leaf nodes
				cl.lpart = get_block_number(buffer)*BLOCK_SIZE;
				memcpy(&split_node[4],cl.cpart,POINTER_SIZE);
				if(DEBUG)
					cout << "Set " << get_block_number(split_node) << " next to " << cl.lpart << endl;
			}
		}
		//Make the parent_node parent of initial and split node
		cl.lpart = get_block_number(parent_node)*BLOCK_SIZE;
		memcpy(&buffer[12],cl.cpart,POINTER_SIZE);
		memcpy(&split_node[12],cl.cpart,POINTER_SIZE);

		if(DEBUG){
			cout << "This was a leaf node to parent creation" << endl;
			print_buffer(buffer,"original");
			print_buffer(split_node,"new");
			print_buffer(parent_node,"parent");
		}

		//Change the global root pointer location
		file_out.seekg(root_location,ios::beg);
		cl.lpart = get_block_number(parent_node)*BLOCK_SIZE;
		file_out.write(cl.cpart,POINTER_SIZE);

		file_out.seekg(get_block_number(buffer)*BLOCK_SIZE,ios::beg);
		file_out.write(buffer,BLOCK_SIZE);
		file_out.seekg(get_block_number(split_node)*BLOCK_SIZE,ios::beg);
		file_out.write(split_node,BLOCK_SIZE);
		file_out.seekg(get_block_number(parent_node)*BLOCK_SIZE,ios::beg);
		file_out.write(parent_node,BLOCK_SIZE);
	} else{
		if(!isDuplicate){
			//This means we got a previous parent to fill in the split nodes
			memcpy(cl.cpart,&buffer[12],POINTER_SIZE);
			parent_loc = cl.lpart;
			//Make the parent_node parent of initial and split node
			memcpy(&split_node[12],cl.cpart,POINTER_SIZE);

			if(DEBUG)
				cout << "Incoming key_pair is: " << key_pair << endl;

			file_out.seekg(parent_loc,ios::beg);
			file_out.read(parent_node,BLOCK_SIZE);
			
			if(left_compare > 0 && right_compare > 0){
				memmove(&buffer[BLOCK_META_DATA_LEN],&buffer[BLOCK_META_DATA_LEN+key_pair_size],BLOCK_SIZE-key_pair_size);
				sprintf(&buffer[1],"%d",get_curr_records(buffer)-1);
			}

			if((left_compare < 0 && right_compare > 0) || (left_compare < 0 && right_compare < 0)){
				if(buffer[0] == 'L'){
					//Now assign the next for leaf nodes
					memcpy(cl.cpart,&buffer[4],POINTER_SIZE);
					if(cl.lpart != 99999999){
						memcpy(&split_node[4],cl.cpart,POINTER_SIZE);
					}
					cl.lpart = get_block_number(split_node)*BLOCK_SIZE;
					memcpy(&buffer[4],cl.cpart,POINTER_SIZE);
					if(DEBUG)
						cout << "Set " << get_block_number(buffer) << " next to " << cl.lpart << endl;
				}
			} else if(left_compare > 0 && right_compare > 0){
				if(split_node[0] == 'L'){
					//Now assign the next for leaf nodes
					memcpy(cl.cpart,&split_node[4],POINTER_SIZE);
					if(cl.lpart != 99999999){
						memcpy(&buffer[4],cl.cpart,POINTER_SIZE);
					}
					cl.lpart = get_block_number(buffer)*BLOCK_SIZE;
					memcpy(&split_node[4],cl.cpart,POINTER_SIZE);
					if(DEBUG)
						cout << "Set " << get_block_number(split_node) << " next to " << cl.lpart << endl;
				}
			}

			curr_records = get_curr_records(parent_node);
			if(DEBUG){
				cout << "Records in this block: " << curr_records << endl;
				cout << "This already had a parent node" << endl;
				print_buffer(buffer,"original");
				print_buffer(split_node,"new");
			}
			file_out.seekg(get_block_number(buffer)*BLOCK_SIZE,ios::beg);
			file_out.write(buffer,BLOCK_SIZE);
			file_out.seekg(get_block_number(split_node)*BLOCK_SIZE,ios::beg);
			file_out.write(split_node,BLOCK_SIZE);

			cl.lpart = get_block_number(split_node)*BLOCK_SIZE;			
			if(!domino){
				memcpy(key_pair,&split_node[BLOCK_META_DATA_LEN],keylength);
			}

			if (left_compare > 0 && right_compare > 0 && buffer[0] == 'L'){
				memcpy(key_pair,&buffer[BLOCK_META_DATA_LEN],keylength);
			}
		
			memcpy(&key_pair[keylength],cl.cpart,POINTER_SIZE);

			if(curr_records + 1 == bplus_order){
				if(DEBUG){
					cout << "After splitting,parent node is full too, need to split it too" << endl;
					cout << "New Incoming key_pair now is: " << key_pair << endl;
					print_blocks(file_out);
				}
				isDuplicate = domino_effect(parent_node,parent_loc,file_out,key_pair,keylength,root_location,bplus_order,index_file_name);
				return isDuplicate;
			} else{
				if (left_compare > 0 && right_compare > 0){
				
					int count = find_loc_in_block(parent_node,key_pair,keylength);
					if(DEBUG)
						cout << "we got count as: " << count << " " << BLOCK_META_DATA_LEN+count*key_pair_size-POINTER_SIZE << endl<< endl<<endl;
					if(count > 0)
						memcpy(&parent_node[BLOCK_META_DATA_LEN+count*key_pair_size-POINTER_SIZE],cl.cpart,POINTER_SIZE);
					else if(count == 0)
						memcpy(&parent_node[4],cl.cpart,POINTER_SIZE);
				
					cl.lpart = get_block_number(buffer)*BLOCK_SIZE;
					memcpy(&key_pair[keylength],&cl.cpart,POINTER_SIZE);
				}
				isDuplicate = insert_in_block(key_pair,keylength,parent_node,BLOCK_META_DATA_LEN,curr_records);
				if(!isDuplicate)
					sprintf(&parent_node[1],"%d",curr_records+1);
				else
					sprintf(&parent_node[1],"%d",curr_records);
			}

			if(DEBUG){
				print_blocks(file_out);
				print_buffer(parent_node,"parent");
			}

			file_out.seekg(get_block_number(parent_node)*BLOCK_SIZE,ios::beg);
			file_out.write(parent_node,BLOCK_SIZE);
		} else {
			cout << "Found a duplicate, skip" << endl;
			return true;
		}
	}
	return isDuplicate;
}

//INDEX -create <input file> <output file> <key size>
void create_index(string input_file_name, string output_file_name,int keylength) {
	fstream file_in,file_out;
	string line;
	streampos beg,ending;

	file_in.open(input_file_name, ios::binary | ios::in);
	file_out.open(output_file_name,ios::out);
	file_out.close();
	file_out.open(output_file_name,ios::out | ios::in | ios::binary);

	beg = file_in.tellg();
	file_in.seekg(0,ios::end);
	ending = file_in.tellg();
	file_in.seekg(0, ios::beg);

	int file_size = ending - beg;

	// minus 12 => 1 for leaf/non-leaf node, 3 for curr_#_records, 8 for prev pointer,
	// 8 for its parent pointer, 3 for block number for better debugging 
	int node_size = BLOCK_SIZE - BLOCK_META_DATA_LEN;
	int max_records = node_size / (keylength + POINTER_SIZE);
	int bplus_order = max_records + 1;
	
	//for now change bplus_order to 3, so max 2 keys in it
	//bplus_order = 5;
	if(DEBUG){
		cout << "Size of input file is "  << file_size << " bytes" <<endl;
		cout << "Max records / node: "  << max_records << endl;
	}
    char buffer[BLOCK_SIZE];
	int buffer_ptr = 0;
	int curr_records = 0;
	block_number = 0;
	int last_starting = file_in.tellp();
	bool isLeafNode = false;
	block_number++;
	init_file_meta_data(keylength,input_file_name,file_out);

	int root_location = 4 + input_file_name.length();
	int next_pointer_loc = 4;
	while(getline(file_in,line)){
		buffer_ptr = 0;
		isLeafNode = false;
		int key_pair_size = keylength + POINTER_SIZE;
		char key_pair[key_pair_size];
		make_key_pair(key_pair,keylength,line,buffer_ptr,last_starting);
		
		//Assuming root adrress gets updated here during split operation
		//This is to keep track of global root
		int root_address = get_root_address(file_out, root_location);

		//We read the 1k block
		file_out.seekg(root_address,ios::beg);
		file_out.read(buffer,BLOCK_SIZE);
		if(DEBUG){
			print_buffer(buffer, "current buffer");
			cout << "Root at: " << root_address << endl;
		}

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

		curr_records = get_curr_records(buffer);
		if(DEBUG)
			cout << "Records in this block: " << curr_records << endl;

		//the write address will change to go to the leaf node		
		int write_address = root_address;
		bool isDuplicate = false;

		if(!isLeafNode){
			//This is a root node, so we need to go using the key pointers
			//cout << "Encountered a non-leaf node" << endl;			
			//We need to decide which leaf node the the incoming pair would go
			
			cl.lpart = get_leaf_node(buffer,key_pair,keylength,file_out, write_address);
			file_out.seekg(cl.lpart,ios::beg);
			file_out.read(buffer,BLOCK_SIZE);
			if(DEBUG){				
				cout << "Found the leaf node for key_pair " << key_pair << " address: " << cl.lpart << endl;
				print_buffer(buffer,"original");
			}
			
			curr_records = get_curr_records(buffer);
			//cout << "Records in this block: " << curr_records << endl;
			isLeafNode = true;
			write_address = cl.lpart;
		}

		if (curr_records + 1 >= bplus_order){
			if(DEBUG)
				cout << "Order maxed, Need to split on incoming node" << endl;
			isDuplicate = domino_effect(buffer,write_address,file_out,key_pair,keylength,root_location,bplus_order,output_file_name);
			if(DEBUG)
				print_blocks(file_out);
			last_starting = file_in.tellp();
			continue;
		}

		memcpy(cl.cpart,&key_pair[keylength],POINTER_SIZE);
		if(DEBUG)
			cout<< "On a leaf node, now entering the key_pair " << key_pair << " loc:" << cl.lpart << endl;
		
		isDuplicate = insert_in_block(key_pair,keylength,buffer,BLOCK_META_DATA_LEN,curr_records);
		//If there is a duplicate key_pair then just skip insertion for that key_pair
		if(isDuplicate){
			last_starting = file_in.tellp();
			continue;
		}

		curr_records++;
		sprintf(&buffer[1],"%d",curr_records);
		padding(buffer, BLOCK_META_DATA_LEN + curr_records*key_pair_size);

		file_out.seekg(write_address,ios::beg);
		long before = file_out.tellp();
		file_out.write(buffer,BLOCK_SIZE);
		long after = file_out.tellp();
		last_starting = file_in.tellp();

		if(DEBUG){
			cout << "Before: " << before << " After: " << after << endl;
			print_buffer(buffer,"modified buffer");
			print_blocks(file_out);
		}
	}

	file_in.close();
	file_out.close();
	
}
//INDEX -find <index filename> <key>
int find_record(string index_file_name, char key[],bool display) {
	
	fstream file_out,file_in;
	cout << index_file_name << " " << key << endl;
	file_out.open(index_file_name,ios::out | ios::in | ios::binary);
	char* input_file_name = new char[50];
	int keylength;
	keylength = read_keylength_filename(input_file_name,index_file_name);

	char buffer[BLOCK_SIZE];
	//Assuming root adrress gets updated here during split operation
	//This is to keep track of global root
	int root_location = 4 + file_length;
	int root_address = get_root_address(file_out, root_location);
	
	if(DEBUG)
		cout << "Root at: " << root_address << endl;

	//We read the 1k block
	file_out.seekg(root_address,ios::beg);
	file_out.read(buffer,BLOCK_SIZE);

	cl.lpart = get_leaf_node(buffer,key,keylength,file_out,root_address);
	file_out.seekg(cl.lpart,ios::beg);
	file_out.read(buffer,BLOCK_SIZE);
	if(DEBUG)
		print_buffer(buffer,"found/not");
	int curr_records = get_curr_records(buffer);
	int key_pair_size = keylength + POINTER_SIZE;
	char record[key_pair_size];
	char pair[keylength];
	int buffer_ptr = BLOCK_META_DATA_LEN;
	bool foundLeaf = false;
	int i;
	for(i=1;i<=curr_records;i++){
		memcpy(record,&buffer[buffer_ptr],key_pair_size);
		char record_key[keylength];

		form_key(record_key,record,keylength);
		form_key(pair,key,keylength);
		
		int compare = strcmp(record_key, pair);
		if(DEBUG)
			cout << "Comparing " << record_key << " " << pair << " " << compare << endl;
		//Since the keys are ordered, We will get out position in one pass in that block
		if(compare == 0) {
			
			foundLeaf = true;
			memcpy(cl.cpart,&record[keylength],POINTER_SIZE);
			if(display)
				cout << key << " found at " << cl.lpart << endl;
			file_in.open(input_file_name,ios::in | ios::binary);
			file_in.seekg(cl.lpart,ios::beg);
			string line;
			getline(file_in,line);
			cout << line << endl;
			file_in.close();
			return i;
		} else if(compare > 0)
			break;
		buffer_ptr += key_pair_size;
	}
	if(!foundLeaf && display)
		cout << key << " NOT found" << endl;
	file_out.close();
	return -(i-1);

}

//INDEX -insert <index filename> "new text line to be inserted."
bool insert_record(string index_file_name, string line) {
	fstream file_out,file_in;
	char* input_file_name = new char[50];
	int keylength;

	keylength = read_keylength_filename(input_file_name,index_file_name);

	file_out.open(index_file_name,ios::out | ios::in | ios::binary);
	file_in.open(input_file_name,ios::in | ios::binary);
	char buffer[BLOCK_SIZE];
	//Assuming root adrress gets updated here during split operation
	//This is to keep track of global root
	
	int root_location = 4 + file_length;

	int buffer_ptr = 0;
	bool isLeafNode = false;
	int key_pair_size = keylength + POINTER_SIZE;
	char key_pair[key_pair_size];

	file_in.seekg(0,ios::end);
	int last_starting = file_in.tellg();
	//this is for the end of file insert
	make_key_pair(key_pair,keylength,line,buffer_ptr, last_starting);
	

	// minus 12 => 1 for leaf/non-leaf node, 3 for curr_#_records, 8 for prev pointer,
	// 8 for its parent pointer, 3 for block number for better debugging 
	int node_size = BLOCK_SIZE - BLOCK_META_DATA_LEN;
	int max_records = node_size / (keylength + POINTER_SIZE);
	int bplus_order = max_records + 1;

	//Assuming root adrress gets updated here during split operation
	//This is to keep track of global root
	int root_address = get_root_address(file_out, root_location);
	
	//We read the 1k block
	file_out.seekg(root_address,ios::beg);
	file_out.read(buffer,BLOCK_SIZE);
	if(DEBUG){
		print_buffer(buffer, "current buffer");
		cout << "Root at: " << root_address << endl;
	}
	
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

	int curr_records = get_curr_records(buffer);
	//cout << "Records in this block: " << curr_records << endl;

	//the write address will change to go to the leaf node		
	int write_address = root_address;
	bool isDuplicate = false;

	if(!isLeafNode){
		//This is a root node, so we need to go using the key pointers
		if(DEBUG)
			cout << "Encountered a non-leaf node" << endl;			
		//We need to decide which leaf node the the incoming pair would go
		
		cl.lpart = get_leaf_node(buffer,key_pair,keylength,file_out, write_address);
		file_out.seekg(cl.lpart,ios::beg);
		file_out.read(buffer,BLOCK_SIZE);
		
		curr_records = get_curr_records(buffer);
		if(DEBUG){
			cout << "Records in this block: " << curr_records << endl;
			cout << "Found the leaf node for key_pair " << key_pair << " address: " << cl.lpart << endl;
			print_buffer(buffer,"original");
		}
		isLeafNode = true;
		write_address = cl.lpart;
	}

	if (curr_records + 1 >= bplus_order){
		//cout << "Order maxed, Need to split on incoming node" << endl;
		isDuplicate = domino_effect(buffer,write_address,file_out,key_pair,keylength,root_location,bplus_order,index_file_name);
		//print_blocks(file_out);
		return isDuplicate;
	}

	memcpy(cl.cpart,&key_pair[keylength],POINTER_SIZE);
	//cout<< "On a leaf node, now entering the key_pair " << key_pair << " loc:" << cl.lpart << endl;
	isDuplicate = insert_in_block(key_pair,keylength,buffer,BLOCK_META_DATA_LEN,curr_records);
	//If there is a duplicate key_pair then just skip insertion for that key_pair
	if(isDuplicate){
		return true;
	}

	curr_records++;
	sprintf(&buffer[1],"%d",curr_records);
	padding(buffer, BLOCK_META_DATA_LEN + curr_records*key_pair_size);

	file_out.seekg(write_address,ios::beg);
	long before = file_out.tellp();
	file_out.write(buffer,BLOCK_SIZE);
	long after = file_out.tellp();
	if(DEBUG){
		cout << "Before: " << before << " After: " << after << endl;
		print_buffer(buffer,"modified buffer");
	}
	last_starting = file_in.tellp();
	file_in.close();
	//print_blocks(file_out);
	if(!isDuplicate){
		ofstream file;
		file.open(input_file_name,ios::app);
		cout << "Write to the file at:" << last_starting << " " << line << endl;
		file << line;
		file.close();
	}

	
	file_out.close();
	return isDuplicate;
}

//INDEX -list <index filename> <starting key> <count>
void list_record(string index_file_name,char key[],int count) {
	fstream file_out,file_in;
	char* input_file_name = new char[50];
	int keylength;

	keylength = read_keylength_filename(input_file_name,index_file_name);
	file_out.open(index_file_name,ios::out | ios::in | ios::binary);
	file_in.open(input_file_name,ios::in | ios::binary);
	
	char buffer[BLOCK_SIZE];
	//Assuming root adrress gets updated here during split operation
	//This is to keep track of global root
	
	int root_location = 4 + file_length;
	int root_address = get_root_address(file_out, root_location);
	if(DEBUG)
		cout << "Root at: " << root_address << endl;

	//We read the 1k block
	file_out.seekg(root_address,ios::beg);
	file_out.read(buffer,BLOCK_SIZE);

	int inBplusTree = false;
	int position_in_block = find_record(index_file_name,key,false);
	if( position_in_block > 0)
		inBplusTree = true;

	cl.lpart = get_leaf_node(buffer,key,keylength,file_out,root_address);
	file_out.seekg(cl.lpart,ios::beg);
	file_out.read(buffer,BLOCK_SIZE);
	if(DEBUG)
		print_buffer(buffer,"found/not");
	int curr_records = get_curr_records(buffer);
	int key_pair_size = keylength + POINTER_SIZE;
	char record[key_pair_size];
	int buffer_ptr;
	int i = 0,j = 1;

	int skip = 0;

	buffer_ptr = BLOCK_META_DATA_LEN;
	//if not in bplus tree, then we need to go to the next greater key
	if(!inBplusTree){
		 skip = find_loc_in_block(buffer,key,keylength);
		//cout << "skip is: " << skip << " pointer: " << buffer_ptr << " j:" << j << " cur_rec: " << curr_records << endl;
	} else{
		skip = position_in_block - 1;
	}

	buffer_ptr += skip * key_pair_size;
	j = skip + 1;

	string line;
	while(i < count){
		for(;j<=curr_records;j++){
			memcpy(record,&buffer[buffer_ptr],key_pair_size);
			memcpy(cl.cpart,&record[keylength],POINTER_SIZE);
			file_in.seekg(cl.lpart,ios::beg);
			getline(file_in,line);
			cout << "loc: " << cl.lpart << " " << line << endl;
			i++;
			if(i == count)
				break;
			buffer_ptr += key_pair_size;
		}
		j = 1;
		if(i == count)
			break;
		memcpy(cl.cpart,&buffer[4],POINTER_SIZE);
		//cout << "Next address is: " << cl.lpart << endl;
		if(cl.lpart == 99999999)
			break;
		file_out.seekg(cl.lpart,ios::beg);
		file_out.read(buffer,BLOCK_SIZE);
		curr_records = get_curr_records(buffer);
		if(DEBUG)
			print_buffer(buffer,"updated");
		buffer_ptr = BLOCK_META_DATA_LEN;
	}
		
	
	file_in.close();
	file_out.close();
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
		create_index(input_file_name,output_file_name,keylength);
		cout << "All records inserted successfully" << endl;
	} else if(index_type ==  "-find") {
		cout << "Need to search records based on the indexed data" << endl; 
		string index_file_name = argv[2];
		string key = argv[3];
		char find_key[40];
		strcpy(find_key,key.c_str());
		cout << "Index file: " << index_file_name << " Key: " << find_key << endl;
		find_record(index_file_name,find_key,true);
	} else if(index_type ==  "-insert") {
		cout << "Need to insert records in the indexed data" << endl; 
		string index_file_name = argv[2];
		string insertion_line = argv[3];
		cout << "Index file: " << index_file_name << " Insertion line: " << insertion_line << endl;
		bool hasDuplicate = insert_record(index_file_name,insertion_line);
		if(!hasDuplicate){
			cout << "Successfully inserted record " << insertion_line << endl;
		} else {
			cout << "It was a duplicate key, so skip insertion" << endl;
		}
	} else if(index_type ==  "-list") {
		cout << "Need to list records based on the indexed data" << endl; 
		string index_file_name = argv[2];
		char* starting_key = argv[3];
		int count_afterwards = stoi(argv[4]);
		cout << "Index file: " << index_file_name << " Starting Key: " << starting_key << " Count: " << count_afterwards << endl;
		list_record(index_file_name,starting_key,count_afterwards);
	}

    return 0;
}