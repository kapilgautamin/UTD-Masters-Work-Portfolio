/******************************************************************************
 * Random-access file example program in C++.
 *
 * This shows how to treat a text file as binary and access records within
 * it.  This reads only the first two records from a 1K block, keeping track
 * of where each record starts and ends in the buffer.
 *
 * Written by John Cole at The University of Texas at Dallas on November 1,
 * 2016.  Copy, adapt, and use this code freely.
 ******************************************************************************/

#include <stdio.h>
#include <io.h>
#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <stdlib.h>
using namespace std;

union charLong
  {
  long long lpart;
  char cpart[8];
  };
charLong cl;

struct Student
  {
  char name[20];  //size of char array 20
  char grade;   //size of char 1 //has a padding of 3
  string school;    //size of string 24
  double gpa;   //size of double 8
  int year;     //size of int 4
  };

int main()
{
  Student s;
  Student t;
  s.grade = 'A';
  s.gpa = 3.97;
  s.year = 3;
  strcpy(s.name, "Bill");
  s.school = "The University of Texas at Dallas";

  cout << "Memory addresses " << reinterpret_cast<int>(&s.name) << " " << reinterpret_cast<int>(&s.grade) << " ";
  cout << reinterpret_cast<int>(&s.gpa) << " " << reinterpret_cast<int>(&s.year) << " ";
  cout << reinterpret_cast<int>(&s.school) << endl;
  cout << "Size of a double datatype " << sizeof(double) << endl;
  cout << "Size of a string datatype " << sizeof(string) << endl;
  //strlen works only on const char *
  cout << "Size of school string " << s.school.size() << "\nLength of school string " <<(s.school).length() << endl;
  cout << "Size of char array: " << sizeof(char[20]) << endl;
  cout << "Size of structure: " << sizeof(s) << endl;
  fstream stu;
  stu.open("Student.txt", ios::out | ios::binary);
  stu.write(reinterpret_cast<char *>(&s), sizeof(Student));
  //cout<<s.name << " " <<s.school << " " <<s.year<<endl;


  long long whereami = stu.tellp();

  //char* memory = new char[whereami];
  //stu.seekg(0,ios::beg);
  //stu.read(memory, whereami);
  //cout<<memory << endl;
  stu.close();


  stu.open("Student.txt", ios::in | ios::binary);
  //stu.seekg(0,ios::beg);
  stu.read(reinterpret_cast<char *>(&t), sizeof(Student));
  //copying the structure s to structure t
  cout<<t.name<<endl;
  stu.close();


  fstream f;
  char buffer[1024];
  int recptr = 0;
  int startptr = 0;
  char record1[100];
  char record2[100];
  long long l1, l2;
  char fname[] = "JCBinary1.txt";
  f.open(fname, ios::in | ios::out | ios::binary);
  if (f.fail())
    {
    cout << "error opening file" << endl;
    f.open(fname, ios::out);
    if (f.fail())
      {
      cout << "Real error opening file" << endl;
      system("pause");
      return 0;
      }
    f.close();
    f.open(fname, ios::in | ios::out | ios::binary);
    }

  int x = 1;
  int bufptr = 0;
  int keylen = 15;
  //These below strings have a '\0' at the end of , so size gives a answer including that, 
  //while strlen gives it without that.
  char str1[] = "This is a slightly longer test\n";
  char str2[] = "After the long\n";
  cout<<" Size of string str2: " << sizeof(str2)<< " " << strlen(str2)<<endl;

  // This section writes a string, then binary data, then another string,
  // to a file that is open for input and output.
  strcpy(buffer, str1);
  bufptr += strlen(str1);
  cl.lpart = 13;
  memcpy(&buffer[bufptr], cl.cpart, 8);
  bufptr += sizeof(charLong);
  strcpy(&buffer[bufptr], str2);
  bufptr += strlen(str2);
  cl.lpart = 123456789;
  memcpy(&buffer[bufptr], cl.cpart, 8);
  try
    {
    f.write(buffer, 1024);

    // Having written the file, reposition to the beginning, read in the
    // 1K block we wrote, and pull data out of it.
    memset(&buffer[0], 0, 1024);
    f.seekg(0l, ios::beg);
    f.read(&buffer[0], 1024);
    bufptr = 0;
    x = 0;
    while (buffer[x] != '\n')
      {
      record1[x] = buffer[bufptr];
      x++;
      bufptr++;
      }
    record1[x] = buffer[bufptr];
    x++;
    bufptr++;
    record1[x] = 0;
    memcpy(cl.cpart, &buffer[bufptr], 8);
    l1 = cl.lpart;
    bufptr += 8;
    x = 0;
    while (buffer[bufptr] != '\n')
      {
      record2[x] = buffer[bufptr];
      x++;
      bufptr++;
      }
    record2[x] = buffer[bufptr];
    x++;
    bufptr++;
    record2[x] = 0;
    memcpy(cl.cpart, &buffer[bufptr], 8);
    l2 = cl.lpart;
    cout << "First long: " << l1 << "  Second long: " << l2 << endl;

    // Illustrate updating:
    f.seekg(strlen(str1), ios::beg);
    l2 = 65 * 256 + 66;     //16706 //in essence
    f.write(reinterpret_cast<char *>(&l2), sizeof(l2));
    f.seekg(strlen(str1), ios::beg);
    cout << reinterpret_cast<char *>(&l2) << endl;
    f.read(reinterpret_cast<char *>(&l1), sizeof(l1));
    cout << "l1 variable after rewrite and read: " << l1 << endl;
    f.close();
    }
  catch (exception ex)
    {
    cout << "Error in binary I/O " << ex.what() << endl;
    }



//How to read contents from a file
  x = 1;
//???  f = new fstream();
  f.open("example.txt", ios::in | ios::binary);
  f.read(buffer, 1024);
  // Find the end of the first record.
  while (buffer[recptr] != '\n')
    recptr++;

  cout << "Record1 is " << recptr << " bytes long" << endl;
  strncpy(record1, &buffer[startptr], recptr);
  record1[recptr] = 0;
  cout << "Record1: " << record1 << endl;

  // Move past the NL character, then save that as the start of the second line.
  recptr++;
  startptr = recptr;

  // Find the end of the second record.
  while (buffer[recptr] != '\n')
    recptr++;

  cout << "Record2 is " << recptr << " bytes long" << endl;
  strncpy(record2, &buffer[startptr], recptr-startptr);
  record2[recptr] = 0;
  cout << "Record2: " << record2 << endl;

  f.close();
  //system("pause");
  return 0;
}

