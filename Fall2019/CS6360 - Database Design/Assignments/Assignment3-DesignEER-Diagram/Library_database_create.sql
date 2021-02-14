drop database if exists Library;
create database Library;

Use Library;

drop table if exists user;
create table user(
	utdId	char(10) not null,
	netId	char(9) not null unique,
	deptFrom varchar(50),
	dob		datetime not null,
	sex		char(1),
	fName	char(50) not null,
	mName	char(50),
	lName	char(50),
	CHECK (year(dob) > 1800),
	CONSTRAINT uq_user UNIQUE(netId),
	CONSTRAINT pk_user primary key (utdId)
);

drop table if exists userAddress;
create table userAddress(
	addressId	char(10) not null,
	steetAddress char(100),
	city		char(20),
	state		char(20),
	country		char(20) not null,
	zip			char(6),
	userId	char(10) not null,
	CONSTRAINT pk_userAddress primary key (addressId),
	CONSTRAINT fk_userAddress foreign key (userId) references user(utdID)
);

drop table if exists userPhone;
create table userPhone(
	countryCode	char(2),
	phoneno		char(10),
	userId	char(10) not null,
	CONSTRAINT pk_userPhone primary key (userId,countryCode,phoneno),
	CONSTRAINT fk_userPhone foreign key (userId) references user(utdID)
);

drop table if exists userEmail;
create table userEmail(
	email		char(30),
	userId	char(10) not null,
	CONSTRAINT pk_userEmail primary key (userId,email),
	CONSTRAINT fk_userEmail foreign key (userId) references user(utdID)
);

drop table if exists userAcademicStatus;
create table userAcademicStatus(
	status					char(10) not null,
	academicId				int,
	maxItemsAllowed			int,
	maxBorrowDuration		int,
	userId				char(10),
	CHECK (maxItemsAllowed >0 and maxItemsAllowed < 300),
	CONSTRAINT pk_userAcademicStatus primary key (academicId),
	CONSTRAINT fk_userAcademicStatus foreign key (userId) references user(utdID)
);

drop table if exists userAcademicStatusUndergrad;
create table userAcademicStatusUndergrad(
	Id			int,
	major		char(20) not null,		
	CONSTRAINT pk_userAcademicStatusUndergrad primary key (Id,major),
	CONSTRAINT fk_userAcademicStatusUndergrad foreign key (Id) references userAcademicStatus(academicId)
);

drop table if exists userAcademicStatusGrad;
create table userAcademicStatusGrad(
	Id			int,
	major		char(20) not null,		
	CONSTRAINT pk_userAcademicStatusGrad primary key (Id,major),
	CONSTRAINT fk_userAcademicStatusGrad foreign key (Id) references userAcademicStatus(academicId)
);

drop table if exists userAcademicStatusFaculty;
create table userAcademicStatusFaculty(
	Id					int,
	specialization		char(20) not null,		
	CONSTRAINT pk_userAcademicStatusFaculty primary key (Id,specialization),
	CONSTRAINT fk_userAcademicStatusFaculty foreign key (Id) references userAcademicStatus(academicId)
);

drop table if exists itemTypes;
create table itemTypes(
	itemTypeId		int	unique,
	itemIsbn		char(13),
	itemTypeName	char(15),
	itemTypeMaxFine	int,
	copies			int,
	CHECK (itemTypeMaxFine > 0 and copies > 0),
	CONSTRAINT pk_itemTypes primary key (itemTypeId)
);

drop table if exists itemCatalog;
create table itemCatalog(
	isbn				char(13) not null,
	edition				int not null default 1,
	titleHeadline		char(500),
	titleSubHeadline	char(250),
	description			char(1000),
	totalPages			int,
	genre				char(15),
	lccn				char(12),
	itemId				int,
    fiction				boolean,
	CHECK (edition > 0 AND totalPages > 0),
	CONSTRAINT pk_itemCatalog primary key (isbn)
);

drop table if exists itemLocation;
create table itemLocation(
	itemIsbn			char(13),
	locationItemId		int,
	locationFloor		int,
	locationShelve		int,
	CHECK (locationFloor > 0 AND locationShelve > 0),
	CONSTRAINT pk_itemLocation primary key (locationItemId),
	CONSTRAINT fk_itemLocation foreign key (itemIsbn) references itemCatalog(isbn)
);


drop table if exists nonFiction;
create table nonFiction(
	itemIsbn			char(13),
	nonFictionId		int not null auto_increment,
	ddsn				char(13),
	CONSTRAINT pk_nonFiction primary key (nonFictionId),
	CONSTRAINT fk_nonFiction foreign key (itemIsbn) references itemCatalog(isbn)
);

drop table if exists fiction;
create table fiction(
	itemIsbn			char(13),
	fictionId			int not null auto_increment,
	plot				char(255),
	CONSTRAINT pk_fiction primary key (fictionId),
	CONSTRAINT fk_fiction foreign key (itemIsbn) references itemCatalog(isbn)
);


drop table if exists itemTypeRareItem;
create table itemTypeRareItem(
	itemTypeId		int,
	rareItemId		int,
	yearWritten		datetime,
	CONSTRAINT pk_itemTypeRareItem primary key (rareItemId),
	CONSTRAINT fk_itemTypeRareItem foreign key (itemTypeId) references itemTypes(itemTypeId)
);

drop table if exists itemTypePeriodicals;
create table itemTypePeriodicals(
	itemTypeId				int,
	periodicalsId			int,
	frequencyPerMonth		int,
	CONSTRAINT pk_itemTypePeriodicals primary key (periodicalsId),
	CONSTRAINT fk_itemTypePeriodicals foreign key (itemTypeId) references itemTypes(itemTypeId)
);

drop table if exists itemTypeJournal;
create table itemTypeJournal(
	itemTypeId				int,
	journalId				int,
	issn					char(12),
	CONSTRAINT pk_itemTypeJournal primary key (issn),
	CONSTRAINT fk_itemTypeJournal foreign key (itemTypeId) references itemTypes(itemTypeId)
);

drop table if exists itemTypeBooks;
create table itemTypeBooks(
	itemTypeId				int,
	booksId					int,
	price					int,
	CHECK (price > 0),
	CONSTRAINT pk_itemTypeBooks primary key (booksId),
	CONSTRAINT fk_itemTypeBooks foreign key (itemTypeId) references itemTypes(itemTypeId)
);

drop table if exists itemTypeBooksPaperback;
create table itemTypeBooksPaperback(
	paperbackId				int,
	booksId					int,
	CONSTRAINT pk_itemTypeBooksPaperback primary key (paperbackId),
	CONSTRAINT fk_itemTypeBooksPaperback foreign key (booksId) references itemTypeBooks(booksId)
);

drop table if exists itemTypeBooksEbook;
create table itemTypeBooksEbook(
	ebookId					int,
	booksId					int,
	link					char(100),
	CONSTRAINT pk_itemTypeBooksEbook primary key (ebookId),
	CONSTRAINT fk_itemTypeBooksEbook foreign key (booksId) references itemTypeBooks(booksId)
);

drop table if exists itemTypeBooksHardcover;
create table itemTypeBooksHardcover(
	hardcoverId				int,
	booksId					int,
	barcode					char(43),
	CONSTRAINT pk_itemTypeBooksHardcover primary key (hardcoverId),
	CONSTRAINT fk_itemTypeBooksHardcover foreign key (booksId) references itemTypeBooks(booksId)
);

drop table if exists publisher;
create table publisher(
	pId				int not null auto_increment,
    pName			char(50),
	pEmail			char(50),
	pcountryCode	char(2),
	pPhone			char(10),
	CONSTRAINT pk_publisher primary key (pId)
);

drop table if exists publishedItem;
create table publishedItem(
	publisherId			int not null,
	publishedItemIsbn	char(13),
    pDate				datetime,
	CONSTRAINT pk_publishedItem foreign key (publisherId) references publisher(pId),
	CONSTRAINT fk_publishedItem foreign key (publishedItemIsbn) references itemCatalog(isbn)
);

drop table if exists author;
create table author(
	authorId			int not null auto_increment,
    authorName			char(20),
	authorSex			char(1),
	authorAddress		char(70),
	authorPhone			char(10),
	authorCountryCode	char(2),
	CONSTRAINT pk_author primary key (authorId)
);

drop table if exists authoredItem;
create table authoredItem(
	authoredItemSNo		int not null auto_increment,
	authorId			int,
    authorName			char(20),
	authoredIsbn	char(13),
	CONSTRAINT pk_authoredItem primary key (authoredItemSNo),
	CONSTRAINT fk_authorId foreign key (authorId) references author(authorId),
	CONSTRAINT fk_authoredIsbn foreign key (authoredIsbn) references itemCatalog(isbn)
);

drop table if exists borrowedItems;
create table borrowedItems(
	serialno			int,
	itemId				char(13),
	borrowedItemsId		char(10),
	borrowedItemsPeriod	int,
	issueDate			datetime,
	returnDate			datetime,
	CHECK (returnDate > issueDate),
	CHECK (borrowedItemsPeriod > 0),
	CONSTRAINT pk_borrowedItems primary key (serialno),
	CONSTRAINT fk_borrowedItemsuser foreign key (borrowedItemsId) references user(utdId),
	CONSTRAINT fk_borrowedItemsIsbn foreign key (itemId) references itemCatalog(isbn)
);

drop table if exists loan;
create table loan(
	loanId		int,
	finePerDay	int,
	CHECK (finePerDay > 0),
	CONSTRAINT fk_loan foreign key (loanId) references borrowedItems(serialno)
);

Alter table itemTypes
ADD CONSTRAINT fk_itemTypes foreign key (itemIsbn) references itemCatalog(isbn);
Alter table itemCatalog
ADD CONSTRAINT fk_itemCatalogType foreign key (itemId) references itemTypes(itemTypeId);
-- ALTER table publisher
-- ADD CONSTRAINT ck_future_date CHECK (pdate > 400 and pdate < year(CURDATE()));
-- ALTER table borrowedItems
-- ADD CONSTRAINT ck_correct_date CHECK (issueDate > getdate() and returnDate > getdate());
