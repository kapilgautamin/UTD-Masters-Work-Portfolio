import javax.swing.*;
import javax.swing.table.DefaultTableModel;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.FocusAdapter;
import java.awt.event.FocusEvent;
import java.sql.*;
// Created by Kapil Gautam 
// CS6363.004 Database Design Assignment#4
// This is a GUI program file for a SQL database Library which offers
// GUI tools to insert/search for a book using various inputs supported by the updates
// in Status bar and protected against invalid inputs and SQL injection attacks.
// The program does not take delete or update operations as Professor Cole relaxed the criterion
// and asked to make the program relatively simpler by taking care of insertion operations only.
//Using swing components to create the GUI in Java Language
public class GUI {

    private JButton saveButton;
    private JButton clearButton;
    private JButton searchButton;
    private JLabel titleLabel;
    private JLabel authorSLabel;
    private JLabel ISBNLabel;
    private JLabel deweyNumberLabel;
    private JLabel publisherLabel;
    private JPanel LibraryView;
    private JTextField isbnText;
    private JTextField deweyText;
    private JTextField titleText;
    private JTextField authorText;
    private JComboBox publisherDropDown;
    private JTextField statusBar;
    private JTable outputTable;
    private Connection SQLConnection;
    private final int IntegrityConstraintViolation = -1;
    private final int SyntaxError = -2;
    private DefaultTableModel model = new DefaultTableModel();
    private String[] table_default_columns = {"ISBN", "Title", "Authors"};

    //Function to execute the SELECT queries
	// The function creates a connection with the SQL server if not already created and then exceutes the 
	// select operation on that query.
    private ResultSet query_database(String query) {
        try {
            if (SQLConnection == null) {
                String host = "jdbc:mysql://localhost:3306/library";
                String user = "root";
                String password = "kapil183";
                SQLConnection = DriverManager.getConnection(host, user, password);
                System.out.println("SQL server connected.");
            }
            Statement stmt = SQLConnection.createStatement();
            //String sql = "select * from author";
            System.out.println(query);
            return stmt.executeQuery(query);
        } catch (SQLException e) {
            statusBar.setText(e.getMessage());
            e.printStackTrace();
        }
        return null;
    }

    //Function to execute the INSERT,UPDATE,DELETE queries
	// The function creates a connection with the SQL server if not already created and then exceutes the 
	// insert/update/delete operation on that query.
    private int change_database(String query) {
        try {
            if (SQLConnection == null) {
                String host = "jdbc:mysql://localhost:3306/library";
                String user = "root";
                String password = "kapil183";
                SQLConnection = DriverManager.getConnection(host, user, password);
                System.out.println("SQL server connected.");
            }
            Statement stmt = SQLConnection.createStatement();
            //Insert, update, delete statements
            System.out.println(query);
            return stmt.executeUpdate(query);
		// catch multiple exceptions and update the user with the problem
        } catch (SQLIntegrityConstraintViolationException e1) {
            e1.printStackTrace();
            System.out.println(e1.getMessage());
            statusBar.setText(e1.getMessage());
            return IntegrityConstraintViolation;
        } catch (SQLSyntaxErrorException e2) {
            e2.printStackTrace();
            System.out.println(e2.getMessage());
            statusBar.setText("Please enter a valid string");
            return SyntaxError;
        } catch (SQLException e) {
            e.printStackTrace();
            System.out.println(e.getMessage());
            statusBar.setText(e.getMessage());
            return -999;
        }
    }

    //Main function to create the GUI
    public static void main(String[] args) {
        JFrame frame = new JFrame("Library Book Entry");
        frame.setContentPane(new GUI().LibraryView);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.pack();
        frame.setVisible(true);

    }

    public GUI() {
        //ActionListener for the save Button in GUI
        saveButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                statusBar.setText("Saving the book information.");

                //Trim all the inputs for any leading or trailing spaces
                String title = titleText.getText().trim();     //itemCatalog (titleHeadline)
                String author = authorText.getText().trim();   //author (authorIsbn,authorName)
				//If there are multiple authors then we split them.
                String[] authors = author.split(";");
                String isbn = isbnText.getText().trim();      //itemCatalog(isbn)
                String dewey = deweyText.getText().trim();     //nonFiction(itemIsbn, ddsn)
                String publisher = publisherDropDown.getSelectedItem().toString();    //publisher -> publishedItem

                // Process the input data to avoid SQL injection attacks
                if (title.contains("\"") || author.contains("\"") || isbn.contains("\"") || publisher.contains("\"") ||
                        title.contains("--") || author.contains("--") || isbn.contains("--") || publisher.contains("--"))
                    statusBar.setText("Please enter a valid value.");
                else if (title.isEmpty())
                    statusBar.setText("Please enter something in title.");
                else if (isbn.isEmpty() || isbn.length() != 13)
                    statusBar.setText("Please enter a valid ISBN number");
                else if (author.isEmpty())
                    statusBar.setText("Please enter author(s) seperated by a ';'");
                else if (publisher.isEmpty())
                    statusBar.setText("Enter the publisher");
                else if (!dewey.isEmpty() && dewey.length() != 13)
                    statusBar.setText("For a non-fiction book, the dewey decimal system takes 13 characters.");
                else {

                    int out;
                    String query1 = "insert into library.itemCatalog (isbn,titleHeadline,fiction) values (\"" + isbn + "\",\"" + title + "\"," + dewey.isEmpty() + ");";
                    //String query1 = "select * from author";
                    out = change_database(query1);
                    // Process the data only if there are no integrity violations
                    if (out > 0) {
                        if (authors.length > 1)
                            for (String a : authors)
                                change_database("insert into author (authoredIsbn,authorName) values (\"" + isbn + "\",\"" + a.trim() + "\");");
                        else
                            change_database("insert into author (authoredIsbn,authorName) values (\"" + isbn + "\",\"" + author.trim() + "\");");

                        if (!dewey.isEmpty())
                            change_database("insert into nonFiction (itemIsbn, ddsn) values (\"" + isbn + "\",\"" + dewey + "\");");
                        else
                            change_database("insert into fiction (itemIsbn) values (\"" + isbn + "\");");

                        if (!publisher.isEmpty()) {
                            try {
                                ResultSet rs = query_database("select pid from publisher where pName = \"" + publisher + "\";");
                                rs.next();
                                int pId = rs.getInt("pId");
                                System.out.println(pId + " " + publisher);
                                change_database("insert into publishedItem(publisherId,publishedItemIsbn,pDate) values (" + pId + ",\"" + isbn + "\", curdate())");
                            } catch (SQLException ex) {
                                ex.printStackTrace();
                            }
                        }
                        //Once the data is processed, show it on the GUI on the status bar
                        model.addRow(new Object[]{isbn, title, author});

                    }
                }
            }
        });

        //Action Listener for the search button in the GUI
        searchButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
				//Keep updating the user about the operations being performed.
                statusBar.setText("Searching books for the given information.");
                int row_count = outputTable.getRowCount();
                //System.out.println(row_count);
                for (int r = 0; r < row_count; r++)
                    model.removeRow(0);


                //Trim all the input for any leading or trailing spaces
                String title = titleText.getText().trim();     //itemCatalog (titleHeadline)
                String author = authorText.getText().trim();   //author (authorIsbn,authorName)
                String[] authors = author.split(";");
                String isbn = isbnText.getText().trim();     //itemCatalog(isbn)
                String dewey = deweyText.getText().trim();     //nonFiction(itemIsbn, ddsn)
                String publisher = publisherDropDown.getSelectedItem().toString();    //publisher -> publishedItem

                // Process the input data to avoid SQL injection attacks
                if (title.contains("\"") || author.contains("\"") || isbn.contains("\"") || publisher.contains("\"") ||
                        title.contains("--") || author.contains("--") || isbn.contains("--") || publisher.contains("--"))
                    statusBar.setText("Please enter a valid value to search.");
                else if (!isbn.isBlank() && isbn.length() != 13)
                    statusBar.setText("Please enter a valid ISBN number to search");
                else if (!dewey.isBlank() && dewey.length() != 13)
                    statusBar.setText("The dewey decimal system takes 13 characters to search.");
                else {
                    //The data could also have been replaced for input in SQL query for further protection.
                    //title = new String(title.trim().replace("\"", "").replace("--", ""));
                    //author = new String(author.trim().replace("\"", "").replace("--", ""));
                    //isbn = new String(isbn.trim().replace("\"", "").replace("--", ""));
                    //dewey = new String(dewey.trim().replace("\"", "").replace("--", ""));

                    try {
                        //Listing all the queries for the specific items
                        String title_query = "select isbn,titleHeadline,authorName from itemCatalog join author " +
                                "on authoredIsbn=isbn where titleHeadline like \"%" + title + "%\"";
                        String author_query = "select isbn,titleHeadline,authorName from itemCatalog join author " +
                                "on authoredIsbn=isbn where authorName like \"%" + author + "%\"";
                        String isbn_query = "select isbn,titleHeadline,authorName from itemCatalog join author " +
                                "on authoredIsbn=isbn where isbn=\"" + isbn + "\"";
                        String dewey_query = "select isbn,titleHeadline,authorName from itemCatalog join author " +
                                "on authoredIsbn=isbn join nonFiction on isbn=itemIsbn where ddsn=\"" + dewey + "\"";

                        //Creating combinations of different queries and returning distinct values using UNION and join condtions
                        String query = "";
                        if (!title.isBlank() && !author.isBlank() && !isbn.isBlank() && !dewey.isBlank()) {
                            query = title_query + " union " + author_query + " union " + isbn_query + " union " + dewey_query;
                        } else if (!title.isBlank() && !author.isBlank() && !isbn.isBlank()) {
                            query = title_query + " union " + author_query + " union " + isbn_query;
                        } else if (!title.isBlank() && !author.isBlank() && !dewey.isBlank()) {
                            query = title_query + " union " + author_query + " union " + dewey_query;
                        } else if (!title.isBlank() && !dewey.isBlank() && !isbn.isBlank()) {
                            query = title_query + " union " + dewey_query + " union " + isbn_query;
                        } else if (!author.isBlank() && !dewey.isBlank() && !isbn.isBlank()) {
                            query = author_query + " union " + dewey_query + " union " + isbn_query;
                        } else if (!title.isBlank() && !author.isBlank()) {
                            query = title_query + " union " + author_query;
                        } else if (!title.isBlank() && !isbn.isBlank()) {
                            query = title_query + " union " + isbn_query;
                        } else if (!title.isBlank() && !dewey.isBlank()) {
                            query = title_query + " union " + isbn_query;
                        } else if (!author.isBlank() && !isbn.isBlank()) {
                            query = author_query + " union " + isbn_query;
                        } else if (!author.isBlank() && !dewey.isBlank()) {
                            query = author_query + " union " + dewey_query;
                        } else if (!title.isBlank() && !isbn.isBlank()) {
                            query = title_query + " union " + isbn_query;
                        } else if (!isbn.isBlank() && !dewey.isBlank()) {
                            query = isbn_query + " union " + dewey_query;
                        } else if (!title.isBlank()) {
                            query = title_query;
                        } else if (!author.isBlank()) {
                            query = author_query;
                        } else if (!isbn.isBlank()) {
                            query = isbn_query;
                        } else if (!dewey.isBlank()) {
                            query = dewey_query;
                        }

                        query += ";";
                        ResultSet search = query_database(query);
                        while (search.next()) {
                            String table_isbn = search.getString("isbn");
                            String table_title = search.getString("titleHeadline");
                            String table_author = search.getString("authorName");
                            model.addRow(new Object[]{table_isbn, table_title, table_author});
                        }
                        statusBar.setText(model.getRowCount() + " items found");
                    } catch (SQLException ex) {
                        ex.printStackTrace();
                        System.out.println(ex.getMessage());
                        statusBar.setText(ex.getMessage());
                    }
                }
            }
        });

        //Action Listener for clear button in GUI
        clearButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                statusBar.setText("You can enter or search for a book");
                int row_count = outputTable.getRowCount();
                System.out.println(row_count);
                for (int r = 0; r < row_count; r++) {
                    System.out.println(model.getValueAt(0, 0));
                    model.removeRow(0);
                }
                titleText.setText("");
                authorText.setText("");
                isbnText.setText("");
                deweyText.setText("");
                publisherDropDown.setSelectedIndex(0);
            }
        });


        //It is being used as the preprocessing step for intialising the GUI
        titleText.addFocusListener(new FocusAdapter() {
            @Override
            public void focusGained(FocusEvent e) {
                super.focusGained(e);
                model.setColumnIdentifiers(table_default_columns);
                outputTable.setModel(model);
                statusBar.setText("You can enter or search for a book");
                //System.out.println(query_database("select * from author"));
                String query = "select pName from publisher";
                publisherDropDown.addItem("");
                try {
                    ResultSet rs = query_database(query);
                    while (rs.next()) {
                        publisherDropDown.addItem(rs.getString("pName"));
                    }
                } catch (SQLException ex) {
                    ex.printStackTrace();
                }

            }
        });
    }

    {
// GUI initializer generated by IntelliJ IDEA GUI Designer
// >>> IMPORTANT!! <<<
// DO NOT EDIT OR ADD ANY CODE HERE!
        $$$setupUI$$$();
    }

    /**
     * Method generated by IntelliJ IDEA GUI Designer
     * >>> IMPORTANT!! <<<
     * DO NOT edit this method OR call it in your code!
     *
     * @noinspection ALL
     */
    private void $$$setupUI$$$() {
        LibraryView = new JPanel();
        LibraryView.setLayout(new com.intellij.uiDesigner.core.GridLayoutManager(8, 14, new Insets(5, 5, 10, 10), -1, -1));
        authorSLabel = new JLabel();
        authorSLabel.setHorizontalAlignment(4);
        authorSLabel.setHorizontalTextPosition(11);
        authorSLabel.setText("Author(s)");
        LibraryView.add(authorSLabel, new com.intellij.uiDesigner.core.GridConstraints(1, 0, 1, 1, com.intellij.uiDesigner.core.GridConstraints.ANCHOR_EAST, com.intellij.uiDesigner.core.GridConstraints.FILL_NONE, com.intellij.uiDesigner.core.GridConstraints.SIZEPOLICY_CAN_SHRINK | com.intellij.uiDesigner.core.GridConstraints.SIZEPOLICY_CAN_GROW, com.intellij.uiDesigner.core.GridConstraints.SIZEPOLICY_CAN_SHRINK | com.intellij.uiDesigner.core.GridConstraints.SIZEPOLICY_CAN_GROW, null, new Dimension(83, 18), null, 0, false));
        ISBNLabel = new JLabel();
        ISBNLabel.setHorizontalAlignment(4);
        ISBNLabel.setHorizontalTextPosition(11);
        ISBNLabel.setText("ISBN");
        LibraryView.add(ISBNLabel, new com.intellij.uiDesigner.core.GridConstraints(2, 0, 1, 1, com.intellij.uiDesigner.core.GridConstraints.ANCHOR_EAST, com.intellij.uiDesigner.core.GridConstraints.FILL_NONE, com.intellij.uiDesigner.core.GridConstraints.SIZEPOLICY_CAN_SHRINK | com.intellij.uiDesigner.core.GridConstraints.SIZEPOLICY_CAN_GROW, com.intellij.uiDesigner.core.GridConstraints.SIZEPOLICY_CAN_SHRINK | com.intellij.uiDesigner.core.GridConstraints.SIZEPOLICY_CAN_GROW, null, new Dimension(83, 18), null, 0, false));
        deweyNumberLabel = new JLabel();
        deweyNumberLabel.setHorizontalAlignment(4);
        deweyNumberLabel.setHorizontalTextPosition(11);
        deweyNumberLabel.setText("Dewey Number");
        LibraryView.add(deweyNumberLabel, new com.intellij.uiDesigner.core.GridConstraints(3, 0, 1, 1, com.intellij.uiDesigner.core.GridConstraints.ANCHOR_EAST, com.intellij.uiDesigner.core.GridConstraints.FILL_NONE, com.intellij.uiDesigner.core.GridConstraints.SIZEPOLICY_CAN_SHRINK | com.intellij.uiDesigner.core.GridConstraints.SIZEPOLICY_CAN_GROW, com.intellij.uiDesigner.core.GridConstraints.SIZEPOLICY_CAN_SHRINK | com.intellij.uiDesigner.core.GridConstraints.SIZEPOLICY_CAN_GROW, null, new Dimension(83, 18), null, 0, false));
        publisherLabel = new JLabel();
        publisherLabel.setHorizontalAlignment(4);
        publisherLabel.setHorizontalTextPosition(11);
        publisherLabel.setText("Publisher");
        LibraryView.add(publisherLabel, new com.intellij.uiDesigner.core.GridConstraints(4, 0, 1, 1, com.intellij.uiDesigner.core.GridConstraints.ANCHOR_EAST, com.intellij.uiDesigner.core.GridConstraints.FILL_NONE, com.intellij.uiDesigner.core.GridConstraints.SIZEPOLICY_CAN_SHRINK | com.intellij.uiDesigner.core.GridConstraints.SIZEPOLICY_CAN_GROW, com.intellij.uiDesigner.core.GridConstraints.SIZEPOLICY_CAN_SHRINK | com.intellij.uiDesigner.core.GridConstraints.SIZEPOLICY_CAN_GROW, null, new Dimension(83, 18), null, 0, false));
        titleLabel = new JLabel();
        titleLabel.setHorizontalAlignment(4);
        titleLabel.setHorizontalTextPosition(11);
        titleLabel.setText("Title");
        LibraryView.add(titleLabel, new com.intellij.uiDesigner.core.GridConstraints(0, 0, 1, 1, com.intellij.uiDesigner.core.GridConstraints.ANCHOR_EAST, com.intellij.uiDesigner.core.GridConstraints.FILL_NONE, com.intellij.uiDesigner.core.GridConstraints.SIZEPOLICY_CAN_SHRINK | com.intellij.uiDesigner.core.GridConstraints.SIZEPOLICY_CAN_GROW, com.intellij.uiDesigner.core.GridConstraints.SIZEPOLICY_CAN_SHRINK | com.intellij.uiDesigner.core.GridConstraints.SIZEPOLICY_CAN_GROW, null, new Dimension(83, 18), null, 1, false));
        titleText = new JTextField();
        titleText.setText("Enter title for the book");
        LibraryView.add(titleText, new com.intellij.uiDesigner.core.GridConstraints(0, 1, 1, 13, com.intellij.uiDesigner.core.GridConstraints.ANCHOR_WEST, com.intellij.uiDesigner.core.GridConstraints.FILL_HORIZONTAL, com.intellij.uiDesigner.core.GridConstraints.SIZEPOLICY_WANT_GROW, com.intellij.uiDesigner.core.GridConstraints.SIZEPOLICY_FIXED, null, new Dimension(50, -1), new Dimension(-1, 40), 0, false));
        authorText = new JTextField();
        authorText.setText("Enter name of author(s) seperated by ';'");
        LibraryView.add(authorText, new com.intellij.uiDesigner.core.GridConstraints(1, 1, 1, 13, com.intellij.uiDesigner.core.GridConstraints.ANCHOR_WEST, com.intellij.uiDesigner.core.GridConstraints.FILL_HORIZONTAL, com.intellij.uiDesigner.core.GridConstraints.SIZEPOLICY_WANT_GROW, com.intellij.uiDesigner.core.GridConstraints.SIZEPOLICY_FIXED, null, new Dimension(50, -1), null, 0, false));
        isbnText = new JTextField();
        isbnText.setText("Enter a 13 length ISBN number");
        LibraryView.add(isbnText, new com.intellij.uiDesigner.core.GridConstraints(2, 1, 1, 12, com.intellij.uiDesigner.core.GridConstraints.ANCHOR_WEST, com.intellij.uiDesigner.core.GridConstraints.FILL_HORIZONTAL, com.intellij.uiDesigner.core.GridConstraints.SIZEPOLICY_WANT_GROW, com.intellij.uiDesigner.core.GridConstraints.SIZEPOLICY_FIXED, null, new Dimension(50, -1), null, 0, false));
        deweyText = new JTextField();
        deweyText.setText("Enter this number for non-fiction books");
        LibraryView.add(deweyText, new com.intellij.uiDesigner.core.GridConstraints(3, 1, 1, 10, com.intellij.uiDesigner.core.GridConstraints.ANCHOR_WEST, com.intellij.uiDesigner.core.GridConstraints.FILL_HORIZONTAL, com.intellij.uiDesigner.core.GridConstraints.SIZEPOLICY_WANT_GROW, com.intellij.uiDesigner.core.GridConstraints.SIZEPOLICY_FIXED, null, new Dimension(30, -1), null, 0, false));
        publisherDropDown = new JComboBox();
        LibraryView.add(publisherDropDown, new com.intellij.uiDesigner.core.GridConstraints(4, 1, 1, 1, com.intellij.uiDesigner.core.GridConstraints.ANCHOR_WEST, com.intellij.uiDesigner.core.GridConstraints.FILL_HORIZONTAL, com.intellij.uiDesigner.core.GridConstraints.SIZEPOLICY_CAN_GROW, com.intellij.uiDesigner.core.GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        final JPanel panel1 = new JPanel();
        panel1.setLayout(new FlowLayout(FlowLayout.LEFT, 5, 5));
        LibraryView.add(panel1, new com.intellij.uiDesigner.core.GridConstraints(5, 0, 1, 9, com.intellij.uiDesigner.core.GridConstraints.ANCHOR_CENTER, com.intellij.uiDesigner.core.GridConstraints.FILL_BOTH, com.intellij.uiDesigner.core.GridConstraints.SIZEPOLICY_CAN_SHRINK | com.intellij.uiDesigner.core.GridConstraints.SIZEPOLICY_CAN_GROW, com.intellij.uiDesigner.core.GridConstraints.SIZEPOLICY_CAN_SHRINK | com.intellij.uiDesigner.core.GridConstraints.SIZEPOLICY_CAN_GROW, null, null, null, 10, false));
        saveButton = new JButton();
        saveButton.setText("Save");
        panel1.add(saveButton);
        searchButton = new JButton();
        searchButton.setText("Search");
        panel1.add(searchButton);
        clearButton = new JButton();
        clearButton.setText("Clear");
        panel1.add(clearButton);
        statusBar = new JTextField();
        statusBar.setEditable(false);
        LibraryView.add(statusBar, new com.intellij.uiDesigner.core.GridConstraints(7, 0, 1, 14, com.intellij.uiDesigner.core.GridConstraints.ANCHOR_WEST, com.intellij.uiDesigner.core.GridConstraints.FILL_HORIZONTAL, com.intellij.uiDesigner.core.GridConstraints.SIZEPOLICY_WANT_GROW, com.intellij.uiDesigner.core.GridConstraints.SIZEPOLICY_FIXED, null, new Dimension(150, -1), null, 2, false));
        final JScrollPane scrollPane1 = new JScrollPane();
        LibraryView.add(scrollPane1, new com.intellij.uiDesigner.core.GridConstraints(6, 0, 1, 2, com.intellij.uiDesigner.core.GridConstraints.ANCHOR_CENTER, com.intellij.uiDesigner.core.GridConstraints.FILL_BOTH, com.intellij.uiDesigner.core.GridConstraints.SIZEPOLICY_CAN_SHRINK | com.intellij.uiDesigner.core.GridConstraints.SIZEPOLICY_WANT_GROW, com.intellij.uiDesigner.core.GridConstraints.SIZEPOLICY_CAN_SHRINK | com.intellij.uiDesigner.core.GridConstraints.SIZEPOLICY_WANT_GROW, null, null, null, 2, false));
        outputTable = new JTable();
        scrollPane1.setViewportView(outputTable);
    }

    /**
     * @noinspection ALL
     */
    public JComponent $$$getRootComponent$$$() {
        return LibraryView;
    }
}
