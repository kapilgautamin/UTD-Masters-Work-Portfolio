/*
 * Sample program to connect to a SQL Server database and read information
 * from a table.  
 */
package dbconnect;

import java.*;
import java.sql.*;

/**
 *
 * @author John Cole
 */
public class DBConnectMain
  {

  private java.sql.Connection con = null;
  private final String url = "jdbc:sqlserver://";
  private final String serverName = "Infopoint104";
  private final String userName = "sa";
  private final String password = "MyPassword";
  // Informs the driver to use server a side-cursor,
  // which permits more than one active statement
  // on a connection.
  private final String selectMethod = "cursor";
  private static String strSql;
  private static Statement stmtSQL;
  private String strdata;

  // Constructor
  public DBConnectMain()
    {
    }

  // Get the connection string.  Often this is read from a configuration
  // file.
  private String getConnectionUrl()
    {
//???          return "jdbc:sqlserver://localhost:1433;instanceName=Infopoint104;databaseName=Company;integratedSecurity=true";
    return "jdbc:sqlserver://INFOPOINT104\\Infopoint104;databaseName=company";

    }

  // Return a connection to a database, or null if one cannot be found.
  private java.sql.Connection getConnection()
    {
    try
      {
      // Load the driver. This is specific to Microsoft SQL Server.
      Class.forName("com.microsoft.sqlserver.jdbc.SQLServerDriver");
      System.out.println(getConnectionUrl());
      // Use a static method of DriverManager to get a connectio to the
      // database.
      con = java.sql.DriverManager.getConnection(getConnectionUrl(), userName, password);
      if (con != null)
        {
        System.out.println("Connection Successful!");
        }
      } catch (Exception e)
      {
      e.printStackTrace();
      System.out.println("Error Trace in getConnection() : " + e.getMessage());
      }
    return con;
    }

  /*
   * Display the driver properties, database details
   */
  public void displayDbProperties()
    {
    java.sql.DatabaseMetaData dm = null;
    java.sql.ResultSet rs = null;
    try
      {
      con = this.getConnection();
      if (con != null)
        {
        dm = con.getMetaData();
        System.out.println("Driver Information");
        System.out.println("\tDriver Name: " + dm.getDriverName());
        System.out.println("\tDriver Version: " + dm.getDriverVersion());
        System.out.println("\nDatabase Information ");
        System.out.println("\tDatabase Name: " + dm.getDatabaseProductName());
        System.out.println("\tDatabase Version: " + dm.getDatabaseProductVersion());
        System.out.println("Avalilable Catalogs ");
        rs = dm.getCatalogs();
        // Show all databases within the connection.
        while (rs.next())
          {
          System.out.println("\tcatalog: " + rs.getString(1));
          }
        rs.close();
        rs = null;

        // Create a SQL statement object and run a query against it to return
        // all employees in last name order.
        stmtSQL = con.createStatement();
        ResultSet rs1 = stmtSQL.executeQuery("SELECT * FROM Employee order by lname");

        // Read all records in the result set and show info.
        while (rs1.next())
          {
          strdata = rs1.getString("fname") + " " + rs1.getString("lname") + " "
                  + rs1.getInt("dno");
          System.out.println(strdata);
          }

        closeConnection();
        } else
        {
        System.out.println("Error: No active Connection");
        }
      } catch (Exception e)
      {
      e.printStackTrace();
      }
    dm = null;
    }

  private void closeConnection()
    {
    try
      {
      if (con != null)
        {
        con.close();
        }
      con = null;
      }
    catch (Exception e)
      {
      e.printStackTrace();
      }
    }

  public static void main(String[] args) throws Exception
    {
    DBConnectMain myDbTest = new DBConnectMain();
    myDbTest.displayDbProperties();
    }
  }
