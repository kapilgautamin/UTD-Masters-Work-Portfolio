
import java.sql.*;

public class DBTest
{
static Connection conn = null;
	/**
	 * If this doesn't work, do the following (from StackOverflow):
         * 
         * right click on
         * project properties --> Java Build Path --> Libraries --> add Jar to your
         * project(which you already downloaded)  This is the MySQL JDBC
         * driver.  This worked for me in NetBeans.
	 */
	public static void main(String[] args)
	{
	try
		{
                Class.forName("com.mysql.jdbc.Driver").newInstance();
		conn = DriverManager.getConnection("jdbc:mysql://localhost/world?user=root&password=SQL1");
		Statement stmt = conn.createStatement();
		ResultSet rs = stmt.executeQuery("SELECT * FROM CITY;");
		while (rs.next())
			{
			System.out.println(rs.getString("name"));
			}
		rs.close();
		System.out.println("Success!!");
		}
	catch(Exception ex)
		{
		System.out.println("Error in connection: " + ex.getMessage());
		}
	}

}
