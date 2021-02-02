<html>
<?
session_start();

$username = $_POST["username"];
$password = $_POST["password"];

if(empty($username) || empty($password)){
	header('Location:page1.html');
} else{
	$conn = mysqli_connect("localhost","root","root","test_class");
	if (!$conn){
		echo "Database connection failed.";
	} else{
		$sql = "SELECT * FROM users WHERE username='$username" AND password='$password';
		echo $sql;
		$result = mysqli_query($conn,$sql);
		$row = mysqli_fetch_array($result);
		echo "username: " . $row["username"] . ", fullname: " . $row["fullname"];
	}
}


?>
</html>