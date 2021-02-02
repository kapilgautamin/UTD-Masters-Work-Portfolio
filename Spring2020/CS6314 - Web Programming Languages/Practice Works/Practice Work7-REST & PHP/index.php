<?
$conn = mysqli_connect("localhost","root","root","PW7");
if (!$conn){
	echo "Database connection failed.";
} else{
	$sql = "SELECT * FROM Books";

	$result = mysqli_query($conn,$sql);
	
	$rows = array();
	while($r = mysqli_fetch_assoc($result)) {
		$rows[] = $r;
	}
	echo json_encode($rows);
}

?>