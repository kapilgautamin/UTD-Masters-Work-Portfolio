<?
$conn = mysqli_connect("localhost","root","root","PW7");
if (!$conn){
	echo "Database connection failed.";
} else{
	$url = $_SERVER['REQUEST_URI'];
	$var = explode('/',$url);
	// print(json_encode(explode('/',$url)));
	// print(sizeof($var));
	if(sizeof($var) > 2 && isset($var[2]) && $var[2] != null){
		// print("Got id".$var[2]);
		$sql = "SELECT * FROM Books where ISBN=\"". $var[2]. "\";";
		$result = mysqli_query($conn,$sql);
		$r = mysqli_fetch_row($result);
		echo json_encode($r);
	} else{
		$sql = "SELECT * FROM Books";

		$result = mysqli_query($conn,$sql);
		
		$rows = array();
		while($r = mysqli_fetch_assoc($result)) {
			$rows[] = $r;
		}
		echo json_encode($rows);
	}
}
?>