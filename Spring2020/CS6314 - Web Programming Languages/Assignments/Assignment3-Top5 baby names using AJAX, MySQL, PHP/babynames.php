<?
$year = intval($_GET["year"]);
$gender = $_GET["gender"];

$conn = mysqli_connect("localhost","root","root","HW3");
if (!$conn) {
	echo "Database connection failed.";
} else {
	if($year == 'All years' and $gender == 'Both') {
		$sql = "SELECT * FROM BabyNames ORDER BY year desc,gender,ranking;";
	}else if($year == 'All years' and $gender != 'Both') {
		$sql = "SELECT * FROM BabyNames WHERE gender='$gender' ORDER BY year desc,ranking;";
	} else if($gender == 'Both' and $year != 'All years') {
		$sql = "SELECT * FROM BabyNames WHERE year=$year ORDER BY gender,ranking;";
	} else {
		$sql = "SELECT * FROM BabyNames WHERE year=$year 
		AND gender='$gender' ORDER BY ranking;";
	}
	
	// echo $sql . "<br>";
	$result = mysqli_query($conn, $sql);
	if($gender == 'Both'){
		echo "<strong class='mx-4 my-4'>Now Displaying: Year: $year, Gender: $gender </strong>";
		echo("<div class='col-9'>
			<table class='table'>
			  <tr class='table-success'>
			  	<th>Year</th>
			    <th colspan='5' class='text-center'>Female</th>
			    <th colspan='5' class='text-center'>Male</th>
			  </tr>
			  <tr class='table-success'>
			  	<th></th>
			    <th>Rank 1</th>
			    <th>Rank 2</th>
			    <th>Rank 3</th>
			    <th>Rank 4</th>
			    <th>Rank 5</th>

			    <th>Rank 1</th>
			    <th>Rank 2</th>
			    <th>Rank 3</th>
			    <th>Rank 4</th>
			    <th>Rank 5</th>
			  </tr>");
		
		$count = 0;
		while($row = mysqli_fetch_array($result)) {
			if($count == 0){
				echo("<tr><td>".$row['year']."</td>");
			}

			echo("<td>".$row['name']."</td>");
		  	$count += 1;

		  	if($count == 10){
		  		echo("</tr>");
		  		$count = 0;
		  	}
		}
		echo("</table></div>");			
	} else if($gender == 'm' || $gender == 'f'){
		echo "<strong class='mx-4'>Now Displaying: Year: $year, Gender: $gender </strong>";
		echo("<div class='col-5'>
			<table class='table'>
			  <tr class='table-success'>
			  	<th>Year</th>
			    <th colspan='5' class='text-center'>");
		if($gender == 'f')
			echo "Female";
		else
			echo "Male";
		echo("</th>
			  </tr>
			  <tr class='table-success'>
			  	<th></th>
			    <th>Rank 1</th>
			    <th>Rank 2</th>
			    <th>Rank 3</th>
			    <th>Rank 4</th>
			    <th>Rank 5</th>
			  </tr>");
		
		$count = 0;
		while($row = mysqli_fetch_array($result)) {
			if($count == 0){
				echo("<tr><td>".$row['year']."</td>");
			}

			echo("<td>".$row['name']."</td>");
		  	$count += 1;

		  	if($count == 5){
		  		echo("</tr>");
		  		$count = 0;
		  	}
		}
		echo("</table></div>");
	}
}

?>