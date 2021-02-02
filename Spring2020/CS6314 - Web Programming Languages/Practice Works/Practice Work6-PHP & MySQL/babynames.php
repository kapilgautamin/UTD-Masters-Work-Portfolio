<!DOCTYPE html>
<html>
<head>
	<title>Top-5 Baby Names</title>
	<link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css" integrity="sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/iJTQUOhcWr7x9JvoRxT2MZw1T" crossorigin="anonymous">
	<style type="text/css">
		table{
			border: 1px solid black;
		}
		th:nth-child(1),th:nth-child(6),td:nth-child(1),td:nth-child(6){
			border-right:1px solid black;
		}
		td:nth-child(n){
			background: #eedd82;
		}
		td:nth-child(2n){
			background: yellow;
		}
	</style>
</head>
<body>
<form action="?" method="post">
	<div class='mx-4 my-3'>
		Year : 
		<select name="year">
			<option value="All years">All years</option>
			<option value="2005">2005</option>
			<option value="2006">2006</option>
			<option value="2007">2007</option>
			<option value="2008">2008</option>
			<option value="2009">2009</option>
			<option value="2010">2010</option>
			<option value="2011">2011</option>
			<option value="2012">2012</option>
			<option value="2013">2013</option>
			<option value="2014">2014</option>
			<option value="2015">2015</option>
		</select>
		Gender :
		<select name="gender">
			<option value="Both">Both</option>
			<option value="m">Male</option>
			<option value="f">Female</option>
		</select>
	
	<button class='btn-primary'>Submit</button>
	</div>
</form>

<?
$year = $_POST["year"];
$gender = $_POST["gender"];
echo "<strong class='mx-4'>Now Displaying: Year: $year, Gender: $gender </strong>";

$conn = mysqli_connect("localhost","root","root","HW3");
if (!$conn){
	echo "Database connection failed.";
} else{
	if($year == 'All years' and $gender == 'Both'){
		$sql = "SELECT * FROM BabyNames ORDER BY year desc,gender,ranking;";
	}else if($year == 'All years' and $gender != 'Both'){
		$sql = "SELECT * FROM BabyNames WHERE gender='$gender' ORDER BY year desc,ranking;";
	} else if($gender == 'Both' and $year != 'All years'){
		$sql = "SELECT * FROM BabyNames WHERE year=$year ORDER BY gender,ranking;";
	} else{
		$sql = "SELECT * FROM BabyNames WHERE year=$year 
		AND gender='$gender' ORDER BY ranking;";
	}
	
	// echo $sql . "<br>";
	$result = mysqli_query($conn,$sql);
	if($gender == 'Both'){
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
	} else {
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

</body>
</html>