<!DOCTYPE html>
<html>
<head>
	<title>Exam2 Orders</title>
	<link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css" integrity="sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/iJTQUOhcWr7x9JvoRxT2MZw1T" crossorigin="anonymous">
	<style type="text/css">
		
	</style>
</head>
<body>
<form action="?" method="post">
<div>
    <label for="name"></label>
    <input type="text" name="name" id="name" placeholder = "Enter here to search">
    <button class='btn-primary' id="submit">Submit</button>
</div>


<?
if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $name = $_POST["name"];

    $conn = mysqli_connect("localhost","root","root","Exam2");
    if (!$conn){
        echo "Database connection failed.";
    } else{
        // echo "Database connected";
        // echo "Searched for ".$name."<br>";

        if($name == 'All'){
            $sql = "SELECT customer.cname,drink.dname,size.sname from orders 
            join customer on customer.cid = orders.cid
            join size on size.sid = orders.sid
            join drink on drink.did = orders.did;";
        }else {
            $sql = "SELECT customer.cname,drink.dname,size.sname from orders 
            join customer on customer.cid = orders.cid
            join size on size.sid = orders.sid
            join drink on drink.did = orders.did
            where customer.cname = '$name';";
        }
        
        $result = mysqli_query($conn,$sql);
        // echo "<ul>";
        while($row = mysqli_fetch_array($result)) {
            echo "<ul>";
            // echo json_encode($row);
            echo "<li> Customer : ".$row[0]."</li>";
            echo "<li> Drink : ".$row[1]."</li>";
            echo "<li> Size : ".$row[2]."</li>";
            echo "</ul>";
        }
        // echo "</ul>";


    }
}
?>

</form>
</body>
</html>