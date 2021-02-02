<?php
session_start();

if (isset($_SESSION["email"]) && isset($_SESSION["name"])) {

    echo "Welcome " . $_SESSION["name"] . "<br>";
    $_SESSION['visits'] += 1;
    $ran = rand(1,10);
	// echo "Random gif " . $ran . "<br>";
	$src = '"avatars/' . $ran . '.gif"';
	echo "Visits " . $_SESSION['visits'] . "<br>";
	echo "<div><img src=" . $src . "></div>";
	// echo '<pre>';
 //    print_r($_SESSION);
 //    echo '</pre>';
} else {
    echo "Please log in first to see this page.";
    header("Location: login.html");
}


?>

<div>
<a href="logout.php">Sign Out</a>
</div>