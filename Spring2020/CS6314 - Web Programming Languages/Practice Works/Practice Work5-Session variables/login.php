<?php
// Start the session
session_start();

if ($_SERVER["REQUEST_METHOD"] == "POST") {
  $name = $email = $password = "";
  
  $name = test_input($_POST["name"]);
  $email = test_input($_POST["email"]);
  $password = test_input($_POST["password"]);

  echo $name . " " . strlen($name) . "<br>";
  echo $email . " " . strlen($email) . "<br>";
  echo $password . " " . strlen($password) . "<br>";

  if (!strlen($name) || !preg_match("/^[\w* ]*$/",$name)) {
  	echo "Entering name error<br>";
    header( "Location: login.html" );
  } elseif (!strlen($email) || !preg_match("/^[\w*.]*[@]\w*[.]\w*$/",$email)) {
  	echo "Entering email error<br>";
    header( "Location: login.html" );
  } elseif (!strlen($password) || strlen($password)<6) {
  	echo "Entering password error<br>";
    header( "Location: login.html" );
  } elseif ($_SESSION["email"] === $email && $_SESSION["name"] === $name) {
    //This is for the case when use relogins with the same email and name using submit button to not reset the session variables
    //Not checking password - check below else section for explaination
    echo "valid user";
    header( "Location: welcome.php" );
  } else {
    
    // Set session variables
    $_SESSION["email"] = $email;
    $_SESSION["name"] = $name;
    //Should not store password info to recheck if the user is re-logging from login.html.
    //We could have stored a token for that purpose, but that would require a database to retrieve that token and verify it, which currently is out of scope of this practice example, so we would only check with name and email
    // $_SESSION["token"] = md5(uniqid());
    $_SESSION['visits'] = 0;
    // echo "Session variables are set.";
    header( "Location: welcome.php" );
  }
} else {
	header( "Location: login.html" );
}

function test_input($data) {
  $data = trim($data);
  $data = stripslashes($data);
  $data = htmlspecialchars($data);
  return $data;
}
?>