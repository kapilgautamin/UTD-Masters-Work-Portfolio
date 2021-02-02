<?php
session_start();

if (isset($_SESSION["email"]) && isset($_SESSION["name"])) {
    // remove all session variables
	session_unset();

	// destroy the session
	session_destroy();
}

header("Location: login.html");
?>
