console.log("connected");

$("#submit").click(function(){
	// console.log("button clicked");
	var year = $("#year_dropdown").val();
	var gender = $("#gender_dropdown").val();
	// console.log(year,gender);

	if (window.XMLHttpRequest) {
		xmlhttp = new XMLHttpRequest();
		xmlhttp.onreadystatechange = function() {
			if (xmlhttp.readyState == 4 && xmlhttp.status == 200) {
				$("#table").html(xmlhttp.responseText);
			}
		}
		xmlhttp.open("GET","babynames.php?year=" + year + "&gender=" + gender,true);
		xmlhttp.send();
	}
});
