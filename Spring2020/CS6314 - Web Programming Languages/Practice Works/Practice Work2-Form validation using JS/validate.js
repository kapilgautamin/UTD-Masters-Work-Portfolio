window.onload = function(){

	var span_username = document.createElement("span");
	span_username.innerHTML = "test";
	span_username.className = "info";
	span_username.style.display = "none";

	var username = document.getElementById("username");
	username.parentNode.appendChild(span_username);

	username.onfocus = function(){
		span_username.innerHTML = "The username field must contain only alphanumeric characters.";
		span_username.className = "info";
		span_username.style.display = "inline";
	}

	username.onblur = function(){
		if (username.value.length == 0)
			span_username.style.display = "none";
		const regex = /^\w+$/;
		const found = username.value.match(regex);
		console.log(found);
		if (found){
			span_username.className = "ok";
			span_username.innerHTML = "OK";
		}
		else{
			span_username.className = "error";
			span_username.innerHTML = "Error";
		}
	}

	var span_password = document.createElement("span");
	span_password.innerHTML = "test";
	span_password.className = "info";
	span_password.style.display = "none";

	var password = document.getElementById("password");
	password.parentNode.appendChild(span_password);

	password.onfocus = function(){
		span_password.className = "info";
		span_password.innerHTML = "The password field should be at least six characters long.";
		span_password.style.display = "inline";
	}

	password.onblur = function(){

		if (password.value.length == 0)
			span_password.style.display = "none";
		else if (password.value.length < 6){
			span_password.className = "error";
			span_password.innerHTML = "Error";
		}
		else{
			span_password.className = "ok";
			span_password.innerHTML = "OK";
		}
	}

	var span_email = document.createElement("span");
	span_email.innerHTML = "test";
	span_email.className = "info";
	span_email.style.display = "none";

	var email = document.getElementById("email");
	email.parentNode.appendChild(span_email);

	email.onfocus = function(){
		span_email.innerHTML = "The email field should be a valid email address (abc@def.xyz).";
		span_email.className = "info";
		span_email.style.display = "inline";
	}

	email.onblur = function(){
		if (email.value.length == 0)
			span_email.style.display = "none";
		const regex = /^\w+\.?\w+@\w+\.\w+$/;
		const found = email.value.match(regex);
		//console.log(found);
		if (found) {
			span_email.className = "ok";
			span_email.innerHTML = "OK";
		}
		else{
			span_email.className = "error";
			span_email.innerHTML = "Error";
		}
	}

}


