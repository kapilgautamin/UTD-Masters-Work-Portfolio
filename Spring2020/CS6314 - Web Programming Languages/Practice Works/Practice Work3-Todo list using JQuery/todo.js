$(document).ready(function(){
		
		$("#todo > h3").on("click","i",function(){
			$("#new").slideToggle();
			$("#new").val("");
		});

		$("#todo ul").on("mouseenter","li",function(){
			$(this).find("span").fadeIn();
		});

		$("#todo ul").on("mouseleave","li",function(){
			$(this).find("span").fadeOut();
		});

		$('#new').on("focus",function(){
			$(this).css("border","2px solid blue");
		})
		$('#new').on("blur",function(){
			$(this).css("border","2px solid gray");
		})

		$('#new').on("keypress",function(event){
			if (event.which == 13 && $("#new").val().length) {
				//grab new todo text from input
				$("ul").append("<li><span><i class='fa fa-trash'></i></span>" + $("#new").val() + "</li>")
			}
		});

		$("ul").on("click","li",function(){
			$(this).toggleClass("done");
		});

		$("ul").on("click","span",function(){
			$(this).parents("li").remove();
		});
});
