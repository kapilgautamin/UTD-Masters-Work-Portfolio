//start ajax request
$.ajax({
    url: "js/data.json",
    //force to handle it as text
    dataType: "text",
    success: function(data) {
        //data downloaded so we call parseJSON function 
        //and pass downloaded data
        var img_data = $.parseJSON(data);
        //now json variable contains data in json format
        //let's display a few items
        $.each(img_data,function(idx,img){
        	// console.log(idx,img);
        	var loc = "images/square/" + img.path;
        	var title = img.title;
        	var image = $('<img src = "'+ loc + '" alt = "' + title + '">');
        	$('.gallery').append(image);
        	var offset = 25;
        	mid_width = $(document).width()/2; 
        	$(image).mouseenter(function(){
        		 $(this).addClass("gray");
        		 var big_image_src = "images/medium/" + img.path;

        		 var medium_img = '<img src="'+ big_image_src +'" alt="'+ title +'">';
        		 var big_image = '<div id="preview">' + medium_img;
        		 big_image_place = img.city + ', ' + img.country;
        		 big_image_date = img.taken;
        		 info = '<p>' + title + '<br>'+big_image_place+' ['+big_image_date+']'+'</p>';
        		 
        		 big_image += info + '</div>';

        		 pageX = event.pageX;
        		 if(pageX > mid_width)
        		 	pageX -= mid_width;
        		 else
        		 	pageX += 2 * offset;
        		 pageY = event.pageY - 8 * offset; 
        		 // console.log("mouse enter",pageX,pageY);
        		 $("#preview").css({
        		 	top: pageY,
        		 	left: pageX,
        		 	display: "block"
        		 });
        		 $("body").append(big_image);
        	}).mouseleave(function(){
        		// console.log("mouse leave");
        		$(this).removeClass("gray");
        		$("#preview").remove();
        	}).mousemove(function(){
        		var pageX = event.pageX;
        		// console.log("height",$("#preview img").height(),"width",$("#preview img").width());
        		var preview_width = $("#preview img").width();
        		if(pageX > mid_width)
        			if (preview_width && preview_width < 500)
        		 		pageX -= mid_width - 6*offset;
        		 	else
        		 		pageX -= mid_width;
        		 else
        		 	pageX += 2 * offset;
        		pageY = event.pageY - 8 * offset; 
        		
        		$("#preview").css({
        			top: pageY,
        			left: pageX,
        			display: "block"
        		});
        	})

        	
        });        
    }
});