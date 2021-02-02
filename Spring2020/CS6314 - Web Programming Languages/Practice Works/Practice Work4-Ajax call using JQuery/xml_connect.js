function loadDoc() {
  var xhttp = new XMLHttpRequest();
  xhttp.onreadystatechange = function() {
    if (this.readyState == 4 && this.status == 200) {
    myFunction(this);
    }
  };
  xhttp.open("GET", "books.xml", true);
  xhttp.send();
}
function myFunction(xml) {
  var i;
  var xmlDoc = xml.responseXML;
  var table="<tr><th>Title</th><th>Authors</th><th>Year</th><th>Price</th><th>Category</th></tr>";
  var x = xmlDoc.getElementsByTagName("book");
  for (i = 0; i < x.length; i++) {
    table += "<tr><td>" +
    x[i].getElementsByTagName("title")[0].childNodes[0].nodeValue +
    "</td><td>";
    
    authors = x[i].getElementsByTagName("author");
    // console.log(authors)
    for (j = 0;j < authors.length - 1;j++){
      table += authors[j].childNodes[0].nodeValue + ", ";
    }
    table += authors[j].childNodes[0].nodeValue + "</td>";
    table += "<td>" +
    x[i].getElementsByTagName("year")[0].childNodes[0].nodeValue +
    "</td><td>" +
    x[i].getElementsByTagName("price")[0].childNodes[0].nodeValue +
    "</td><td>" +
    x[i].getAttribute("category") +
    "</td></tr>";
  }
  document.getElementById("load_table").innerHTML = table;
}