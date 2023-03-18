function getResult(num) {

    var url = "http://localhost:8000";   // The URL and the port number must match server-side
    var endpoint = "/result";            // Endpoint must match server endpoint

    var http = new XMLHttpRequest();

    // prepare GET request
    http.open("GET", url+endpoint, true);


    http.onreadystatechange = function() {
        var DONE = 4;       // 4 means the request is done.
        var OK = 200;       // 200 means a successful return.
        if (http.readyState == DONE && http.status == OK && http.responseText) {

            // JSON string       
             
			var replyString = http.responseText;
			const obj = JSON.parse(replyString);
			var bName = "result" + num;
			document.getElementById(bName).innerHTML += "Answer: " + obj["Answer"];
            // convert JSON string into JavaScript object and get the scores
		






        }
    };

    // Send request
    http.send();
}

function getPrediction() {

    var url = "http://localhost:8000";   // The URL and the port number must match server-side
    var endpoint = "/result_cnn";            // Endpoint must match server endpoint
	
    var http = new XMLHttpRequest();

    // prepare GET request
    http.open("GET", url+endpoint, true);
	

    http.onreadystatechange = function() {
        var DONE = 4;       // 4 means the request is done.
        var OK = 200;       // 200 means a successful return.
        if (http.readyState == DONE && http.status == OK && http.responseText) {

            // JSON string       
             
			var replyString = http.responseText;
			const obj = JSON.parse(replyString);
			
			document.getElementById("resultCNN").innerHTML = "Answer: " + obj["Answer"];

            // convert JSON string into JavaScript object and get the scores
		






        }
    };

    // Send request
    http.send();
}