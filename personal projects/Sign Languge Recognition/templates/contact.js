window.addEventListener("scroll", scrollActive);

function sendMessage() {
  var name = document.getElementById("name").value;
  var phone = document.getElementById("phone").value; // Corrected variable name
  var email = document.getElementById("email").value;
  var message = document.getElementById("message").value;

  if (name && email && phone && message) {
    // Corrected variable name
    var body =
      "Name: " +
      name +
      "\nEmail: " +
      email +
      "\nPhone no : " +
      phone +
      "\nMessage: " +
      message;
    window.location.href =
      "mailto:pusulurijahnavi15@gmail.com?subject=Contact%20Form&body=" +
      encodeURIComponent(body);
  } else {
    alert("Please fill in all fields.");
  }
}
