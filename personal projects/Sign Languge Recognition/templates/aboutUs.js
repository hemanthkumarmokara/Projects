// JavaScript for review slideshow
let reviews = document.querySelectorAll(".review");
let index = 0;

function showReview() {
  reviews.forEach((review) => (review.style.display = "none")); // Hide all reviews
  index = (index + 1) % reviews.length; // Increment index cyclically
  reviews[index].style.display = "block"; // Show current review
}

showReview();

setInterval(showReview, 5000);
