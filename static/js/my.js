$(document).on('change', '.div-toggle', function() {
  var target = $(this).data('target');
  var show = $("option:selected", this).data('show');
  $(target).children().addClass('hide');
  $(show).removeClass('hide');
});
$(document).ready(function(){
	$('.div-toggle').trigger('change');
});

// Smooth Scroll Effect
document.addEventListener("DOMContentLoaded", function () {
	document.querySelectorAll('a[href^="#"]').forEach(anchor => {
		anchor.addEventListener("click", function (e) {
			e.preventDefault();
			const targetId = this.getAttribute("href");
			const targetElement = document.querySelector(targetId);
			if (targetElement) {
				window.scrollTo({
					top: targetElement.offsetTop - 50,
					behavior: "smooth"
				});
			}
		});
	});
});

// Scroll-to-Top Button Functionality
window.onscroll = function () {
	let scrollBtn = document.getElementById("scrollTop");
	if (document.documentElement.scrollTop > 200) {
		scrollBtn.style.display = "block";
	} else {
		scrollBtn.style.display = "none";
	}
};

function scrollToTop() {
	window.scrollTo({ top: 0, behavior: "smooth" });
}

// Loading Animation
window.onload = function () {
	document.getElementById("loading").style.display = "none";
};
// Navbar Toggle (for mobile view)
let menuBtn = document.getElementById("menuBtn");
let navbar = document.querySelector(".navbar");

menuBtn.addEventListener("click", function () {
    navbar.classList.toggle("active");
});

// Smooth Scroll to Top
let scrollTopBtn = document.getElementById("scrollTop");

window.onscroll = function () {
    if (document.documentElement.scrollTop > 300) {
        scrollTopBtn.style.display = "block";
    } else {
        scrollTopBtn.style.display = "none";
    }
};

scrollTopBtn.addEventListener("click", function () {
    window.scrollTo({ top: 0, behavior: "smooth" });
});

// Preloader
window.addEventListener("load", function () {
    let loader = document.getElementById("loading");
    loader.style.display = "none";
});

// Form Validation
document.querySelector("form").addEventListener("submit", function (event) {
    let inputFields = document.querySelectorAll("input");
    let isValid = true;

    inputFields.forEach((input) => {
        if (input.value.trim() === "") {
            isValid = false;
            input.style.border = "2px solid red";
        } else {
            input.style.border = "2px solid green";
        }
    });

    if (!isValid) {
        event.preventDefault();
        alert("Please fill in all fields correctly.");
    }
});
