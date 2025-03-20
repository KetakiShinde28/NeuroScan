document.addEventListener("DOMContentLoaded", function () {

    /* ✅ Registration Form Validation */
    const registerForm = document.getElementById("register-form");
    if (registerForm) {
        registerForm.addEventListener("submit", function (event) {
            event.preventDefault();

            let name = document.getElementById("name").value.trim();
            let email = document.getElementById("email").value.trim();
            let password = document.getElementById("password").value.trim();
            let confirmPassword = document.getElementById("confirm-password").value.trim();

            if (name === "" || email === "" || password === "" || confirmPassword === "") {
                alert("All fields are required.");
                return;
            }

            if (password !== confirmPassword) {
                alert("Passwords do not match.");
                return;
            }

            alert("Registration Successful!");
            window.location.href = "login.html"; // Redirect to login page after success
        });
    }

    /* ✅ Login Form Validation */
    const loginForm = document.getElementById("login-form");
    if (loginForm) {
        loginForm.addEventListener("submit", function (event) {
            event.preventDefault();

            let email = document.getElementById("login-email").value.trim();
            let password = document.getElementById("login-password").value.trim();

            if (email === "" || password === "") {
                alert("All fields are required.");
                return;
            }

            alert("Login Successful! Redirecting...");
            window.location.href = "upload.html"; // Redirect after successful login
        });
    }

    /* ✅ MRI Upload & Model Prediction */
    const uploadForm = document.getElementById("upload-form");
    if (uploadForm) {
        uploadForm.addEventListener("submit", function (event) {
            event.preventDefault();

            let fileInput = document.getElementById("file-input").files[0];
            if (!fileInput) {
                alert("Please select an MRI image to upload.");
                return;
            }

            let formData = new FormData();
            formData.append("file", fileInput);

            // Send the file to the Flask backend
            fetch("http://127.0.0.1:5000/predict", {  // ✅ Changed to Flask URL
                method: "POST",
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    alert(`Error: ${data.error}`);
                    return;
                }

                document.getElementById("prediction-results").classList.remove("hidden");
                document.getElementById("predicted-class").innerText = `Tumor Type: ${data.predicted_class}`;
                document.getElementById("confidence-score").innerText = `Confidence: ${data.confidence_score}%`;

                // Display Grad-CAM heatmap if available
                if (data.gradcam_image) {
                    let heatmapImg = document.getElementById("gradcam-image");
                    heatmapImg.src = data.gradcam_image;
                    heatmapImg.classList.remove("hidden");
                }
            })
            .catch(error => {
                console.error("Error:", error);
                alert("Server error. Please try again.");
            });
        });
    }

});
