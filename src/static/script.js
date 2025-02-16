function previewImage() {
    let input = document.getElementById("imageUpload");
    let file = input.files[0];

    if (file) {
        // Display file name
        document.getElementById("fileName").innerText = `File: ${file.name}`;

        // Display image preview
        let reader = new FileReader();
        reader.onload = function (e) {
            let imagePreview = document.getElementById("imagePreview");
            imagePreview.src = e.target.result;
            document.getElementById("imagePreviewContainer").style.display = "block";
        };
        reader.readAsDataURL(file);
    } else {
        document.getElementById("fileName").innerText = "No file chosen";
        document.getElementById("imagePreviewContainer").style.display = "none";
    }
}

function uploadImage() {
    let input = document.getElementById("imageUpload");
    let file = input.files[0];

    if (!file) {
        alert("Please upload an image.");
        return;
    }

    let formData = new FormData();
    formData.append("file", file);

    fetch("/predict", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("predictionResult").innerText = `Predicted Digit: ${data.prediction}`;
    })
    .catch(error => console.error("Error:", error));
}
