const startCaptureButton = document.getElementById('startCapture');
const stopCaptureButton = document.getElementById('stopCapture');
const captureImageButton = document.getElementById('captureImage');
const cameraFeed = document.getElementById('cameraFeed');
const canvas = document.getElementById('canvas');
const context = canvas.getContext('2d');
const gallery = document.getElementById('gallery');
const capturedImage = document.getElementById('capturedImage');


let stream;
let captureInterval;
let images = [];

// Function to convert dataURL to Blob
function dataURLToBlob(dataURL) {
    const [header, data] = dataURL.split(',');
    const mime = header.split(':')[1].split(';')[0];
    const byteString = atob(data);
    const u8arr = new Uint8Array(byteString.length);

    for (let i = 0; i < byteString.length; i++) {
        u8arr[i] = byteString.charCodeAt(i);
    }

    return new Blob([u8arr], { type: mime });
}

// Start the camera
async function startCamera() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        cameraFeed.srcObject = stream;
    } catch (err) {
        console.error('Error accessing webcam:', err);
    }
}
function captureImage() {
    canvas.width = cameraFeed.videoWidth;
    canvas.height = cameraFeed.videoHeight;
    context.drawImage(cameraFeed, 0, 0, canvas.width, canvas.height);
    const imageData = canvas.toDataURL('image/jpeg');



    const img = document.createElement('img');
    img.src = imageData;
    img.title = `Captured on ${new Date().toLocaleString()}`;
    img.addEventListener('click', () => {
        capturedImage.src = imageData;
    });

    gallery.appendChild(img);

    // Save image data for future upload
    images.push(imageData);
    // addImageToList(imageData);


    // Add image to gallery
    const imgElement = document.createElement('img');
    imgElement.src = imageData;
    // imageList.appendChild(imageData);
}

// Function to capture and upload image
function captureAndUploadImage() {
    canvas.width = cameraFeed.videoWidth;
    canvas.height = cameraFeed.videoHeight;
    context.drawImage(cameraFeed, 0, 0, canvas.width, canvas.height);
    const imageData = canvas.toDataURL('image/jpeg');

    const img = document.createElement('img');
    img.src = imageData;
    img.title = `Captured on ${new Date().toLocaleString()}`;
    img.addEventListener('click', () => {
        capturedImage.src = imageData;
    });


    // Save image data for future upload
    images.push(imageData);

    // Convert the image data to Blob
    const blob = dataURLToBlob(imageData);
    const formData = new FormData();
    formData.append('file', blob, 'captured_image.jpg');

    // Upload captured image to the server
    fetch(' https://192.168.0.119:8080/upload', {
        method: 'POST',
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            console.log('Upload successful:', data);

            // Get references to the input fields
            const customerField = document.getElementById('customer');
            const AsField = document.getElementById('As');
            const trackingField = document.getElementById('Tracking');
            const carrierField = document.getElementById('Carrier');

            // Update the customer and As fields with the response data only if they are empty
            if (!customerField.value) {
                customerField.value = data.name;
            }
            if (!AsField.value) {
                AsField.value = data.As;
            }
            if (!trackingField.value) {
                trackingField.value = data.tracking_number;
            }
            if (!carrierField.value) {
                carrierField.value = data.carrier;
            }
        })
        .catch(error => {
            console.error('Error uploading image:', error);
        });
}

// Start capturing images every second
function startCapturing() {
    captureInterval = setInterval(captureAndUploadImage, 1000); // Capture and upload every second
}

// Stop capturing images
function stopCapturing() {
    clearInterval(captureInterval);
}

// Start the camera when the page loads
window.onload = startCamera;

// Add event listeners for buttons
startCaptureButton.addEventListener('click', () => {
    startCaptureButton.disabled = true;
    stopCaptureButton.disabled = false;
    startCapturing();
});

stopCaptureButton.addEventListener('click', () => {
    stopCaptureButton.disabled = true;
    startCaptureButton.disabled = false;
    stopCapturing();
});

captureImageButton.addEventListener('click', function (event) {
    captureImage(event);
});