// script.js
const video = document.getElementById('webcam');

navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        video.srcObject = stream;
    })
    .catch(err => {
        console.error('Error accessing the camera: ', err);
    });

function takePicture(snapshotIndex) {
    const canvas = document.getElementById(`snapshot${snapshotIndex}`);
    const context = canvas.getContext('2d');
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
}

function printLabel() {
    alert('Print Label button clicked');
}

function printPackageIDLabel() {
    alert('Print package ID label button clicked');
}

document.getElementById('info-form').addEventListener('submit', (event) => {
    event.preventDefault();
    alert('Form submitted');
});
