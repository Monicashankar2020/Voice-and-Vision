let stream = null;
let currentMode = 'upload';

const uploadArea = document.getElementById('uploadArea');
const cameraArea = document.getElementById('cameraArea');
const fileInput = document.getElementById('fileInput');
const video = document.getElementById('video');
const canvas = document.getElementById('captureCanvas');
const loading = document.getElementById('loading');
const error = document.getElementById('error');
const previewArea = document.getElementById('previewArea');

// Mode switching
function switchMode(mode) {
    currentMode = mode;
    document.querySelectorAll('.mode-btn').forEach(btn => btn.classList.remove('active'));
    event.target.classList.add('active');
    
    if (mode === 'upload') {
        uploadArea.style.display = 'block';
        cameraArea.style.display = 'none';
        stopCamera();
    } else {
        uploadArea.style.display = 'none';
        cameraArea.style.display = 'block';
        startCamera();
    }
    resetUpload();
}

// Start camera
async function startCamera() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
                width: 640, 
                height: 480,
                facingMode: 'user'
            } 
        });
        video.srcObject = stream;
    } catch (err) {
        showError('Could not access camera: ' + err.message);
        switchMode('upload');
    }
}

// Stop camera
function stopCamera() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
        video.srcObject = null;
    }
}

// Capture image from camera with countdown
let countdownInterval = null;

function captureImage() {
    const captureBtn = document.querySelector('.camera-btn.capture');
    const originalText = captureBtn.innerHTML;
    let countdown = 3;
    
    // Disable button during countdown
    captureBtn.disabled = true;
    captureBtn.style.opacity = '0.7';
    
    // Update button text with countdown
    captureBtn.innerHTML = `📸 Capturing in ${countdown}...`;
    
    countdownInterval = setInterval(() => {
        countdown--;
        if (countdown > 0) {
            captureBtn.innerHTML = `📸 Capturing in ${countdown}...`;
        } else {
            clearInterval(countdownInterval);
            captureBtn.innerHTML = '📸 Processing...';
            
            // Capture the image
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const ctx = canvas.getContext('2d');
            
            // Flip horizontally for mirror effect
            ctx.translate(canvas.width, 0);
            ctx.scale(-1, 1);
            ctx.drawImage(video, 0, 0);
            
            // Convert to blob and send
            canvas.toBlob(blob => {
                stopCamera();
                handleFile(blob, true);
                
                // Reset button
                captureBtn.innerHTML = originalText;
                captureBtn.disabled = false;
                captureBtn.style.opacity = '1';
            }, 'image/jpeg', 0.95);
        }
    }, 1000);
}

// Drag and drop for upload
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
        handleFile(e.dataTransfer.files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});

// Handle file upload or camera capture
function handleFile(file, fromCamera = false) {
    if (!fromCamera && !file.type.startsWith('image/')) { 
        showError('Please upload an image file'); 
        return; 
    }
    
    const formData = new FormData();
    formData.append('image', file);
    
    uploadArea.style.display = 'none';
    cameraArea.style.display = 'none';
    loading.style.display = 'block';
    error.style.display = 'none';
    previewArea.style.display = 'none';
    
    fetch('/detect_sign', { 
        method: 'POST', 
        body: formData 
    })
    .then(r => r.json())
    .then(data => {
        loading.style.display = 'none';
        if (data.error) {
            showError(data.error);
        } else {
            showResults(data);
        }
    })
    .catch(err => {
        loading.style.display = 'none';
        showError('Error processing image: ' + err.message);
    });
}

// Show results
function showResults(data) {
    document.getElementById('originalImage').src = 'data:image/jpeg;base64,' + data.original_image;
    document.getElementById('detectedImage').src = 'data:image/jpeg;base64,' + data.detected_image;
    document.getElementById('prediction').textContent = data.prediction;
    document.getElementById('confidence').textContent = data.confidence.toFixed(1) + '% Confidence';
    
    let topHtml = '<h3>Top 3 Predictions</h3>';
    data.top_predictions.forEach((pred, idx) => {
        topHtml += `<div class="prediction-item">
                <span>${idx + 1}. ${pred.label}</span>
                <span>${pred.confidence.toFixed(1)}%</span>
            </div>`;
    });
    document.getElementById('topPredictions').innerHTML = topHtml;
    
    previewArea.style.display = 'block';
}

// Show error
function showError(message) {
    error.textContent = message;
    error.style.display = 'block';
    if (currentMode === 'upload') {
        uploadArea.style.display = 'block';
    } else {
        cameraArea.style.display = 'block';
        startCamera();
    }
}

// Reset upload
function resetUpload() {
    if (currentMode === 'upload') {
        uploadArea.style.display = 'block';
    } else {
        cameraArea.style.display = 'block';
        startCamera();
    }
    previewArea.style.display = 'none';
    error.style.display = 'none';
    fileInput.value = '';
}

// Cleanup on page unload
window.addEventListener('beforeunload', stopCamera);
