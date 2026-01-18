let currentTab = 'object';

// Tab switching function
function switchTab(tab) {
    currentTab = tab;
    
    // Update tab buttons
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    event.target.classList.add('active');
    
    // Update tab content
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    document.getElementById(tab + '-tab').classList.add('active');
    
    updateStatus();
}

// Update status from server
async function updateStatus() {
    try {
        const response = await fetch('/status');
        const data = await response.json();
        
        // Update Object Detection Tab
        const objectStatus = document.querySelector('#object-status .status-value');
        const objectBtn = document.getElementById('btn-object');
        const detectBtn = document.getElementById('btn-detect');
        
        if (data.object_camera_active) {
            objectStatus.textContent = 'Camera Active - ' + (data.detection_enabled ? 'Detecting' : 'Paused');
            objectBtn.innerHTML = '<span>⏸️</span> Stop Camera';
            objectBtn.classList.remove('btn-primary');
            objectBtn.classList.add('btn-danger');
        } else {
            objectStatus.textContent = 'Camera Inactive';
            objectBtn.innerHTML = '<span>📹</span> Start Camera';
            objectBtn.classList.remove('btn-danger');
            objectBtn.classList.add('btn-primary');
        }

        detectBtn.innerHTML = data.detection_enabled ? 
            '<span>⏸️</span> Pause Detection' : 
            '<span>▶️</span> Resume Detection';

        // Update detected objects
        const objectsGrid = document.getElementById('detected-objects');
        if (data.detected_objects && data.detected_objects.length > 0) {
            objectsGrid.innerHTML = data.detected_objects
                .map(obj => `<div class="detection-item">${obj}</div>`)
                .join('');
        } else {
            objectsGrid.innerHTML = '';
        }

        // Update Text Detection Tab
        const textStatus = document.querySelector('#text-status .status-value');
        const textBtn = document.getElementById('btn-text');
        
        if (data.text_camera_active) {
            textStatus.textContent = 'Camera Active - Reading Text';
            textBtn.innerHTML = '<span>⏸️</span> Stop Camera';
            textBtn.classList.remove('btn-primary');
            textBtn.classList.add('btn-danger');
        } else {
            textStatus.textContent = 'Camera Inactive';
            textBtn.innerHTML = '<span>📹</span> Start Camera';
            textBtn.classList.remove('btn-danger');
            textBtn.classList.add('btn-primary');
        }

    } catch (error) {
        console.error('Status update error:', error);
    }
}

// Toggle camera
async function toggleCamera(type) {
    await fetch(`/toggle_camera/${type}`, { method: 'POST' });
    updateStatus();
}

// Toggle detection
async function toggleDetection() {
    await fetch('/toggle_detection', { method: 'POST' });
    updateStatus();
}

// Text Input Tab Functions
function updateCharacterCount() {
    const textarea = document.getElementById('user-text-input');
    const charCount = document.getElementById('char-count');
    if (textarea && charCount) {
        charCount.textContent = textarea.value.length;
    }
}

async function speakInputText() {
    const textarea = document.getElementById('user-text-input');
    const speakBtn = document.getElementById('btn-speak-input');
    const statusText = document.getElementById('input-status-text');
    
    const text = textarea.value.trim();
    
    if (!text) {
        statusText.textContent = 'Please enter some text first';
        statusText.style.color = '#f56565';
        setTimeout(() => {
            statusText.textContent = 'Ready to speak';
            statusText.style.color = 'white';
        }, 2000);
        return;
    }
    
    // Disable button and show loading state
    speakBtn.disabled = true;
    speakBtn.innerHTML = '<span>⏳</span> Speaking...';
    statusText.textContent = 'Speaking your text...';
    statusText.style.color = '#48bb78';
    
    try {
        const response = await fetch('/speak_input_text', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: text })
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            statusText.textContent = 'Speaking complete!';
            statusText.style.color = '#48bb78';
        } else {
            statusText.textContent = 'Error: ' + data.message;
            statusText.style.color = '#f56565';
        }
    } catch (error) {
        console.error('Error speaking text:', error);
        statusText.textContent = 'Error communicating with server';
        statusText.style.color = '#f56565';
    } finally {
        // Re-enable button
        setTimeout(() => {
            speakBtn.disabled = false;
            speakBtn.innerHTML = '<span>🔊</span> Speak Text';
            statusText.textContent = 'Ready to speak';
            statusText.style.color = 'white';
        }, 2000);
    }
}

function clearInputText() {
    const textarea = document.getElementById('user-text-input');
    const statusText = document.getElementById('input-status-text');
    
    textarea.value = '';
    updateCharacterCount();
    
    statusText.textContent = 'Text cleared';
    statusText.style.color = '#4299e1';
    
    setTimeout(() => {
        statusText.textContent = 'Ready to speak';
        statusText.style.color = 'white';
    }, 1500);
}

function loadSampleText(text) {
    const textarea = document.getElementById('user-text-input');
    const statusText = document.getElementById('input-status-text');
    
    textarea.value = text;
    updateCharacterCount();
    
    statusText.textContent = 'Sample text loaded';
    statusText.style.color = '#667eea';
    
    setTimeout(() => {
        statusText.textContent = 'Ready to speak';
        statusText.style.color = 'white';
    }, 1500);
}

// Event listener for textarea input
document.addEventListener('DOMContentLoaded', function() {
    const textarea = document.getElementById('user-text-input');
    if (textarea) {
        textarea.addEventListener('input', updateCharacterCount);
        
        // Allow Enter key with Ctrl to trigger speak
        textarea.addEventListener('keydown', function(e) {
            if (e.ctrlKey && e.key === 'Enter') {
                speakInputText();
            }
        });
    }
    
    // Initialize status update
    updateStatus();
});

// Regular status updates
setInterval(updateStatus, 1000);