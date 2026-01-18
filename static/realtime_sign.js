let isDetecting = false;
let frameCount = 0;
let detectionHistory = [];
let updateInterval = null;

// Initialize page
document.addEventListener('DOMContentLoaded', function() {
    console.log('Real-time sign language detection page loaded');
    initializeChart();
});

// Start detection
function startDetection() {
    fetch('/start_realtime_sign', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                isDetecting = true;
                updateUI(true);
                startStatusPolling();
                console.log('Detection started');
            } else {
                showError(data.message || 'Failed to start detection');
            }
        })
        .catch(error => {
            console.error('Error starting detection:', error);
            showError('Failed to start detection');
        });
}

// Stop detection
function stopDetection() {
    fetch('/stop_realtime_sign', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                isDetecting = false;
                updateUI(false);
                stopStatusPolling();
                console.log('Detection stopped');
            }
        })
        .catch(error => {
            console.error('Error stopping detection:', error);
        });
}

// Refresh feed
function refreshFeed() {
    const videoFeed = document.getElementById('video-feed');
    const timestamp = new Date().getTime();
    videoFeed.src = `/realtime_sign_feed?t=${timestamp}`;
    frameCount = 0;
    document.getElementById('frame-count').textContent = '0';
}

// Update UI based on detection state
function updateUI(detecting) {
    const startBtn = document.getElementById('start-btn');
    const stopBtn = document.getElementById('stop-btn');
    const statusElement = document.getElementById('status');

    if (detecting) {
        startBtn.disabled = true;
        stopBtn.disabled = false;
        statusElement.textContent = 'Detecting...';
        statusElement.className = 'info-value status-detecting';
    } else {
        startBtn.disabled = false;
        stopBtn.disabled = true;
        statusElement.textContent = 'Ready';
        statusElement.className = 'info-value status-ready';
    }
}

// Start polling for detection status
function startStatusPolling() {
    updateInterval = setInterval(updateDetectionStatus, 500);
}

// Stop polling
function stopStatusPolling() {
    if (updateInterval) {
        clearInterval(updateInterval);
        updateInterval = null;
    }
}

// Update detection status
function updateDetectionStatus() {
    fetch('/realtime_sign_status')
        .then(response => response.json())
        .then(data => {
            if (data.active) {
                updateDetectionInfo(data);
            }
        })
        .catch(error => {
            console.error('Error fetching status:', error);
        });
}

// Update detection information
function updateDetectionInfo(data) {
    // Update current gesture
    const gestureElement = document.getElementById('current-gesture');
    if (data.gesture && data.gesture !== 'None') {
        gestureElement.textContent = data.gesture;
        gestureElement.style.color = '#667eea';
    } else {
        gestureElement.textContent = 'No gesture detected';
        gestureElement.style.color = '#999';
    }

    // Update confidence
    const confidenceElement = document.getElementById('confidence');
    if (data.confidence) {
        const conf = (data.confidence * 100).toFixed(1);
        confidenceElement.textContent = conf + '%';
        
        // Color based on confidence
        if (conf > 80) {
            confidenceElement.style.color = '#22c55e';
        } else if (conf > 60) {
            confidenceElement.style.color = '#f59e0b';
        } else {
            confidenceElement.style.color = '#ef4444';
        }
    }

    // Update frame count
    if (data.frame_count !== undefined) {
        frameCount = data.frame_count;
        document.getElementById('frame-count').textContent = frameCount;
    }

    // Add to history if new detection
    if (data.gesture && data.gesture !== 'None' && data.confidence > 0.7) {
        addToHistory(data.gesture, data.confidence);
    }
}

// Add detection to history
function addToHistory(gesture, confidence) {
    const now = new Date();
    const timeStr = now.toLocaleTimeString();
    
    // Check if this is a duplicate of the last entry
    if (detectionHistory.length > 0) {
        const lastEntry = detectionHistory[0];
        if (lastEntry.gesture === gesture && 
            (now - lastEntry.timestamp) < 2000) {
            return; // Skip duplicate within 2 seconds
        }
    }

    const entry = {
        gesture: gesture,
        confidence: confidence,
        time: timeStr,
        timestamp: now
    };

    detectionHistory.unshift(entry);
    
    // Keep only last 10 entries
    if (detectionHistory.length > 10) {
        detectionHistory.pop();
    }

    updateHistoryDisplay();
    updateChart();
}

// Update history display
function updateHistoryDisplay() {
    const historyContainer = document.getElementById('detection-history');
    
    if (detectionHistory.length === 0) {
        historyContainer.innerHTML = '<p class="no-history">No detections yet. Start detection to see results.</p>';
        return;
    }

    let html = '';
    detectionHistory.forEach(entry => {
        const conf = (entry.confidence * 100).toFixed(1);
        html += `
            <div class="history-item">
                <span class="history-gesture">${entry.gesture}</span>
                <span class="history-confidence">${conf}%</span>
                <span class="history-time">${entry.time}</span>
            </div>
        `;
    });

    historyContainer.innerHTML = html;
}

// Chart functionality
let detectionChart = null;

function initializeChart() {
    const canvas = document.getElementById('detectionChart');
    const ctx = canvas.getContext('2d');
    
    // Simple bar chart for gesture counts
    detectionChart = {
        canvas: canvas,
        ctx: ctx,
        data: {}
    };
}

function updateChart() {
    if (!detectionChart) return;

    const ctx = detectionChart.ctx;
    const canvas = detectionChart.canvas;
    
    // Count gestures
    const gestureCounts = {};
    detectionHistory.forEach(entry => {
        gestureCounts[entry.gesture] = (gestureCounts[entry.gesture] || 0) + 1;
    });

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw chart
    const labels = Object.keys(gestureCounts);
    const values = Object.values(gestureCounts);
    const maxValue = Math.max(...values, 1);
    
    const barWidth = canvas.width / labels.length;
    const barMaxHeight = canvas.height - 40;

    labels.forEach((label, index) => {
        const value = values[index];
        const barHeight = (value / maxValue) * barMaxHeight;
        const x = index * barWidth + 10;
        const y = canvas.height - barHeight - 30;

        // Draw bar
        ctx.fillStyle = '#667eea';
        ctx.fillRect(x, y, barWidth - 20, barHeight);

        // Draw label
        ctx.fillStyle = '#333';
        ctx.font = '10px Inter';
        ctx.textAlign = 'center';
        ctx.fillText(label.substring(0, 8), x + (barWidth - 20) / 2, canvas.height - 10);

        // Draw value
        ctx.fillStyle = '#fff';
        ctx.font = 'bold 12px Inter';
        ctx.fillText(value.toString(), x + (barWidth - 20) / 2, y - 5);
    });
}

// Show error message
function showError(message) {
    const statusElement = document.getElementById('status');
    statusElement.textContent = 'Error';
    statusElement.className = 'info-value status-error';
    
    // Create temporary error notification
    const notification = document.createElement('div');
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: #ef4444;
        color: white;
        padding: 15px 20px;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        z-index: 1000;
        animation: slideIn 0.3s ease-out;
    `;
    notification.textContent = message;
    document.body.appendChild(notification);

    setTimeout(() => {
        notification.remove();
    }, 3000);
}

// Handle page visibility
document.addEventListener('visibilitychange', function() {
    if (document.hidden && isDetecting) {
        // Page is hidden, optionally pause polling
        stopStatusPolling();
    } else if (!document.hidden && isDetecting) {
        // Page is visible again, resume polling
        startStatusPolling();
    }
});

// Cleanup on page unload
window.addEventListener('beforeunload', function() {
    if (isDetecting) {
        stopDetection();
    }
});