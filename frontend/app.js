/**
 * DeepGuard AI - Frontend Application
 * Handles video upload, API communication, and result display
 */

// Configuration
const API_BASE_URL = 'https://deepfake-detection-1-6jc0.onrender.com/api';
const MAX_FILE_SIZE = 20 * 1024 * 1024; // 20MB
const POLL_INTERVAL = 1000; // 1 second
const ALLOWED_TYPES = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'];

// DOM Elements
const uploadZone = document.getElementById('uploadZone');
const videoInput = document.getElementById('videoInput');
const fileSelected = document.getElementById('fileSelected');
const videoPreview = document.getElementById('videoPreview');
const fileName = document.getElementById('fileName');
const fileSize = document.getElementById('fileSize');
const changeFile = document.getElementById('changeFile');
const analyzeBtn = document.getElementById('analyzeBtn');
const processing = document.getElementById('processing');
const progressPercent = document.getElementById('progressPercent');
const processingTitle = document.getElementById('processingTitle');
const processingStatus = document.getElementById('processingStatus');
const results = document.getElementById('results');
const errorState = document.getElementById('errorState');
const errorMessage = document.getElementById('errorMessage');
const retryBtn = document.getElementById('retryBtn');
const newAnalysis = document.getElementById('newAnalysis');

// State
let selectedFile = null;
let currentTaskId = null;
let pollTimer = null;

// ============================================
// File Upload Handling
// ============================================

// Drag and drop events
uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('dragover');
});

uploadZone.addEventListener('dragleave', () => {
    uploadZone.classList.remove('dragover');
});

uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileSelect(files[0]);
    }
});

// Click to upload
uploadZone.addEventListener('click', () => {
    videoInput.click();
});

videoInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileSelect(e.target.files[0]);
    }
});

// Change file button
changeFile.addEventListener('click', () => {
    resetToUpload();
});

// Analyze button
analyzeBtn.addEventListener('click', () => {
    if (selectedFile) {
        startAnalysis();
    }
});

// Retry button
retryBtn.addEventListener('click', () => {
    if (selectedFile) {
        startAnalysis();
    } else {
        resetToUpload();
    }
});

// New analysis button
newAnalysis.addEventListener('click', () => {
    resetToUpload();
});

/**
 * Handle file selection
 */
function handleFileSelect(file) {
    // Validate file type
    const ext = '.' + file.name.split('.').pop().toLowerCase();
    if (!ALLOWED_TYPES.includes(ext)) {
        showError(`Invalid file type. Please select: ${ALLOWED_TYPES.join(', ')}`);
        return;
    }

    // Validate file size
    if (file.size > MAX_FILE_SIZE) {
        showError(`File too large. Maximum size: ${MAX_FILE_SIZE / (1024 * 1024)}MB`);
        return;
    }

    selectedFile = file;

    // Update UI
    fileName.textContent = file.name;
    fileSize.textContent = formatFileSize(file.size);

    // Show video preview
    const url = URL.createObjectURL(file);
    videoPreview.src = url;

    // Show file selected state
    uploadZone.classList.add('hidden');
    fileSelected.classList.remove('hidden');
}

/**
 * Reset to upload state
 */
function resetToUpload() {
    selectedFile = null;
    currentTaskId = null;

    if (pollTimer) {
        clearInterval(pollTimer);
        pollTimer = null;
    }

    videoInput.value = '';
    videoPreview.src = '';

    // Reset UI
    uploadZone.classList.remove('hidden');
    fileSelected.classList.add('hidden');
    processing.classList.add('hidden');
    results.classList.add('hidden');
    errorState.classList.add('hidden');

    // Reset processing steps
    document.querySelectorAll('.step').forEach(step => {
        step.classList.remove('active', 'completed');
    });
}

// ============================================
// API Communication
// ============================================

/**
 * Start video analysis
 */
async function startAnalysis() {
    // Show processing state
    fileSelected.classList.add('hidden');
    processing.classList.remove('hidden');
    results.classList.add('hidden');
    errorState.classList.add('hidden');

    // Reset progress
    updateProgress(0, 'Uploading video...');

    try {
        // Create form data
        const formData = new FormData();
        formData.append('file', selectedFile);

        // Upload file
        const response = await fetch(`${API_BASE_URL}/analyze`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Upload failed');
        }

        const data = await response.json();
        currentTaskId = data.task_id;

        // Start polling for status
        updateProgress(10, 'Processing started...');
        startPolling();

    } catch (error) {
        console.error('Upload error:', error);
        showError(error.message || 'Failed to upload video. Please try again.');
    }
}

/**
 * Start polling for task status
 */
function startPolling() {
    pollTimer = setInterval(async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/status/${currentTaskId}`);

            if (!response.ok) {
                throw new Error('Failed to get status');
            }

            const status = await response.json();

            // Update UI based on status
            updateProcessingUI(status);

            // Check if complete
            if (status.status === 'completed') {
                clearInterval(pollTimer);
                pollTimer = null;
                showResults(status.result);
            } else if (status.status === 'failed') {
                clearInterval(pollTimer);
                pollTimer = null;
                showError(status.message || 'Analysis failed');
            }

        } catch (error) {
            console.error('Polling error:', error);
            // Don't stop polling on transient errors
        }
    }, POLL_INTERVAL);
}

/**
 * Update processing UI based on status
 */
function updateProcessingUI(status) {
    const progress = Math.round(status.progress * 100);
    updateProgress(progress, status.message);

    // Update steps based on progress
    const step1 = document.getElementById('step1');
    const step2 = document.getElementById('step2');
    const step3 = document.getElementById('step3');
    const step4 = document.getElementById('step4');

    if (progress >= 10) {
        step1.classList.add('active');
    }
    if (progress >= 30) {
        step1.classList.remove('active');
        step1.classList.add('completed');
        step2.classList.add('active');
    }
    if (progress >= 50) {
        step2.classList.remove('active');
        step2.classList.add('completed');
        step3.classList.add('active');
    }
    if (progress >= 70) {
        step3.classList.remove('active');
        step3.classList.add('completed');
        step4.classList.add('active');
    }
    if (progress >= 90) {
        step4.classList.remove('active');
        step4.classList.add('completed');
    }
}

/**
 * Update progress display
 */
function updateProgress(percent, message) {
    progressPercent.textContent = `${percent}%`;
    processingStatus.textContent = message;

    // Update progress circle (optional - simplified animation)
    const circle = document.querySelector('.progress-circle');
    if (circle) {
        const circumference = 251.2;
        const offset = circumference - (percent / 100) * circumference;
        circle.style.strokeDashoffset = offset;
    }
}

// ============================================
// Results Display
// ============================================

/**
 * Show analysis results
 */
function showResults(result) {
    processing.classList.add('hidden');
    results.classList.remove('hidden');

    // Get classification info
    const classification = result.classification;
    const score = result.final_score;
    const label = result.classification_label;

    // Update result badge
    const resultBadge = document.getElementById('resultBadge');
    const resultIcon = document.getElementById('resultIcon');
    const resultLabel = document.getElementById('resultLabel');

    resultBadge.className = 'result-badge';

    if (classification === 'real') {
        resultBadge.classList.add('real');
        resultIcon.textContent = '✅';
        resultLabel.textContent = 'Real';
    } else if (classification === 'low_confidence_deepfake') {
        resultBadge.classList.add('low');
        resultIcon.textContent = '⚠️';
        resultLabel.textContent = 'Low Confidence Deepfake';
    } else if (classification === 'most_probably_deepfake') {
        resultBadge.classList.add('probable');
        resultIcon.textContent = '🔶';
        resultLabel.textContent = 'Most Probably Deepfake';
    } else {
        resultBadge.classList.add('definite');
        resultIcon.textContent = '🔴';
        resultLabel.textContent = 'Definitely a Deepfake';
    }

    // Animate score ring
    const scoreProgress = document.getElementById('scoreProgress');
    const scoreNumber = document.getElementById('scoreNumber');
    const scoreDescription = document.getElementById('scoreDescription');

    // Update score gradient color based on classification
    const scoreGradient = document.getElementById('scoreGradient');
    const stops = scoreGradient.querySelectorAll('stop');

    if (classification === 'real') {
        stops[0].setAttribute('stop-color', '#22c55e');
        stops[1].setAttribute('stop-color', '#16a34a');
    } else if (classification === 'low_confidence_deepfake') {
        stops[0].setAttribute('stop-color', '#eab308');
        stops[1].setAttribute('stop-color', '#ca8a04');
    } else if (classification === 'most_probably_deepfake') {
        stops[0].setAttribute('stop-color', '#f97316');
        stops[1].setAttribute('stop-color', '#ea580c');
    } else {
        stops[0].setAttribute('stop-color', '#ef4444');
        stops[1].setAttribute('stop-color', '#dc2626');
    }

    // Animate score
    animateValue(scoreNumber, 0, score, 1500);

    const circumference = 339.3;
    const offset = circumference - (score / 100) * circumference;
    setTimeout(() => {
        scoreProgress.style.strokeDashoffset = offset;
    }, 100);

    // Set description based on label
    scoreDescription.textContent = label.split(' - ')[1] || label;

    // Update component scores
    const cnnScore = result.component_scores.cnn;
    const freqScore = result.component_scores.frequency;
    const tempScore = result.component_scores.temporal;

    document.getElementById('cnnValue').textContent = `${cnnScore}%`;
    document.getElementById('freqValue').textContent = `${freqScore}%`;
    document.getElementById('tempValue').textContent = `${tempScore}%`;

    setTimeout(() => {
        document.getElementById('cnnBar').style.width = `${cnnScore}%`;
        document.getElementById('freqBar').style.width = `${freqScore}%`;
        document.getElementById('tempBar').style.width = `${tempScore}%`;
    }, 300);

    // Update meta info
    document.getElementById('processingTime').textContent = `${result.processing_time}s`;
    document.getElementById('framesAnalyzed').textContent = result.frames_analyzed;
    document.getElementById('confidenceValue').textContent = `${result.confidence}%`;
}

/**
 * Show error state
 */
function showError(message) {
    processing.classList.add('hidden');
    fileSelected.classList.add('hidden');
    results.classList.add('hidden');
    errorState.classList.remove('hidden');

    errorMessage.textContent = message;
}

// ============================================
// Utility Functions
// ============================================

/**
 * Format file size
 */
function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

/**
 * Animate a numeric value
 */
function animateValue(element, start, end, duration) {
    const startTime = performance.now();

    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);

        // Ease out cubic
        const eased = 1 - Math.pow(1 - progress, 3);
        const current = Math.round(start + (end - start) * eased);

        element.textContent = current;

        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }

    requestAnimationFrame(update);
}

// ============================================
// FAQ Accordion
// ============================================

document.querySelectorAll('.faq-question').forEach(button => {
    button.addEventListener('click', () => {
        const item = button.parentElement;
        const isActive = item.classList.contains('active');

        // Close all
        document.querySelectorAll('.faq-item').forEach(i => {
            i.classList.remove('active');
        });

        // Open clicked if it was closed
        if (!isActive) {
            item.classList.add('active');
        }
    });
});

// ============================================
// Smooth Scroll
// ============================================

document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// ============================================
// Initialize
// ============================================

console.log('🎓 IDT Deepfake Detection initialized');
