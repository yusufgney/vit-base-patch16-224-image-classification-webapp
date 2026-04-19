const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const resultSection = document.getElementById('result-section');
const uploadSection = document.querySelector('.upload-section');
const previewImage = document.getElementById('preview-image');
const predictionsList = document.getElementById('predictions-list');
const loader = document.getElementById('loader');
const resetBtn = document.getElementById('reset-btn');

// Trigger file input
dropZone.addEventListener('click', () => fileInput.click());

// Handle file selection
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) {
        handleFile(e.target.files[0]);
    }
});

// Drag and drop events
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('drag-over');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    if (e.dataTransfer.files.length) {
        handleFile(e.dataTransfer.files[0]);
    }
});

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please upload an image file.');
        return;
    }

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        uploadImage(file);
    };
    reader.readAsDataURL(file);
}

async function uploadImage(file) {
    // Show loader, hide upload
    uploadSection.classList.add('hidden');
    loader.classList.remove('hidden');
    resultSection.classList.add('hidden');

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error('Prediction failed');

        const data = await response.json();
        displayResults(data.predictions);
    } catch (error) {
        console.error(error);
        alert('Error: ' + error.message);
        resetUI();
    } finally {
        loader.classList.add('hidden');
    }
}

function displayResults(predictions) {
    predictionsList.innerHTML = '';
    
    if (predictions.length === 0) {
        predictionsList.innerHTML = '<p class="subtitle">Model is not confident enough.</p>';
    } else {
        predictions.forEach(pred => {
            const pct = (pred.confidence * 100).toFixed(2);
            const item = document.createElement('div');
            item.className = 'prediction-item';
            item.innerHTML = `
                <div class="label-row">
                    <span class="label-name">${pred.label}</span>
                    <span class="label-pct">${pct}%</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: 0%"></div>
                </div>
            `;
            predictionsList.appendChild(item);
            
            // Trigger animation
            setTimeout(() => {
                item.querySelector('.progress-fill').style.width = pct + '%';
            }, 100);
        });
    }

    resultSection.classList.remove('hidden');
}

resetBtn.addEventListener('click', resetUI);

function resetUI() {
    uploadSection.classList.remove('hidden');
    resultSection.classList.add('hidden');
    loader.classList.add('hidden');
    fileInput.value = '';
}
