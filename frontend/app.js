const statusEl = document.getElementById('status');
const confidenceEl = document.getElementById('confidence');
const confidenceValueEl = document.querySelector('.confidence-value');
const originalImageEl = document.getElementById('original-image');
const resultImageEl = document.getElementById('result-image');
const detectionsEl = document.getElementById('detections');
const formEl = document.getElementById('upload-form');
const imageInputEl = document.getElementById('image-input');

const API_URL = window.__BACKEND_URL__ || window.location.origin;

confidenceEl.addEventListener('input', () => {
  confidenceValueEl.textContent = Number(confidenceEl.value).toFixed(2);
});

imageInputEl.addEventListener('change', () => {
  const file = imageInputEl.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = (event) => {
    originalImageEl.src = event.target.result;
  };
  reader.readAsDataURL(file);
});

formEl.addEventListener('submit', async (event) => {
  event.preventDefault();

  const file = imageInputEl.files[0];
  if (!file) {
    statusEl.textContent = 'Please choose an image first.';
    return;
  }

  statusEl.textContent = 'Detecting road damage...';

  const formData = new FormData();
  formData.append('image', file);
  formData.append('confidence', confidenceEl.value);

  try {
    const response = await fetch(`${API_URL}/predict`, {
      method: 'POST',
      body: formData,
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || 'Detection failed.');
    }

    resultImageEl.src = data.image;
    detectionsEl.innerHTML = '';

    if (data.detections && data.detections.length > 0) {
      data.detections.forEach((item) => {
        const li = document.createElement('li');
        li.textContent = item;
        detectionsEl.appendChild(li);
      });
    } else {
      const li = document.createElement('li');
      li.textContent = 'No road damage detected.';
      detectionsEl.appendChild(li);
    }

    statusEl.textContent = `${data.message} Detected ${data.count} object(s).`;
  } catch (error) {
    statusEl.textContent = error.message;
    resultImageEl.src = '';
    detectionsEl.innerHTML = '';
  }
});
