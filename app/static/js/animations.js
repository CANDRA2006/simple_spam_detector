// Create floating particles
function createParticles() {
    const particles = document.getElementById('particles');
    const particleCount = 30;
    
    for (let i = 0; i < particleCount; i++) {
        const particle = document.createElement('div');
        particle.className = 'particle';
        
        const size = Math.random() * 5 + 2;
        particle.style.width = `${size}px`;
        particle.style.height = `${size}px`;
        particle.style.left = `${Math.random() * 100}%`;
        particle.style.top = `${Math.random() * 100}%`;
        particle.style.animationDelay = `${Math.random() * 15}s`;
        particle.style.animationDuration = `${Math.random() * 10 + 10}s`;
        
        particles.appendChild(particle);
    }
}

// Character counter
const messageInput = document.getElementById('messageInput');
const charCount = document.getElementById('charCount');

messageInput.addEventListener('input', () => {
    const count = messageInput.value.length;
    charCount.textContent = `${count} character${count !== 1 ? 's' : ''}`;
});

// Example buttons
const exampleButtons = document.querySelectorAll('.example-btn');
exampleButtons.forEach(btn => {
    btn.addEventListener('click', () => {
        messageInput.value = btn.dataset.text;
        messageInput.dispatchEvent(new Event('input'));
        messageInput.focus();
    });
});

// Clear button
document.getElementById('clearBtn').addEventListener('click', () => {
    messageInput.value = '';
    messageInput.dispatchEvent(new Event('input'));
    document.getElementById('resultSection').classList.add('hidden');
});

// Check button
document.getElementById('checkBtn').addEventListener('click', async () => {
    const text = messageInput.value.trim();
    
    if (!text) {
        alert('Please enter a message to check');
        return;
    }
    
    // Show loading
    document.getElementById('resultSection').classList.add('hidden');
    document.getElementById('loadingSection').classList.remove('hidden');
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text })
        });
        
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        // Hide loading
        document.getElementById('loadingSection').classList.add('hidden');
        
        // Display results
        displayResults(data);
        
    } catch (error) {
        document.getElementById('loadingSection').classList.add('hidden');
        alert(`Error: ${error.message}`);
    }
});

function displayResults(data) {
    const resultSection = document.getElementById('resultSection');
    const resultIcon = document.getElementById('resultIcon');
    const resultTitle = document.getElementById('resultTitle');
    const resultDescription = document.getElementById('resultDescription');
    const confidenceValue = document.getElementById('confidenceValue');
    const progressFill = document.getElementById('progressFill');
    const hamProb = document.getElementById('hamProb');
    const spamProb = document.getElementById('spamProb');
    const hamBar = document.getElementById('hamBar');
    const spamBar = document.getElementById('spamBar');
    const warningBox = document.getElementById('warningBox');
    
    // Set icon and title based on prediction
    if (data.is_spam) {
        resultIcon.className = 'result-icon spam';
        resultIcon.textContent = '⚠️';
        resultTitle.textContent = 'SPAM DETECTED';
        resultTitle.style.color = '#ef4444';
        resultDescription.textContent = 'This message has been identified as spam. Be cautious and avoid clicking any links.';
    } else {
        resultIcon.className = 'result-icon ham';
        resultIcon.textContent = '✅';
        resultTitle.textContent = 'SAFE MESSAGE';
        resultTitle.style.color = '#10b981';
        resultDescription.textContent = 'This message appears to be legitimate and safe.';
    }
    
    // Set confidence
    confidenceValue.textContent = `${data.confidence}%`;
    progressFill.style.width = `${data.confidence}%`;
    
    // Set probabilities
    hamProb.textContent = `${data.ham_probability}%`;
    spamProb.textContent = `${data.spam_probability}%`;
    hamBar.style.width = `${data.ham_probability}%`;
    spamBar.style.width = `${data.spam_probability}%`;
    
    // Show warning signs if spam
    if (data.is_spam && data.spam_probability > 70) {
        const warnings = [
            'Urgency or pressure tactics detected',
            'Suspicious links or requests for personal information',
            'Unusual sender or grammatical errors',
            'Promises of prizes, money, or unrealistic offers'
        ];
        
        const warningList = document.getElementById('warningList');
        warningList.innerHTML = '';
        
        warnings.forEach(warning => {
            const li = document.createElement('li');
            li.textContent = warning;
            warningList.appendChild(li);
        });
        
        warningBox.classList.remove('hidden');
    } else {
        warningBox.classList.add('hidden');
    }
    
    // Show result section
    resultSection.classList.remove('hidden');
    
    // Smooth scroll to results
    resultSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Initialize particles on load
window.addEventListener('load', () => {
    createParticles();
});

// Enter key to check
messageInput.addEventListener('keydown', (e) => {
    if (e.ctrlKey && e.key === 'Enter') {
        document.getElementById('checkBtn').click();
    }
});