document.addEventListener('DOMContentLoaded', function() {
    // Chat functionality
    const chatButton = document.getElementById('chatButton');
    const openChatBtn = document.getElementById('openChatBtn');
    const chatOverlay = document.getElementById('chatOverlay');
    const closeChatBtn = document.getElementById('closeChatBtn');
    const chatForm = document.getElementById('chatForm');
    const chatInput = document.getElementById('chatInput');
    const chatMessages = document.getElementById('chatMessages');

    // Toggle chat overlay
    function toggleChat() {
        chatOverlay.classList.toggle('active');
        
        if (chatOverlay.classList.contains('active')) {
            chatInput.focus();
        }
    }

    // Event listeners for chat
    if (chatButton) chatButton.addEventListener('click', toggleChat);
    if (openChatBtn) openChatBtn.addEventListener('click', toggleChat);
    if (closeChatBtn) closeChatBtn.addEventListener('click', toggleChat);

    // Handle chat form submission
    if (chatForm) {
        chatForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const question = chatInput.value.trim();
            
            if (question) {
                // Add user message to chat
                addMessage(question, 'user');
                
                // Clear input
                chatInput.value = '';
                
                // Show loading indicator
                const loadingId = showLoading();
                
                // Send message to backend
                fetch('/api/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ question: question }),
                })
                .then(response => response.json())
                .then(data => {
                    // Remove loading indicator
                    removeLoading(loadingId);
                    
                    // Add bot response to chat
                    addMessage(data.response, 'bot');
                })
                .catch(error => {
                    console.error('Error:', error);
                    removeLoading(loadingId);
                    addMessage('Sorry, I encountered an error. Please try again.', 'bot');
                });
            }
        });
    }

    // Helper function to add message to chat
    function addMessage(text, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', sender);
        
        const messagePara = document.createElement('p');
        messagePara.textContent = text;
        
        messageDiv.appendChild(messagePara);
        chatMessages.appendChild(messageDiv);
        
        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Helper function to show loading indicator
    function showLoading() {
        const loadingId = Date.now();
        const loadingDiv = document.createElement('div');
        loadingDiv.classList.add('message', 'bot', 'loading');
        loadingDiv.id = `loading-${loadingId}`;
        
        const loadingPara = document.createElement('p');
        loadingPara.textContent = 'Thinking...';
        
        loadingDiv.appendChild(loadingPara);
        chatMessages.appendChild(loadingDiv);
        
        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
        
        return loadingId;
    }

    // Helper function to remove loading indicator
    function removeLoading(loadingId) {
        const loadingDiv = document.getElementById(`loading-${loadingId}`);
        if (loadingDiv) {
            loadingDiv.remove();
        }
    }

    // Check if we're on the classify page
    const uploadContainer = document.getElementById('uploadContainer');
    const resultContainer = document.getElementById('resultContainer');
    const fileInput = document.getElementById('fileInput');
    const dropArea = document.getElementById('dropArea');
    const previewImage = document.getElementById('previewImage');
    const wasteType = document.getElementById('wasteType');
    const recyclability = document.getElementById('recyclability');
    const classifyAgainBtn = document.getElementById('classifyAgainBtn');
    
    // Preview container (we'll add this to the HTML)
    let previewContainer = document.getElementById('previewContainer');
    let previewImageElement = document.getElementById('previewImageBeforeUpload');
    let uploadBtn = document.getElementById('uploadBtn');
    let cancelBtn = document.getElementById('cancelBtn');
    
    if (uploadContainer && dropArea) {
        // File upload functionality
        dropArea.addEventListener('click', function() {
            fileInput.click();
        });
        
        // Drag and drop events
        dropArea.addEventListener('dragover', function(e) {
            e.preventDefault();
            dropArea.classList.add('drag-over');
        });
        
        dropArea.addEventListener('dragleave', function() {
            dropArea.classList.remove('drag-over');
        });
        
        dropArea.addEventListener('drop', function(e) {
            e.preventDefault();
            dropArea.classList.remove('drag-over');
            
            if (e.dataTransfer.files.length) {
                handleFile(e.dataTransfer.files[0]);
            }
        });
        
        // File input change
        fileInput.addEventListener('change', function() {
            if (fileInput.files.length) {
                handleFile(fileInput.files[0]);
            }
        });
        
        // Upload button click
        if (uploadBtn) {
            uploadBtn.addEventListener('click', function() {
                if (fileInput.files.length) {
                    uploadImage(fileInput.files[0]);
                }
            });
        }
        
        // Cancel button click
        if (cancelBtn) {
            cancelBtn.addEventListener('click', function() {
                resetUpload();
            });
        }
        
        // Classify again button
        if (classifyAgainBtn) {
            classifyAgainBtn.addEventListener('click', function() {
                // Reset form
                resetUpload();
                uploadContainer.classList.remove('hidden');
                resultContainer.classList.add('hidden');
                
                // Update step indicators
                updateSteps(1);
            });
        }
        
        // Handle file selection and preview
        function handleFile(file) {
            // Check if file is an image
            if (!file.type.startsWith('image/')) {
                alert('Please upload an image file');
                return;
            }
            
            // Show preview
            const reader = new FileReader();
            reader.onload = function(e) {
                // Hide drop area
                dropArea.classList.add('hidden');
                
                // Show preview container
                previewContainer.classList.remove('hidden');
                
                // Set preview image source
                previewImageElement.src = e.target.result;
                
                // Update step indicators
                updateSteps(2);
            };
            reader.readAsDataURL(file);
        }
        
        // Upload image to server
        function uploadImage(file) {
            // Create form data
            const formData = new FormData();
            formData.append('image', file);
            
            // Show loading state
            uploadBtn.disabled = true;
            uploadBtn.textContent = 'Classifying...';
            
            // Send to server
            fetch('/api/classify', {
                method: 'POST',
                body: formData,
            })
            .then(response => response.json())
            .then(data => {
                // Update result container
                previewImage.src = data.image_url;
                wasteType.textContent = data.waste_type;
                recyclability.textContent = data.recyclability;
                
                // Show result container
                uploadContainer.classList.add('hidden');
                resultContainer.classList.remove('hidden');
                
                // Update step indicators
                updateSteps(3);
                
                // Reset upload button
                uploadBtn.disabled = false;
                uploadBtn.textContent = 'Classify';
            })
            .catch(error => {
                console.error('Error:', error);
                alert('Error classifying image. Please try again.');
                
                // Reset upload button
                uploadBtn.disabled = false;
                uploadBtn.textContent = 'Classify';
            });
        }
        
        // Reset upload state
        function resetUpload() {
            fileInput.value = '';
            if (previewContainer) {
                previewContainer.classList.add('hidden');
            }
            if (dropArea) {
                dropArea.classList.remove('hidden');
            }
        }
        
        // Update step indicators
        function updateSteps(currentStep) {
            const steps = document.querySelectorAll('.step');
            
            steps.forEach((step, index) => {
                if (index + 1 < currentStep) {
                    step.classList.add('active');
                } else if (index + 1 === currentStep) {
                    step.classList.add('active');
                } else {
                    step.classList.remove('active');
                }
            });
        }
    }
});