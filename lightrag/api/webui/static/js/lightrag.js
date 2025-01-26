// lightrag.js
// Modify according to the actual API address
const API_BASE = 'http://localhost:9621';

// init
function initializeApp() {
    setupFileUpload();
    setupQueryHandler();
    setupSectionObserver();
    updateFileList();
    // textarea count
    const textArea = document.getElementById('textContent');
    if (textArea) {
        const charCount = document.createElement('div');
        charCount.className = 'char-count';
        textArea.parentNode.appendChild(charCount);

        textArea.addEventListener('input', () => {
            const count = textArea.value.length;
            charCount.textContent = `input ${count} character`;
            charCount.style.color = count > 10000 ? '#ef4444' : 'var (--text-secondary)'
        });
    }
}

// api request
async function apiRequest(endpoint, method = 'GET', body = null) {
    const options = {
        method,
        headers: {
            'Content-Type': 'application/json'
        }
    };

    if (body) {
        options.body = JSON.stringify(body);
    }

    try {
        const response = await fetch(`${API_BASE}${endpoint}`, options);
        if (!response.ok) {
            throw new Error(`request failed: ${response.status}`);
        }
        return response.json();
    } catch (error) {
        console.error('API REQUEST ERROR:', error);
        showToast(error.message, 'error');
        throw error;
    }
}

async function handleTextUpload() {
    const description = document.getElementById('textDescription').value;
    const content = document.getElementById('textContent').value.trim();
    const statusDiv = document.getElementById('textUploadStatus');

    //  clear status tip
    statusDiv.className = 'status-indicator';
    statusDiv.textContent = '';

    //  input valid
    if (!content) {
        showStatus('error', 'TEXT CONTENT NOT NULL', statusDiv);
        return;
    }

    try {
        showStatus('loading', 'UPLOADING...', statusDiv);

        const payload = {
            text: content,
            ...(description && {description})
        };

        const response = await fetch(`${API_BASE}/documents/text`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'UPLOAD FAILED');
        }

        const result = await response.json();

        showStatus('success', `✅ ${result.message} (documents: ${result.document_count})`, statusDiv);


        document.getElementById('textContent').value = '';

        //  update file list
        updateFileList();

    } catch (error) {
        showStatus('error', `❌ ERROR: ${error.message}`, statusDiv);
        console.error('FILE UPLOAD FAILED:', error);
    }
}

function showStatus(type, message, container) {
    container.textContent = message;
    container.className = `status-indicator ${type}`;

    //auto clear success status
    if (type === 'success') {
        setTimeout(() => {
            container.textContent = '';
            container.className = 'status-indicator';
        }, 5000);
    }
}

// upload file
function setupFileUpload() {
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('fileInput');


    // Drag and drop event handling
    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('active');
    });

    dropzone.addEventListener('dragleave', () => {
        dropzone.classList.remove('active');
    });

    dropzone.addEventListener('drop', async (e) => {
        e.preventDefault();
        dropzone.classList.remove('active');
        await handleFiles(e.dataTransfer.files);
    });


    fileInput.addEventListener('change', async (e) => {
        await handleFiles(e.target.files);
    });
}

async function handleFiles(files) {
    const formData = new FormData();
    for (const file of files) {
        formData.append('file', file);
    }
    const statusDiv = document.getElementById('fileUploadStatus');


    statusDiv.className = 'status-indicator';
    statusDiv.textContent = '';
    try {
        showStatus('loading', 'UPLOADING...', statusDiv);
        const response = await fetch(`${API_BASE}/documents/upload`, {
            method: 'POST',
            body: formData
        });
        const result = await response.json();
        showStatus('success', `✅ ${result.message} `, statusDiv);
        updateFileList();
    } catch (error) {
        showToast(error.message, 'error');
    }
}


async function updateFileList() {
    const fileList = document.querySelector('.file-list');
    try {
        const status = await apiRequest('/health');
        fileList.innerHTML = `
            <div>INDEXED FILE: ${status.indexed_files_count}</div>
        `;
    } catch (error) {
        fileList.innerHTML = 'UNABLE TO OBTAIN FILE LIST';
    }
}

// Intelligent retrieval processing
function setupQueryHandler() {
    document.querySelector('#query .btn-primary').addEventListener('click', handleQuery);
}

async function handleQuery() {
    const queryInput = document.querySelector('#query textarea');
    const modeSelect = document.querySelector('#query select');
    const streamCheckbox = document.querySelector('#query input[type="checkbox"]');
    const resultsDiv = document.querySelector('#query .results');

    const payload = {
        query: queryInput.value,
        mode: modeSelect.value,
        stream: streamCheckbox.checked
    };

    resultsDiv.innerHTML = '<div class="loading">SEARCHING...</div>';

    try {
        if (payload.stream) {
            await handleStreamingQuery(payload, resultsDiv);
        } else {
            const result = await apiRequest('/query', 'POST', payload);
            resultsDiv.innerHTML = `<div class="result">${result.response}</div>`;
        }
    } catch (error) {
        resultsDiv.innerHTML = `<div class="error">SEARCH FAILED: ${error.message}</div>`;
    }
}

// handle stream api
async function handleStreamingQuery(payload, resultsDiv) {
    try {
        const response = await fetch(`${API_BASE}/query/stream`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(payload)
        });


        const contentType = response.headers.get('Content-Type') || '';
        const validTypes = ['application/x-ndjson', 'application/json'];
        if (!validTypes.some(t => contentType.includes(t))) {
            throw new Error(`INVALID CONTENT TYPE: ${contentType}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let buffer = '';

        resultsDiv.innerHTML = '';

        while (true) {
            const {done, value} = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, {stream: true});

            //  Split by line break (NDJSON format requirement)
            let lineEndIndex;
            while ((lineEndIndex = buffer.indexOf('\n')) >= 0) {
                const line = buffer.slice(0, lineEndIndex).trim();
                buffer = buffer.slice(lineEndIndex + 1);

                if (!line) continue;

                try {
                    const data = JSON.parse(line);

                    if (data.response) {
                        resultsDiv.innerHTML += data.response;
                        resultsDiv.scrollTop = resultsDiv.scrollHeight;
                    }

                    if (data.error) {
                        resultsDiv.innerHTML += `<div class="error">${data.error}</div>`;
                    }
                } catch (error) {
                    console.error('JSON PARSING FAILED:', {
                        error,
                        rawLine: line,
                        bufferRemaining: buffer
                    });
                }
            }
        }

        // Process remaining data
        if (buffer.trim()) {
            try {
                const data = JSON.parse(buffer.trim());
                if (data.response) {
                    resultsDiv.innerHTML += data.response;
                }
            } catch (error) {
                console.error('TAIL DATA PARSING FAILED:', error);
            }
        }

    } catch (error) {
        resultsDiv.innerHTML = `<div class="error">REQUEST FAILED: ${error.message}</div>`;
    }
}


// Knowledge Q&A Processing
function setupChatHandler() {
    const sendButton = document.querySelector('#chat button');
    const chatInput = document.querySelector('#chat input');

    sendButton.addEventListener('click', () => handleChat(chatInput));
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') handleChat(chatInput);
    });
}

async function handleChat(chatInput) {
    const chatHistory = document.querySelector('#chat .chat-history');


    const userMessage = document.createElement('div');
    userMessage.className = 'message user';
    userMessage.textContent = chatInput.value;
    chatHistory.appendChild(userMessage);


    const botMessage = document.createElement('div');
    botMessage.className = 'message bot loading';
    botMessage.textContent = 'THINKING...';
    chatHistory.appendChild(botMessage);
    chatHistory.scrollTop = chatHistory.scrollHeight;

    try {
        const response = await fetch(`${API_BASE}/api/chat`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                messages: [{role: "user", content: chatInput.value}],
                stream: true
            })
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        botMessage.classList.remove('loading');
        botMessage.textContent = '';

        while (true) {
            const {done, value} = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value);
            const data = JSON.parse(chunk);
            botMessage.textContent += data.message?.content || '';
            chatHistory.scrollTop = chatHistory.scrollHeight;
        }
    } catch (error) {
        botMessage.textContent = `ERROR: ${error.message}`;
        botMessage.classList.add('error');
    }

    chatInput.value = '';
}

// system status update
async function updateSystemStatus() {
    const statusElements = {
        health: document.getElementById('healthStatus'),
        storageProgress: document.getElementById('storageProgress'),
        indexedFiles: document.getElementById('indexedFiles'),
        storageUsage: document.getElementById('storageUsage'),
        llmModel: document.getElementById('llmModel'),
        embedModel: document.getElementById('embedModel'),
        maxTokens: document.getElementById('maxTokens'),
        workingDir: document.getElementById('workingDir'),
        inputDir: document.getElementById('inputDir'),
        kv_storage: document.getElementById("kv_storage"),
        doc_status_storage: document.getElementById("doc_status_storage"),
        graph_storage: document.getElementById("graph_storage"),
        vector_storage: document.getElementById("vector_storage")
    };

    try {
        const status = await apiRequest('/health');

        //  heath status
        statusElements.health.className = 'status-badge';
        statusElements.health.textContent = status.status === 'healthy' ?
            '✅ Healthy operation in progress' : '⚠️ Service exception';

        //  kv status
        const progressValue = Math.min(Math.round((status.indexed_files_count / 1000) * 100), 100);
        statusElements.storageProgress.value = progressValue;
        statusElements.indexedFiles.textContent = `INDEXED FILES：${status.indexed_files_count}`;
        statusElements.storageUsage.textContent = `USE PERCENT：${progressValue}%`;

        //  model state
        statusElements.llmModel.textContent = `${status.configuration.llm_model} (${status.configuration.llm_binding})`;
        statusElements.embedModel.textContent = `${status.configuration.embedding_model} (${status.configuration.embedding_binding})`;
        statusElements.maxTokens.textContent = status.configuration.max_tokens.toLocaleString();

        //  dir msg
        statusElements.workingDir.textContent = status.working_directory;
        statusElements.inputDir.textContent = status.input_directory;

        // stack msg
        statusElements.kv_storage.textContent = status.configuration.kv_storage;
        statusElements.doc_status_storage.textContent = status.configuration.doc_status_storage;
        statusElements.graph_storage.textContent = status.configuration.graph_storage;
        statusElements.vector_storage.textContent = status.configuration.vector_storage

    } catch (error) {
        statusElements.health.className = 'status-badge error';
        statusElements.health.textContent = '❌GET STATUS FAILED';
        statusElements.storageProgress.value = 0;
        statusElements.indexedFiles.textContent = 'INDEXED FILES：GET FAILED';
        console.error('STATUS UPDATE FAILED:', error);
    }
}


//  Area switching monitoring
function setupSectionObserver() {
    const observer = new MutationObserver(mutations => {
        mutations.forEach(mutation => {
            if (mutation.attributeName === 'style') {
                const isVisible = mutation.target.style.display !== 'none';
                if (isVisible && mutation.target.id === 'status') {
                    updateSystemStatus();
                }
            }
        });
    });

    document.querySelectorAll('.card').forEach(section => {
        observer.observe(section, {attributes: true});
    });
}

// Display prompt information
function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    document.body.appendChild(toast);

    setTimeout(() => {
        toast.remove();
    }, 3000);
}


// Dynamically load tag list
async function loadLabels() {
    try {
        const response = await fetch('/graph/label/list');
        const labels = await response.json();
        renderLabels(labels);
    } catch (error) {
        console.error('DYNAMICALLY LOAD TAG LIST FAILED:', error);
    }
}

async function loadGraph(label) {
    try {
        // render label list
        openGraphModal(label)
    } catch (error) {
        console.error('LOADING LABEL FAILED:', error);


        const labelList = document.getElementById("label-list");
        labelList.innerHTML = `
            <div class="error-message">
                LOADING ERROR: ${error.message}
            </div>
        `;
    }
}

// render graph label list
function renderLabels(labels) {
    const container = document.getElementById('label-list');
    container.innerHTML = labels.map(label => `
        <div class="label-item">
            <span style="font-weight: 500; color: var(--text-primary);">
                ${label}
            </span>
            <div class="label-actions">
                <button class="btn btn-primary"
                        onclick="handleLabelAction('${label}')">
                    📋 graph
                </button>
            </div>
        </div>
    `).join('');
}


function handleLabelAction(label) {
    loadGraph(label)
}


function refreshLabels() {
    showToast('LOADING GRAPH LABELS...', 'info');
    loadLabels();
}

function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    document.body.appendChild(toast);

    setTimeout(() => {
        toast.remove();
    }, 3000);
}

document.addEventListener('DOMContentLoaded', initializeApp);
