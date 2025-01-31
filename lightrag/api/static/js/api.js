// State management
const state = {
    apiKey: localStorage.getItem('apiKey') || '',
    files: [],
    indexedFiles: [],
    currentPage: 'file-manager'
};

// Utility functions
const showToast = (message, duration = 3000) => {
    const toast = document.getElementById('toast');
    toast.querySelector('div').textContent = message;
    toast.classList.remove('hidden');
    setTimeout(() => toast.classList.add('hidden'), duration);
};

const fetchWithAuth = async (url, options = {}) => {
    console.log(`Calling server with api key : ${state.apiKey}`)
    const headers = {
        ...(options.headers || {}),
        ...(state.apiKey ? { 'X-API-Key': state.apiKey } : {}) // Use X-API-Key instead of Bearer
    };
    return fetch(url, { ...options, headers });
};


// Page renderers
const pages = {
    'file-manager': () => `
        <div class="space-y-6">
            <h2 class="text-2xl font-bold text-gray-800">File Manager</h2>

            <div class="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-gray-400 transition-colors">
                <input type="file" id="fileInput" multiple accept=".txt,.md,.doc,.docx,.pdf,.pptx" class="hidden">
                <label for="fileInput" class="cursor-pointer">
                    <svg class="mx-auto h-12 w-12 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"/>
                    </svg>
                    <p class="mt-2 text-gray-600">Drag files here or click to select</p>
                    <p class="text-sm text-gray-500">Supported formats: TXT, MD, DOC, PDF, PPTX</p>
                </label>
            </div>

            <div id="fileList" class="space-y-2">
                <h3 class="text-lg font-semibold text-gray-700">Selected Files</h3>
                <div class="space-y-2"></div>
            </div>
            <div id="uploadProgress" class="hidden mt-4">
                <div class="w-full bg-gray-200 rounded-full h-2.5">
                    <div class="bg-blue-600 h-2.5 rounded-full" style="width: 0%"></div>
                </div>
                <p class="text-sm text-gray-600 mt-2"><span id="uploadStatus">0</span> files processed</p>
            </div>
            <div class="flex items-center space-x-4 bg-gray-100 p-4 rounded-lg shadow-md">
                <button id="rescanBtn" class="flex items-center bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20" fill="currentColor" class="mr-2">
                        <path d="M12 4a8 8 0 1 1-8 8H2.5a9.5 9.5 0 1 0 2.8-6.7L2 3v6h6L5.7 6.7A7.96 7.96 0 0 1 12 4z"/>
                    </svg>
                    Rescan Files
                </button>
            
                <button id="uploadBtn" class="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors">
                    Upload & Index Files
                </button>
            </div>

            <div id="indexedFiles" class="space-y-2">
                <h3 class="text-lg font-semibold text-gray-700">Indexed Files</h3>
                <div class="space-y-2"></div>
            </div>


        </div>
    `,

    'query': () => `
        <div class="space-y-6">
            <h2 class="text-2xl font-bold text-gray-800">Query Database</h2>

            <div class="space-y-4">
                <div>
                    <label class="block text-sm font-medium text-gray-700">Query Mode</label>
                    <select id="queryMode" class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500">
                        <option value="hybrid">Hybrid</option>
                        <option value="local">Local</option>
                        <option value="global">Global</option>
                        <option value="naive">Naive</option>
                    </select>
                </div>

                <div>
                    <label class="block text-sm font-medium text-gray-700">Query</label>
                    <textarea id="queryInput" rows="4" class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"></textarea>
                </div>

                <button id="queryBtn" class="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors">
                    Send Query
                </button>

                <div id="queryResult" class="mt-4 p-4 bg-white rounded-lg shadow"></div>
            </div>
        </div>
    `,

    'knowledge-graph': () => `
        <div class="flex items-center justify-center h-full">
            <div class="text-center">
                <svg class="mx-auto h-12 w-12 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"/>
                </svg>
                <h3 class="mt-2 text-sm font-medium text-gray-900">Under Construction</h3>
                <p class="mt-1 text-sm text-gray-500">Knowledge graph visualization will be available in a future update.</p>
            </div>
        </div>
    `,

    'status': () => `
        <div class="space-y-6">
            <h2 class="text-2xl font-bold text-gray-800">System Status</h2>
            <div id="statusContent" class="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div class="p-6 bg-white rounded-lg shadow-sm">
                    <h3 class="text-lg font-semibold mb-4">System Health</h3>
                    <div id="healthStatus"></div>
                </div>
                <div class="p-6 bg-white rounded-lg shadow-sm">
                    <h3 class="text-lg font-semibold mb-4">Configuration</h3>
                    <div id="configStatus"></div>
                </div>
            </div>
        </div>
    `,

    'settings': () => `
        <div class="space-y-6">
            <h2 class="text-2xl font-bold text-gray-800">Settings</h2>

            <div class="max-w-xl">
                <div class="space-y-4">
                    <div>
                        <label class="block text-sm font-medium text-gray-700">API Key</label>
                        <input type="password" id="apiKeyInput" value="${state.apiKey}"
                            class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500">
                    </div>

                    <button id="saveSettings" class="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors">
                        Save Settings
                    </button>
                </div>
            </div>
        </div>
    `
};

// Page handlers
const handlers = {
    'file-manager': () => {
        const fileInput = document.getElementById('fileInput');
        const dropZone = fileInput.parentElement.parentElement;
        const fileList = document.querySelector('#fileList div');
        const indexedFiles = document.querySelector('#indexedFiles div');
        const uploadBtn = document.getElementById('uploadBtn');

        const updateFileList = () => {
            fileList.innerHTML = state.files.map(file => `
                <div class="flex items-center justify-between bg-white p-3 rounded-lg shadow-sm">
                    <span>${file.name}</span>
                    <button class="text-red-600 hover:text-red-700" onclick="removeFile('${file.name}')">
                        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"/>
                        </svg>
                    </button>
                </div>
            `).join('');
        };

        const updateIndexedFiles = async () => {
            const response = await fetchWithAuth('/health');
            const data = await response.json();
            indexedFiles.innerHTML = data.indexed_files.map(file => `
                <div class="flex items-center justify-between bg-white p-3 rounded-lg shadow-sm">
                    <span>${file}</span>
                </div>
            `).join('');
        };

        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('border-blue-500');
        });

        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('border-blue-500');
        });

        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('border-blue-500');
            const files = Array.from(e.dataTransfer.files);
            state.files.push(...files);
            updateFileList();
        });

        fileInput.addEventListener('change', () => {
            state.files.push(...Array.from(fileInput.files));
            updateFileList();
        });

        uploadBtn.addEventListener('click', async () => {
            if (state.files.length === 0) {
                showToast('Please select files to upload');
                return;
            }
            let apiKey = localStorage.getItem('apiKey') || '';
            const progress = document.getElementById('uploadProgress');
            const progressBar = progress.querySelector('div');
            const statusText = document.getElementById('uploadStatus');
            progress.classList.remove('hidden');

            for (let i = 0; i < state.files.length; i++) {
                const formData = new FormData();
                formData.append('file', state.files[i]);

                try {
                    await fetch('/documents/upload', {
                        method: 'POST',
                        headers: apiKey ? { 'Authorization': `Bearer ${apiKey}` } : {},
                        body: formData
                    });

                    const percentage = ((i + 1) / state.files.length) * 100;
                    progressBar.style.width = `${percentage}%`;
                    statusText.textContent = `${i + 1}/${state.files.length}`;
                } catch (error) {
                    console.error('Upload error:', error);
                }
            }
            progress.classList.add('hidden');
        });

        rescanBtn.addEventListener('click', async () => {
            const progress = document.getElementById('uploadProgress');
            const progressBar = progress.querySelector('div');
            const statusText = document.getElementById('uploadStatus');
            progress.classList.remove('hidden');

            try {
                // Start the scanning process
                const scanResponse = await fetch('/documents/scan', {
                    method: 'POST',
                });

                if (!scanResponse.ok) {
                    throw new Error('Scan failed to start');
                }

                // Start polling for progress
                const pollInterval = setInterval(async () => {
                    const progressResponse = await fetch('/documents/scan-progress');
                    const progressData = await progressResponse.json();

                    // Update progress bar
                    progressBar.style.width = `${progressData.progress}%`;

                    // Update status text
                    if (progressData.total_files > 0) {
                        statusText.textContent = `Processing ${progressData.current_file} (${progressData.indexed_count}/${progressData.total_files})`;
                    }

                    // Check if scanning is complete
                    if (!progressData.is_scanning) {
                        clearInterval(pollInterval);
                        progress.classList.add('hidden');
                        statusText.textContent = 'Scan complete!';
                    }
                }, 1000); // Poll every second

            } catch (error) {
                console.error('Upload error:', error);
                progress.classList.add('hidden');
                statusText.textContent = 'Error during scanning process';
            }
        });


        updateIndexedFiles();
    },

    'query': () => {
        const queryBtn = document.getElementById('queryBtn');
        const queryInput = document.getElementById('queryInput');
        const queryMode = document.getElementById('queryMode');
        const queryResult = document.getElementById('queryResult');

        let apiKey = localStorage.getItem('apiKey') || '';

        queryBtn.addEventListener('click', async () => {
            const query = queryInput.value.trim();
            if (!query) {
                showToast('Please enter a query');
                return;
            }

            queryBtn.disabled = true;
            queryBtn.innerHTML = `
                <svg class="animate-spin h-5 w-5 mr-3" viewBox="0 0 24 24">
                    <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" fill="none"/>
                    <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"/>
                </svg>
                Processing...
            `;

            try {
                const response = await fetchWithAuth('/query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        query,
                        mode: queryMode.value,
                        stream: false,
                        only_need_context: false
                    })
                });

                const data = await response.json();
                queryResult.innerHTML = marked.parse(data.response);
            } catch (error) {
                showToast('Error processing query');
            } finally {
                queryBtn.disabled = false;
                queryBtn.textContent = 'Send Query';
            }
        });
    },

    'status': async () => {
        const healthStatus = document.getElementById('healthStatus');
        const configStatus = document.getElementById('configStatus');

        try {
            const response = await fetchWithAuth('/health');
            const data = await response.json();

            healthStatus.innerHTML = `
                <div class="space-y-2">
                    <div class="flex items-center">
                        <div class="w-3 h-3 rounded-full ${data.status === 'healthy' ? 'bg-green-500' : 'bg-red-500'} mr-2"></div>
                        <span class="font-medium">${data.status}</span>
                    </div>
                    <div>
                        <p class="text-sm text-gray-600">Working Directory: ${data.working_directory}</p>
                        <p class="text-sm text-gray-600">Input Directory: ${data.input_directory}</p>
                        <p class="text-sm text-gray-600">Indexed Files: ${data.indexed_files_count}</p>
                    </div>
                </div>
            `;

            configStatus.innerHTML = Object.entries(data.configuration)
                .map(([key, value]) => `
                    <div class="mb-2">
                        <span class="text-sm font-medium text-gray-700">${key}:</span>
                        <span class="text-sm text-gray-600 ml-2">${value}</span>
                    </div>
                `).join('');
        } catch (error) {
            showToast('Error fetching status');
        }
    },

    'settings': () => {
        const saveBtn = document.getElementById('saveSettings');
        const apiKeyInput = document.getElementById('apiKeyInput');

        saveBtn.addEventListener('click', () => {
            state.apiKey = apiKeyInput.value;
            localStorage.setItem('apiKey', state.apiKey);
            showToast('Settings saved successfully');
        });
    }
};

// Navigation handling
document.querySelectorAll('.nav-item').forEach(item => {
    item.addEventListener('click', (e) => {
        e.preventDefault();
        const page = item.dataset.page;
        document.getElementById('content').innerHTML = pages[page]();
        if (handlers[page]) handlers[page]();
        state.currentPage = page;
    });
});

// Initialize with file manager
document.getElementById('content').innerHTML = pages['file-manager']();
handlers['file-manager']();

// Global functions
window.removeFile = (fileName) => {
    state.files = state.files.filter(file => file.name !== fileName);
    document.querySelector('#fileList div').innerHTML = state.files.map(file => `
        <div class="flex items-center justify-between bg-white p-3 rounded-lg shadow-sm">
            <span>${file.name}</span>
            <button class="text-red-600 hover:text-red-700" onclick="removeFile('${file.name}')">
                <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"/>
                </svg>
            </button>
        </div>
    `).join('');
};
