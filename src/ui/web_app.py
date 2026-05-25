import io
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse

from src.dictation.transcriber import Transcriber

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

transcriber: Optional[Transcriber] = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load transcriber on startup, cleanup on shutdown."""
    global transcriber
    logger.info("Loading transcriber model...")
    transcriber = Transcriber(model_size="base")
    logger.info("Transcriber ready")
    yield
    logger.info("Shutting down")


app = FastAPI(title="Radio Dictate Web", lifespan=lifespan)

HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Radio Dictate - Web Workstation</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
</head>
<body class="bg-gray-50 text-gray-800 h-screen flex flex-col font-sans">
    
    <!-- Navbar -->
    <header class="bg-white shadow-sm border-b px-6 py-4 flex items-center justify-between">
        <h1 class="text-xl font-semibold text-blue-800"><i class="fas fa-stethoscope mr-2"></i>Radio Dictate Web</h1>
        <div class="flex space-x-4">
            <button id="clearBtn" class="text-gray-500 hover:text-red-600 transition px-3 py-1 rounded hover:bg-red-50" title="Clear all text (cannot be undone)"><i class="fas fa-trash mr-1"></i> Clear</button>
            <button id="copyBtn" class="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-md transition font-medium" title="Copy text to clipboard"><i id="copyIcon" class="fas fa-copy mr-1"></i> <span id="copyText">Copy</span></button>
        </div>
    </header>

    <!-- Main Content -->
    <main class="flex-grow flex flex-col max-w-4xl w-full mx-auto p-6 md:p-8">
        
        <!-- Instructions -->
        <div class="mb-4 p-4 bg-blue-50 border border-blue-200 rounded-lg">
            <p class="text-sm text-blue-900"><i class="fas fa-info-circle mr-2"></i><strong>How to use:</strong> Click the microphone button to start recording. Your words will appear below.</p>
        </div>

        <!-- Editor Area -->
        <div class="flex-grow flex flex-col bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden mb-6 relative">
            <div id="statusIndicator" class="absolute top-4 right-4 text-sm font-medium text-gray-700 flex items-center bg-white px-3 py-2 rounded-md border border-gray-300 shadow-sm opacity-0 transition-opacity duration-300" aria-live="polite" aria-label="Status indicator">
                <span class="relative flex h-3 w-3 mr-2">
                  <span id="ping1" class="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75 hidden"></span>
                  <span id="ping2" class="relative inline-flex rounded-full h-3 w-3 bg-red-500 hidden"></span>
                </span>
                <span id="statusText">Processing...</span>
            </div>
            
            <label for="editor" class="sr-only">Dictation content</label>
            <textarea id="editor" class="w-full h-full p-6 text-lg md:text-xl resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 text-gray-800 placeholder-gray-400" placeholder="Your dictation will appear here..."></textarea>
        </div>

        <!-- Controls -->
        <div class="flex justify-center items-center h-24">
            <button id="dictateBtn" class="bg-blue-600 hover:bg-blue-700 text-white rounded-full h-20 w-20 flex items-center justify-center shadow-lg transition-transform transform hover:scale-105 active:scale-95 focus:outline-none focus:ring-4 focus:ring-blue-300" aria-label="Start or stop recording">
                <i id="micIcon" class="fas fa-microphone text-3xl" aria-hidden="true"></i>
            </button>
        </div>
        <p class="text-center text-sm text-gray-500 mt-2"><kbd>Space</kbd> to record • <kbd>Ctrl+Z</kbd> to undo</p>

    </main>

    <script>
        let mediaRecorder;
        let audioChunks = [];
        let isRecording = false;
        let undoStack = [''];
        let undoIndex = 0;

        const dictateBtn = document.getElementById('dictateBtn');
        const micIcon = document.getElementById('micIcon');
        const editor = document.getElementById('editor');
        const clearBtn = document.getElementById('clearBtn');
        const copyBtn = document.getElementById('copyBtn');
        const copyText = document.getElementById('copyText');
        const copyIcon = document.getElementById('copyIcon');
        const statusIndicator = document.getElementById('statusIndicator');
        const statusText = document.getElementById('statusText');
        const ping1 = document.getElementById('ping1');
        const ping2 = document.getElementById('ping2');

        // Track text changes for undo
        editor.addEventListener('input', () => {
            if (undoIndex < undoStack.length - 1) undoStack.length = undoIndex + 1;
            undoStack.push(editor.value);
            undoIndex++;
            if (undoStack.length > 50) undoStack.shift();
        });

        // Clear with confirmation
        clearBtn.addEventListener('click', () => {
            if (editor.value.trim() === '') return;
            if (confirm('Are you sure you want to clear all text? This cannot be undone.')) {
                editor.value = '';
                undoStack = [''];
                undoIndex = 0;
            }
        });

        // Copy with visual feedback
        copyBtn.addEventListener('click', () => {
            if (editor.value.trim() === '') {
                copyText.textContent = 'Nothing to copy';
                setTimeout(() => { copyText.textContent = 'Copy'; }, 2000);
                return;
            }
            navigator.clipboard.writeText(editor.value)
                .then(() => {
                    copyText.textContent = '✓ Copied!';
                    copyIcon.className = 'fas fa-check mr-1';
                    setTimeout(() => {
                        copyText.textContent = 'Copy';
                        copyIcon.className = 'fas fa-copy mr-1';
                    }, 2000);
                })
                .catch(err => {
                    console.error('Failed to copy', err);
                    copyText.textContent = 'Copy failed';
                    setTimeout(() => { copyText.textContent = 'Copy'; }, 2000);
                });
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey || e.metaKey) {
                if (e.key === 'z' && !isRecording) { e.preventDefault(); undo(); }
                if (e.key === 'y' && !isRecording) { e.preventDefault(); redo(); }
            }
            if (e.code === 'Space' && e.target === document.body && !isRecording) {
                e.preventDefault();
                dictateBtn.click();
            }
        });

        function undo() {
            if (undoIndex > 0) {
                undoIndex--;
                editor.value = undoStack[undoIndex];
            }
        }

        function redo() {
            if (undoIndex < undoStack.length - 1) {
                undoIndex++;
                editor.value = undoStack[undoIndex];
            }
        }

        async function startRecording() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);

                mediaRecorder.ondataavailable = event => {
                    if (event.data.size > 0) audioChunks.push(event.data);
                };

                mediaRecorder.onstop = sendAudio;

                audioChunks = [];
                mediaRecorder.start();
                isRecording = true;
                dictateBtn.setAttribute('aria-pressed', 'true');

                // Update UI: Recording State
                dictateBtn.classList.replace('bg-blue-600', 'bg-red-500');
                dictateBtn.classList.replace('hover:bg-blue-700', 'hover:bg-red-600');
                micIcon.classList.replace('fa-microphone', 'fa-stop');

                statusIndicator.style.opacity = '1';
                statusText.innerText = '🔴 Recording...';
                ping1.classList.remove('hidden');
                ping2.classList.remove('hidden');

            } catch (err) {
                console.error("Microphone access denied:", err);
                showError('Microphone access denied. Please allow microphone permissions in your browser settings and try again.');
            }
        }

        function stopRecording() {
            if (mediaRecorder && mediaRecorder.state !== 'inactive') {
                mediaRecorder.stop();
                mediaRecorder.stream.getTracks().forEach(track => track.stop());
                isRecording = false;
                dictateBtn.setAttribute('aria-pressed', 'false');

                // Update UI: Processing State
                dictateBtn.classList.replace('bg-red-500', 'bg-blue-600');
                dictateBtn.classList.replace('hover:bg-red-600', 'hover:bg-blue-700');
                micIcon.classList.replace('fa-stop', 'fa-microphone');

                statusText.innerText = '⏳ Transcribing...';
                ping1.classList.add('hidden');
                ping2.classList.remove('hidden');
                ping2.classList.replace('bg-red-500', 'bg-blue-500');
            }
        }

        function showError(message) {
            statusIndicator.style.opacity = '1';
            statusText.innerHTML = '❌ ' + message;
            statusText.style.color = '#dc2626';
            setTimeout(() => {
                statusIndicator.style.opacity = '0';
                statusText.style.color = '';
            }, 5000);
        }

        dictateBtn.setAttribute('aria-pressed', 'false');
        dictateBtn.setAttribute('role', 'button');
        dictateBtn.setAttribute('aria-label', 'Start or stop recording');

        dictateBtn.addEventListener('click', () => {
            if (isRecording) {
                stopRecording();
            } else {
                startRecording();
            }
        });

        async function sendAudio() {
            const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
            const formData = new FormData();
            formData.append("file", audioBlob, "dictation.webm");

            try {
                const response = await fetch('/transcribe', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.detail || 'Transcription failed');
                }

                const result = await response.json();

                if (result.text) {
                    if (editor.value.trim() !== '') {
                        editor.value += ' ' + result.text;
                    } else {
                        editor.value = result.text;
                    }
                    editor.scrollTop = editor.scrollHeight;
                    undoStack.push(editor.value);
                    undoIndex++;
                    statusText.innerText = '✓ Transcription complete';
                    statusText.style.color = '#16a34a';
                    setTimeout(() => { statusIndicator.style.opacity = '0'; }, 2000);
                }
            } catch (err) {
                console.error("Transcription error:", err);
                showError('Transcription failed: ' + err.message + '. Try again.');
            } finally {
                ping2.classList.replace('bg-blue-500', 'bg-red-500');
            }
        }
    </script>
</body>
</html>
"""

@app.get("/")
async def root():
    return HTMLResponse(HTML_CONTENT)

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    # Read uploaded bytes with size limit (50MB max)
    max_size = 50 * 1024 * 1024  # 50MB
    audio_bytes = await file.read(max_size + 1)

    if len(audio_bytes) > max_size:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {max_size / 1024 / 1024:.0f}MB"
        )

    file_like = io.BytesIO(audio_bytes)
    file_like.name = "audio.webm"

    try:
        logger.info("Transcribing audio...")
        text, _ = transcriber.transcribe(
            file_like,
            beam_size=1,
            condition_on_previous_text=False
        )
        return {"text": text}
    except Exception as e:
        logger.error("Transcription failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to transcribe audio")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8005)
