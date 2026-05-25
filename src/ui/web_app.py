from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
import io
import logging

from src.core.transcriber import Transcriber

# Set up simple logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Radio Dictate Web")

# Load transcriber on startup
transcriber = Transcriber(model_size="base") # adjust model size as needed

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
            <button id="clearBtn" class="text-gray-500 hover:text-red-500 transition px-3 py-1"><i class="fas fa-trash mr-1"></i> Clear</button>
            <button id="copyBtn" class="bg-gray-100 hover:bg-gray-200 text-gray-700 px-4 py-2 rounded-md transition font-medium"><i class="fas fa-copy mr-1"></i> Copy Report</button>
        </div>
    </header>

    <!-- Main Content -->
    <main class="flex-grow flex flex-col max-w-4xl w-full mx-auto p-6 md:p-8">
        
        <!-- Editor Area -->
        <div class="flex-grow flex flex-col bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden mb-6 relative">
            <div id="statusIndicator" class="absolute top-4 right-4 text-sm font-medium text-gray-400 flex items-center opacity-0 transition-opacity duration-300">
                <span class="relative flex h-3 w-3 mr-2">
                  <span id="ping1" class="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75 hidden"></span>
                  <span id="ping2" class="relative inline-flex rounded-full h-3 w-3 bg-red-500 hidden"></span>
                </span>
                <span id="statusText">Processing...</span>
            </div>
            
            <textarea id="editor" class="w-full h-full p-6 text-lg md:text-xl resize-none focus:outline-none text-gray-800 placeholder-gray-300" placeholder="Your dictation will appear here...&#10;&#10;Press the microphone button below to start."></textarea>
        </div>

        <!-- Controls -->
        <div class="flex justify-center items-center h-24">
            <button id="dictateBtn" class="bg-blue-600 hover:bg-blue-700 text-white rounded-full h-20 w-20 flex items-center justify-center shadow-lg transition-transform transform hover:scale-105 active:scale-95">
                <i id="micIcon" class="fas fa-microphone text-3xl"></i>
            </button>
        </div>
        <p class="text-center text-sm text-gray-400 mt-2">Click to start / stop recording.</p>

    </main>

    <script>
        let mediaRecorder;
        let audioChunks = [];
        let isRecording = false;

        const dictateBtn = document.getElementById('dictateBtn');
        const micIcon = document.getElementById('micIcon');
        const editor = document.getElementById('editor');
        const clearBtn = document.getElementById('clearBtn');
        const copyBtn = document.getElementById('copyBtn');
        const statusIndicator = document.getElementById('statusIndicator');
        const statusText = document.getElementById('statusText');
        const ping1 = document.getElementById('ping1');
        const ping2 = document.getElementById('ping2');

        clearBtn.addEventListener('click', () => { editor.value = ''; });
        
        copyBtn.addEventListener('click', () => {
            navigator.clipboard.writeText(editor.value)
                .then(() => alert('Copied to clipboard'))
                .catch(err => console.error('Failed to copy', err));
        });

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
                
                // Update UI: Recording State
                dictateBtn.classList.replace('bg-blue-600', 'bg-red-500');
                dictateBtn.classList.replace('hover:bg-blue-700', 'hover:bg-red-600');
                micIcon.classList.replace('fa-microphone', 'fa-stop');
                
                statusIndicator.style.opacity = '1';
                statusText.innerText = 'Recording...';
                ping1.classList.remove('hidden');
                ping2.classList.remove('hidden');

            } catch (err) {
                console.error("Microphone access denied:", err);
                alert("Please allow microphone access to use dictation.");
            }
        }

        function stopRecording() {
            if (mediaRecorder && mediaRecorder.state !== 'inactive') {
                mediaRecorder.stop();
                mediaRecorder.stream.getTracks().forEach(track => track.stop());
                isRecording = false;
                
                // Update UI: Processing State
                dictateBtn.classList.replace('bg-red-500', 'bg-blue-600');
                dictateBtn.classList.replace('hover:bg-red-600', 'hover:bg-blue-700');
                micIcon.classList.replace('fa-stop', 'fa-microphone');
                
                statusText.innerText = 'Transcribing...';
                ping1.classList.add('hidden');
                ping2.classList.remove('hidden');
                ping2.classList.replace('bg-red-500', 'bg-blue-500');
            }
        }

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
                const result = await response.json();
                
                if (result.text) {
                    if (editor.value.trim() !== '') {
                        editor.value += ' ' + result.text;
                    } else {
                        editor.value = result.text;
                    }
                    editor.scrollTop = editor.scrollHeight;
                }
            } catch (err) {
                console.error("Transcription error:", err);
                alert("Error during transcription. See console.");
            } finally {
                // Update UI: Idle State
                statusIndicator.style.opacity = '0';
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

    # We write it to a temporary wrapper because FasterWhisper relies on ffmpeg for formats like webm
    # Python's fp can be parsed if ffmpeg is in system path.
    file_like = io.BytesIO(audio_bytes)
    file_like.name = "audio.webm" # Gives ffmpeg a hint!

    try:
        # Use a greedy beam=1 for FAST processing, which helps with speed issues!
        # keep condition_on_previous_text=False to reduce hallucinations on short clips.
        logger.info("Transcribing segment...")
        text, segments = transcriber.transcribe(
            file_like,
            beam_size=1,
            condition_on_previous_text=False
        )
        return {"text": text}
    except Exception as e:
        logger.error("Error in dictation: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Transcription failed")

if __name__ == "__main__":
    import uvicorn
    # Make sure you installed 'uvicorn[standard]' or 'uvicorn' with 'fastapi'
    uvicorn.run(app, host="127.0.0.1", port=8000)
