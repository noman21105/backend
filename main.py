import os
import asyncio
import json
import base64
import re
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import AsyncGroq
import database
from auth import create_access_token, verify_token, get_user_from_token, UserAuth
import sys
import logging

# Configure logging to go to both console and file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("backend.log", encoding="utf-8")
    ]
)
log = logging.getLogger(__name__)

load_dotenv() 

app = FastAPI(title="Voice Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key = os.getenv("GROQ_API_KEY")
client = AsyncGroq(api_key=api_key) if api_key else None

SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You are a warm, empathetic, and highly human-like conversational partner. "
        "Speak casually as a human would in a normal voice conversation, occasionally using natural filler words like 'Hmm', 'Well', or 'Oh, I see'. "
        "Keep your responses very brief, punchy, and engaging (1-2 sentences max). "
        "Do NOT use any markdown, code blocks, bullet points, asterisks, or emojis. "
        "If the user interrupts you, stop your current thought and seamlessly shift gears to listen and adapt to the new topic."
    )
}

import edge_tts

# ----------------- TTS CONFIG -----------------
# The user originally requested Groq's whisper model, but Whisper is STT only.
# PlayAI/Canopy models require TOS acceptance on Groq console which causes HTTP 400.
# We'll use edge-tts (Microsoft Neural Voices) for a flawless, free "Alexa-like" backend generation.
EDGE_TTS_VOICE = "en-US-JennyNeural" # High-quality professional female voice
EDGE_TTS_FALLBACK_VOICE = "en-US-AriaNeural"
_EDGE_TTS_VOICES_CACHE = None
_EDGE_TTS_VOICES_LOCK = asyncio.Lock()

def sanitize_tts_text(text: str) -> str:
    """Normalize text so Edge TTS is less likely to reject it."""
    if not text:
        return ""
    # Remove control chars and collapse excessive whitespace.
    cleaned = "".join(ch for ch in text if ch == "\n" or ord(ch) >= 32)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    # Edge can fail on very long chunks; keep sentence chunk size bounded.
    return cleaned[:600]

def _contains_script(text: str, start: int, end: int) -> bool:
    return any(start <= ord(ch) <= end for ch in text)

async def get_available_voices() -> set[str]:
    global _EDGE_TTS_VOICES_CACHE
    if _EDGE_TTS_VOICES_CACHE is not None:
        return _EDGE_TTS_VOICES_CACHE
    async with _EDGE_TTS_VOICES_LOCK:
        if _EDGE_TTS_VOICES_CACHE is not None:
            return _EDGE_TTS_VOICES_CACHE
        try:
            voices = await edge_tts.list_voices()
            _EDGE_TTS_VOICES_CACHE = {v.get("ShortName", "") for v in voices if v.get("ShortName")}
        except Exception as e:
            log.warning(f"[TTS WARN] Could not list voices dynamically: {e}")
            _EDGE_TTS_VOICES_CACHE = set()
    return _EDGE_TTS_VOICES_CACHE

async def pick_tts_voice_candidates(text: str) -> list[str]:
    available = await get_available_voices()
    has_devanagari = _contains_script(text, 0x0900, 0x097F)
    has_arabic = _contains_script(text, 0x0600, 0x06FF) or _contains_script(text, 0x0750, 0x077F)

    # Prefer language-appropriate voices first, then multilingual/english fallbacks.
    preferred = []
    if has_devanagari:
        preferred.extend(["hi-IN-SwaraNeural", "hi-IN-MadhurNeural"])
    if has_arabic:
        preferred.extend(["ur-PK-UzmaNeural", "ur-PK-AsadNeural"])
    preferred.extend([
        EDGE_TTS_VOICE,
        EDGE_TTS_FALLBACK_VOICE,
        "en-US-AnaNeural",
        "en-US-GuyNeural",
    ])

    deduped = []
    seen = set()
    for voice in preferred:
        if voice in seen:
            continue
        seen.add(voice)
        # If available list is known, only keep valid voices.
        if available and voice not in available:
            continue
        deduped.append(voice)
    return deduped

def create_wav_header(sample_rate, num_samples, num_channels=1, bits_per_sample=16):
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = num_samples * num_channels * bits_per_sample // 8
    header = b'RIFF'
    header += (36 + data_size).to_bytes(4, 'little')
    header += b'WAVE'
    header += b'fmt '
    header += (16).to_bytes(4, 'little')  # Subchunk1Size
    header += (1).to_bytes(2, 'little')  # AudioFormat: 1 for PCM
    header += num_channels.to_bytes(2, 'little')
    header += sample_rate.to_bytes(4, 'little')
    header += byte_rate.to_bytes(4, 'little')
    header += block_align.to_bytes(2, 'little')
    header += bits_per_sample.to_bytes(2, 'little')
    header += b'data'
    header += data_size.to_bytes(4, 'little')
    return header

async def generate_tts(text: str, websocket: WebSocket, interrupt_event: asyncio.Event):
    """Stream MP3 chunks with buffering for smoother playback."""
    normalized_text = sanitize_tts_text(text)
    if not normalized_text:
        return

    async def stream_voice(voice: str) -> bool:
        communicate = edge_tts.Communicate(normalized_text, voice)
        buffered_audio = bytearray()
        emitted_audio = False
        flush_bytes = 32 * 1024

        async for chunk in communicate.stream():
            if interrupt_event.is_set():
                return emitted_audio
            if chunk["type"] == "audio" and chunk["data"]:
                emitted_audio = True
                buffered_audio.extend(chunk["data"])
                if len(buffered_audio) < flush_bytes:
                    continue
                await websocket.send_json({
                    "type": "audio",
                    "format": "mp3",
                    "content": base64.b64encode(bytes(buffered_audio)).decode("ascii")
                })
                buffered_audio.clear()

        if buffered_audio and not interrupt_event.is_set():
            await websocket.send_json({
                "type": "audio",
                "format": "mp3",
                "content": base64.b64encode(bytes(buffered_audio)).decode("ascii")
            })
        return emitted_audio

    candidates = await pick_tts_voice_candidates(normalized_text)
    if not candidates:
        candidates = [EDGE_TTS_VOICE, EDGE_TTS_FALLBACK_VOICE]

    last_error = None
    for voice in candidates:
        try:
            ok = await stream_voice(voice)
            if ok:
                return
            log.warning(f"[TTS WARN] Voice produced no audio: {voice}")
        except edge_tts.exceptions.NoAudioReceived as e:
            last_error = e
            log.warning(f"[TTS WARN] No audio from voice ({voice}): {e}")
        except Exception as e:
            last_error = e
            log.warning(f"[TTS WARN] Voice attempt failed ({voice}): {e}")

    log.error(f"[TTS ERROR] All voice attempts failed for text: {normalized_text[:80]!r}; last_error={last_error}")

# ----------------- AUTH ENDPOINTS -----------------

@app.post("/signup")
async def signup(user: UserAuth):
    created = await database.create_user_async(user.username, user.password)
    if not created:
        raise HTTPException(status_code=400, detail="Username already exists")
    token = create_access_token({"user_id": created["id"], "username": created["username"]})
    return {"token": token, "user": created}

@app.post("/login")
async def login(user: UserAuth):
    db_user = await database.verify_user_async(user.username, user.password)
    if not db_user:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    token = create_access_token({"user_id": db_user["id"], "username": db_user["username"]})
    return {"token": token, "user": db_user}

# ----------------- THREAD ENDPOINTS -----------------

@app.get("/threads")
async def get_threads(token: str):
    user_id = get_user_from_token(token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Unauthorized")
    threads = await database.get_threads_async(user_id)
    return threads

@app.post("/threads")
async def create_thread(token: str):
    user_id = get_user_from_token(token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Unauthorized")
    thread = await database.create_thread_async(user_id)
    return thread

@app.get("/threads/{thread_id}/messages")
async def get_messages(thread_id: int, token: str):
    user_id = get_user_from_token(token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Unauthorized")
    messages = await database.get_messages_async(thread_id)
    return messages

# ----------------- WEBSOCKET ENDPOINT -----------------
@app.websocket("/ws/{token}/{thread_id}")
async def websocket_endpoint(websocket: WebSocket, token: str, thread_id: int):
    await websocket.accept()
    
    user_id = get_user_from_token(token)
    if not user_id:
        await websocket.send_json({"error": "Unauthorized"})
        await websocket.close(code=1008)
        return
        
    if not client:
        await websocket.send_json({"error": "Groq API key missing on server"})
        await websocket.close(code=1011)
        return

    # Shared state between reader and processor
    message_queue = asyncio.Queue()
    interrupt_event = asyncio.Event()
    disconnect_event = asyncio.Event()

    async def keepalive_task():
        """Sends a ping every 20s to prevent idle WebSocket disconnection."""
        try:
            while not disconnect_event.is_set():
                await asyncio.sleep(20)
                if disconnect_event.is_set():
                    break
                try:
                    await websocket.send_json({"type": "ping"})
                except Exception:
                    break
        except asyncio.CancelledError:
            pass

    async def reader_task():
        """Continuously reads WebSocket messages from the client."""
        try:
            while True:
                raw = await websocket.receive_text()
                data = json.loads(raw)
                msg_type = data.get("type")

                if msg_type == "chat" or msg_type == "audio_input":
                    # Signal any in-progress generation to stop
                    interrupt_event.set()
                    # Put the new message in the queue for the processor
                    await message_queue.put(data)
                elif msg_type == "interrupt":
                    # User explicitly requested interrupt (e.g. started speaking)
                    interrupt_event.set()
                # Ignore 'ping' type messages from client (if any)
        except WebSocketDisconnect:
            disconnect_event.set()
        except Exception as e:
            log.error(f"Reader error: {e}", exc_info=True)
            disconnect_event.set()

    async def prep_llm():
        history = await database.get_messages_async(thread_id)
        history = history[-10:]
        messages_for_llm = [SYSTEM_PROMPT] + [
            {"role": m["role"], "content": m["content"]} for m in history
        ]
        return messages_for_llm

    async def transcribe_audio(data):
        format = data.get("format", "webm")
        content = data.get("content", "")
        audio_bytes = base64.b64decode(content)
        
        if format == "pcm":
            sample_rate = data.get("sampleRate", 44100)
            num_samples = len(audio_bytes) // 2
            MIN_AUDIO_SIZE = int(44100 * 0.5 * 2)
            if len(audio_bytes) < MIN_AUDIO_SIZE:
                content = data.get("fallbackText", "")
                if not content.strip():
                    return ""
                await websocket.send_json({
                    "type": "transcription_update",
                    "content": content,
                    "msgId": data.get("msgId")
                })
                return content
            else:
                wav_header = create_wav_header(sample_rate, num_samples)
                audio_file = ("audio.wav", wav_header + audio_bytes, "audio/wav")
                transcription_resp = await client.audio.transcriptions.create(
                    file=audio_file,
                    model="whisper-large-v3",
                    temperature=0.0
                )
                content = transcription_resp.text.strip()
                if not content:
                    content = "I do not listen."
                await websocket.send_json({
                    "type": "transcription_update",
                    "content": content,
                    "msgId": data.get("msgId")
                })
                return content
        else:
            # WebM format
            hex_signature = audio_bytes[:16].hex(' ').upper()
            log.info(f"[DEBUG] Received audio: {len(audio_bytes)} bytes, signature=[{hex_signature}]")
            MIN_AUDIO_SIZE = 20000
            if len(audio_bytes) < MIN_AUDIO_SIZE:
                content = data.get("fallbackText", "")
                if not content.strip():
                    return ""
                await websocket.send_json({
                    "type": "transcription_update",
                    "content": content,
                    "msgId": data.get("msgId")
                })
                return content
            else:
                audio_file = ("audio.webm", audio_bytes, "audio/webm")
                await websocket.send_json({"type": "transcribing"})
                transcription_resp = await client.audio.transcriptions.create(
                    file=audio_file,
                    model="whisper-large-v3",
                    temperature=0.0
                )
                content = transcription_resp.text.strip()
                if not content:
                    content = "I do not listen."
                await websocket.send_json({
                    "type": "transcription_update",
                    "content": content,
                    "msgId": data.get("msgId")
                })
                return content

    async def processor_task():
        """Processes messages from the queue, streams LLM responses."""
        try:
            while not disconnect_event.is_set():
                # Wait for a message to process, but also watch for disconnect
                try:
                    data = await asyncio.wait_for(message_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue  # No message yet, loop and check disconnect

                content = data.get("content", "")
                msg_type = data.get("type")
                
                log.info(f"[DEBUG] Received queue message: type={msg_type}, len={len(content)}")
                
                if msg_type == "audio_input":
                    try:
                        content = await transcribe_audio(data)
                        if not content:
                            continue

                    except Exception as e:
                        log.error(f"[Whisper Error] {e}", exc_info=True)
                        await websocket.send_json({
                            "type": "error",
                            "error": f"Audio transcription failed: {str(e)}"
                        })
                        # Fallback to the browser's generic text so the assistant doesn't freeze
                        content = data.get("fallbackText", "")
                
                log.info(f"[DEBUG] Content before LLM loop: '{content}'")
                if not content.strip():
                    continue

                # Clear interrupt flag before starting a new generation
                interrupt_event.clear()

                # Save user message
                await database.add_message_async(thread_id, "user", content)

                # Load conversation history (last 10 messages for decent context)
                history = await database.get_messages_async(thread_id)
                history = history[-10:]

                messages_for_llm = [SYSTEM_PROMPT] + [
                    {"role": m["role"], "content": m["content"]} for m in history
                ]

                # Stream from Groq
                chat_completion = await client.chat.completions.create(
                    messages=messages_for_llm,
                    model="llama-3.1-8b-instant",
                    temperature=0.7,
                    stream=True,
                    max_tokens=150
                )

                full_response = ""
                sentence_buffer = ""
                was_interrupted = False

                tts_queue = asyncio.Queue()
                
                async def tts_worker():
                    try:
                        while not interrupt_event.is_set():
                            text = await tts_queue.get()
                            if text is None:  # Poison pill
                                tts_queue.task_done()
                                break
                            if interrupt_event.is_set():
                                tts_queue.task_done()
                                break
                            await generate_tts(text, websocket, interrupt_event)
                            tts_queue.task_done()
                    except asyncio.CancelledError:
                        pass

                tts_worker_task = asyncio.create_task(tts_worker())

                async for chunk in chat_completion:
                    if interrupt_event.is_set():
                        was_interrupted = True
                        log.info("[Interrupt] Stopping generation mid-stream")
                        break

                    token_text = chunk.choices[0].delta.content
                    if token_text:
                        full_response += token_text
                        sentence_buffer += token_text
                        
                        try:
                            await websocket.send_json({
                                "type": "token",
                                "content": token_text
                            })
                        except Exception:
                            tts_worker_task.cancel()
                            return  # WebSocket closed

                        # Split on commas too for much faster Time-To-First-Audio
                        if any(p in token_text for p in [".", "!", "?", "\n", ","]):
                            sentences = re.split(r'(?<=[.!?\n,])\s+', sentence_buffer)
                            for s in sentences[:-1]:
                                if s.strip():
                                    await tts_queue.put(s.strip())
                            sentence_buffer = sentences[-1]

                if was_interrupted:
                    tts_worker_task.cancel()
                    try:
                        await tts_worker_task
                    except asyncio.CancelledError:
                        pass
                    try:
                        await websocket.send_json({"type": "interrupted"})
                    except Exception:
                        return
                else:
                    if sentence_buffer.strip():
                        await tts_queue.put(sentence_buffer.strip())
                    await tts_queue.put(None)  # Signal worker to stop
                    
                    # Wait for worker to finish processing queue
                    try:
                        await tts_worker_task
                    except asyncio.CancelledError:
                        pass
                        
                    if not interrupt_event.is_set():
                        try:
                            await websocket.send_json({"type": "done"})
                        except Exception:
                            return

                # Save whatever was generated (full or partial)
                if full_response:
                    await database.add_message_async(thread_id, "assistant", full_response)

        except Exception as e:
            log.error(f"Processor error: {e}", exc_info=True)

    # Run all tasks concurrently; when any finishes, cancel the others
    reader = asyncio.create_task(reader_task())
    processor = asyncio.create_task(processor_task())
    keepalive = asyncio.create_task(keepalive_task())
    
    done, pending = await asyncio.wait(
        [reader, processor, keepalive],
        return_when=asyncio.FIRST_COMPLETED
    )
    for task in pending:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    log.info(f"Client disconnected from thread {thread_id}")
