from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import os
import boto3
import aiohttp
import tempfile
import shutil
import numpy as np
import librosa
import json
import re
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
S3_BUCKET = os.environ.get("S3_BUCKET")
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
SIDE_CAR_SECRET = os.environ.get("SIDE_CAR_SECRET", "change-me")
WP_SIDE_CAR_CALLBACK = os.environ.get("WP_SIDE_CAR_CALLBACK")  # e.g., https://dev.salestrainer.pro/wp-json/salestrainer/v1/upload-complete
WP_SERVER_TOKEN = os.environ.get("WP_SERVER_TOKEN")
# Optional: also send X-Server-Token header in addition to JSON body (redundant but safe)
SEND_X_SERVER_TOKEN_HEADER = os.environ.get("SEND_X_SERVER_TOKEN_HEADER", "true").lower() == "true"

s3 = boto3.client(
    "s3",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)

# === PHASE 1: CORE AUDIO METRICS EXTRACTION ===

def extract_core_audio_metrics(audio, sample_rate, words, labeled_words):
    """
    Extract 5 core audio metrics for Phase 1 MVP:
    1. Speaking pace (WPM) for USER and BOT
    2. Talk ratio (% speaking time)
    3. Filler words per minute
    4. Average user energy (RMS)
    5. Response latency (filtered 200-3000ms)
    
    Returns dict with metrics_version 1.0
    """
    metrics = {
        "metrics_version": "1.0",
        "speaking_pace_user_wpm": None,
        "speaking_pace_bot_wpm": None,
        "talk_ratio_user_pct": None,
        "talk_ratio_bot_pct": None,
        "filler_words_count": 0,
        "filler_words_per_minute": None,
        "avg_user_energy_rms": None,
        "avg_response_latency_ms": None,
        "conversation_duration_seconds": None
    }
    
    if not words or len(words) == 0:
        return metrics
    
    # Calculate conversation duration
    first_word_time = words[0].get('start', 0)
    last_word_time = words[-1].get('end', 0)
    duration_seconds = last_word_time - first_word_time
    metrics["conversation_duration_seconds"] = round(duration_seconds, 2)
    
    if duration_seconds <= 0:
        return metrics
    
    duration_minutes = duration_seconds / 60.0
    
    # Separate user and bot words
    user_words = []
    bot_words = []
    
    for speaker, word_text, left_e, right_e in labeled_words:
        if speaker == "USER":
            user_words.append(word_text)
        elif speaker == "BOT":
            bot_words.append(word_text)
    
    # 2. Talk ratio (calculate ACTUAL speaking time per speaker, not including silences)
    user_speaking_time = 0
    bot_speaking_time = 0
    user_segments = []  # Track user speech segments for energy calculation
    bot_segments = []
    
    current_speaker = None
    segment_start = None
    
    for i, (speaker, word_text, left_e, right_e) in enumerate(labeled_words):
        word_data = words[i] if i < len(words) else None
        if not word_data:
            continue
            
        word_start = word_data.get('start', 0)
        word_end = word_data.get('end', 0)
        
        if speaker != current_speaker:
            # Save previous segment
            if current_speaker and segment_start is not None:
                segment_duration = word_start - segment_start
                if current_speaker == "USER":
                    user_speaking_time += segment_duration
                    user_segments.append((segment_start, word_start))
                elif current_speaker == "BOT":
                    bot_speaking_time += segment_duration
                    bot_segments.append((segment_start, word_start))
            
            # Start new segment
            current_speaker = speaker
            segment_start = word_start
    
    # Add final segment
    if current_speaker and segment_start is not None and len(words) > 0:
        final_end = words[-1].get('end', 0)
        segment_duration = final_end - segment_start
        if current_speaker == "USER":
            user_speaking_time += segment_duration
            user_segments.append((segment_start, final_end))
        elif current_speaker == "BOT":
            bot_speaking_time += segment_duration
            bot_segments.append((segment_start, final_end))
    
    total_speaking_time = user_speaking_time + bot_speaking_time
    if total_speaking_time > 0:
        metrics["talk_ratio_user_pct"] = round((user_speaking_time / total_speaking_time) * 100, 1)
        metrics["talk_ratio_bot_pct"] = round((bot_speaking_time / total_speaking_time) * 100, 1)
    
    # 1. Speaking pace (WPM) - based on ACTUAL speaking time, not total duration
    if user_words and user_speaking_time > 0:
        user_speaking_minutes = user_speaking_time / 60.0
        metrics["speaking_pace_user_wpm"] = round(len(user_words) / user_speaking_minutes, 1)
    
    if bot_words and bot_speaking_time > 0:
        bot_speaking_minutes = bot_speaking_time / 60.0
        metrics["speaking_pace_bot_wpm"] = round(len(bot_words) / bot_speaking_minutes, 1)
    
    # 3. Filler words detection
    filler_patterns = [
        r'\bum+\b', r'\buh+\b', r'\blike\b', r'\byou know\b', 
        r'\bso+\b', r'\bakshually\b', r'\bactually\b', r'\bbasically\b',
        r'\bliterally\b', r'\bjust\b', r'\bkinda\b', r'\bsorta\b',
        r'\bI mean\b', r'\byeah\b', r'\bok\b', r'\bokay\b'
    ]
    
    user_text = ' '.join(user_words).lower()
    filler_count = 0
    for pattern in filler_patterns:
        filler_count += len(re.findall(pattern, user_text, re.IGNORECASE))
    
    metrics["filler_words_count"] = filler_count
    if duration_minutes > 0:
        metrics["filler_words_per_minute"] = round(filler_count / duration_minutes, 2)
    
    # 4. Average user energy (RMS) - only during actual user speech segments
    if user_segments and len(user_segments) > 0:
        if audio.ndim == 2 and audio.shape[0] == 2:
            # Stereo: left channel = USER
            left_channel = audio[0]
            
            # Extract audio samples only during user speech segments
            user_audio_samples = []
            for seg_start, seg_end in user_segments:
                start_sample = int(seg_start * sample_rate)
                end_sample = int(seg_end * sample_rate)
                # Ensure bounds are within audio length
                start_sample = max(0, start_sample)
                end_sample = min(len(left_channel), end_sample)
                if end_sample > start_sample:
                    user_audio_samples.extend(left_channel[start_sample:end_sample])
            
            # Calculate RMS only on user speech samples
            if len(user_audio_samples) > 0:
                user_energy = np.sqrt(np.mean(np.array(user_audio_samples)**2))
                metrics["avg_user_energy_rms"] = round(float(user_energy), 6)
        elif audio.ndim == 1:
            # Mono: extract segments and calculate energy
            user_audio_samples = []
            for seg_start, seg_end in user_segments:
                start_sample = int(seg_start * sample_rate)
                end_sample = int(seg_end * sample_rate)
                start_sample = max(0, start_sample)
                end_sample = min(len(audio), end_sample)
                if end_sample > start_sample:
                    user_audio_samples.extend(audio[start_sample:end_sample])
            
            if len(user_audio_samples) > 0:
                user_energy = np.sqrt(np.mean(np.array(user_audio_samples)**2))
                metrics["avg_user_energy_rms"] = round(float(user_energy), 6)
    
    # 5. Response latency (filtered 200-3000ms to exclude artificial Convai delays)
    response_latencies = []
    prev_speaker = None
    prev_end_time = None
    
    for i, (speaker, word_text, left_e, right_e) in enumerate(labeled_words):
        word_data = words[i] if i < len(words) else None
        if not word_data:
            continue
            
        word_start = word_data.get('start', 0)
        
        # Detect speaker transition (BOT → USER or USER → BOT)
        if prev_speaker and prev_speaker != speaker and prev_end_time:
            gap_ms = (word_start - prev_end_time) * 1000
            
            # Filter to natural human response range (200-3000ms)
            # Excludes: <200ms (overlap/system), >3000ms (connection delay)
            if 200 <= gap_ms <= 3000:
                response_latencies.append(gap_ms)
        
        prev_speaker = speaker
        prev_end_time = word_data.get('end', 0)
    
    if response_latencies:
        metrics["avg_response_latency_ms"] = round(sum(response_latencies) / len(response_latencies), 1)
    
    return metrics

class ProcessRequest(BaseModel):
    object_key: str
    session_id: str | None = None
    scenario_id: int | None = None
    mime_type: str | None = "audio/webm"
    callback_url: str | None = None
    secret: str | None = None

@app.post("/process-s3")
async def process_s3(payload: ProcessRequest):
    if payload.secret != SIDE_CAR_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

    object_key = payload.object_key
    tmp_dir = tempfile.mkdtemp(prefix="st-sidecar-")
    local_path = os.path.join(tmp_dir, os.path.basename(object_key))

    try:
        s3.download_file(S3_BUCKET, object_key, local_path)
    except Exception as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"failed to download object: {e}")

    try:
        async with aiohttp.ClientSession() as session:
            headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
            data = aiohttp.FormData()
            data.add_field("file", open(local_path, "rb"), filename=os.path.basename(local_path), content_type=payload.mime_type or "audio/webm")
            data.add_field("model", "whisper-1")
            data.add_field("response_format", "verbose_json")
            data.add_field("timestamp_granularities[]", "word")
            async with session.post("https://api.openai.com/v1/audio/transcriptions", headers=headers, data=data, timeout=300) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise Exception(f"OpenAI transcription error {resp.status}: {text}")
                resp_json = await resp.json()
                transcript_text = resp_json.get("text", "")
                words = resp_json.get("words", [])
    except Exception as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(status_code=502, detail=str(e))

    # Perform speaker detection if we have word timestamps
    formatted_transcript = transcript_text
    if words:
        try:
            print(f"🎯 Analyzing stereo audio for speaker detection...")
            print(f"🎯 Audio file path: {local_path}")
            print(f"🎯 File exists: {os.path.exists(local_path)}")
            print(f"🎯 File size: {os.path.getsize(local_path) if os.path.exists(local_path) else 'N/A'}")
            
            # Convert WebM to WAV for reliable stereo processing
            import subprocess
            wav_path = local_path.replace('.webm', '.wav')
            try:
                # Use native ffmpeg (not snap version which has display issues)
                result = subprocess.run([
                    '/usr/bin/ffmpeg', '-nostdin', '-i', local_path, 
                    '-acodec', 'pcm_s16le', '-ac', '2', '-ar', '44100',
                    '-y', wav_path
                ], capture_output=True, text=True)
                
                # Check if WAV was actually created (ignore stderr warnings)
                if not os.path.exists(wav_path) or os.path.getsize(wav_path) == 0:
                    stderr_msg = result.stderr[:500] if result.stderr else "No error output"
                    print(f"⚠️ FFmpeg failed - no WAV created. Stderr: {stderr_msg}")
                    raise Exception(f"FFmpeg conversion failed")
                    
                audio_path = wav_path
                print(f"✅ Converted WebM to WAV: {os.path.getsize(wav_path)} bytes")
            except Exception as conv_err:
                print(f"⚠️ WebM conversion failed: {conv_err}")
                print(f"⚠️ Trying direct WebM load with librosa...")
                audio_path = local_path
            
            # Load audio as stereo (don't convert to mono)
            y, sr = librosa.load(audio_path, sr=None, mono=False)
            
            # Check if stereo
            if y.ndim == 2 and y.shape[0] == 2:
                print(f"✅ Stereo audio detected: {y.shape}")
                
                # Channel assignment: left=user mic, right=bot audio
                left_channel = y[0]   # LEFT channel = USER
                right_channel = y[1]  # RIGHT channel = BOT
                
                # Calculate overall channel energy to verify separation
                left_total_energy = np.sqrt(np.mean(left_channel**2))
                right_total_energy = np.sqrt(np.mean(right_channel**2))
                print(f"📊 Overall channel energy - Left(USER): {left_total_energy:.6f}, Right(BOT): {right_total_energy:.6f}")
                
                # Assign speaker to each word based on channel energy
                labeled_words = []
                for word in words:
                    word_text = word.get('word', '')
                    start_time = word.get('start', 0)
                    end_time = word.get('end', 0)
                    
                    # Calculate sample indices
                    start_sample = int(start_time * sr)
                    end_sample = int(end_time * sr)
                    
                    # Ensure we don't go out of bounds
                    start_sample = max(0, start_sample)
                    end_sample = min(len(left_channel), end_sample)
                    
                    if end_sample > start_sample:
                        # Calculate RMS energy for each channel
                        left_segment = left_channel[start_sample:end_sample]
                        right_segment = right_channel[start_sample:end_sample]
                        
                        left_energy = np.sqrt(np.mean(left_segment**2))
                        right_energy = np.sqrt(np.mean(right_segment**2))
                        
                        # Use a threshold to determine speaker
                        # If both energies are very low, it's silence/ambiguous
                        total_energy = left_energy + right_energy
                        
                        if total_energy < 0.00005:  # Near silence
                            speaker = "UNKNOWN"
                        elif left_energy > right_energy * 1.15:  # Left is louder (15% threshold)
                            speaker = "USER"  # Left = USER
                        elif right_energy > left_energy * 1.15:  # Right is louder (15% threshold)
                            speaker = "BOT"   # Right = BOT
                        else:
                            # Too close, mark as UNKNOWN to be smoothed later
                            speaker = "UNKNOWN"
                        
                        labeled_words.append((speaker, word_text, left_energy, right_energy))
                    else:
                        labeled_words.append(("UNKNOWN", word_text, 0, 0))
                
                # Debug: Log first 20 words with energy values
                print(f"🔍 First 20 words with energy values:")
                for i, (speaker, word_text, left_e, right_e) in enumerate(labeled_words[:20]):
                    ratio = left_e / (right_e + 1e-10) if right_e > 1e-10 else 999
                    print(f"  {i+1}. '{word_text}' -> {speaker} (Left/USER:{left_e:.6f} Right/BOT:{right_e:.6f} Ratio:{ratio:.2f})")
                
                # Apply smoothing: fix UNKNOWN words by looking at neighbors
                smoothed_words = []
                window_size = 3  # Look at 3 words before and after
                
                for i, (speaker, word, left_e, right_e) in enumerate(labeled_words):
                    if speaker == "UNKNOWN":
                        # Collect nearby non-UNKNOWN speakers
                        nearby_speakers = []
                        
                        # Look backward
                        for j in range(max(0, i-window_size), i):
                            if labeled_words[j][0] != "UNKNOWN":
                                nearby_speakers.append(labeled_words[j][0])
                        
                        # Look forward
                        for j in range(i+1, min(len(labeled_words), i+window_size+1)):
                            if labeled_words[j][0] != "UNKNOWN":
                                nearby_speakers.append(labeled_words[j][0])
                        
                        # Use majority vote from neighbors
                        if nearby_speakers:
                            user_count = nearby_speakers.count("USER")
                            bot_count = nearby_speakers.count("BOT")
                            speaker = "USER" if user_count > bot_count else "BOT"
                        # If no neighbors, use energy comparison even if close
                        elif left_e > right_e:
                            speaker = "USER"
                        else:
                            speaker = "BOT"
                    
                    smoothed_words.append((speaker, word, left_e, right_e))
                
                # Format transcript with speaker labels
                formatted_lines = []
                current_speaker = None
                current_text = []
                
                for speaker, word, left_e, right_e in smoothed_words:
                    if speaker != current_speaker:
                        # Save previous line
                        if current_speaker and current_text:
                            formatted_lines.append(f"[{current_speaker}]: {' '.join(current_text).strip()}")
                        # Start new line
                        current_speaker = speaker
                        current_text = [word]
                    else:
                        current_text.append(word)
                
                # Add final line
                if current_speaker and current_text:
                    formatted_lines.append(f"[{current_speaker}]: {' '.join(current_text).strip()}")
                
                formatted_transcript = "\n".join(formatted_lines)
                print(f"✅ Speaker detection complete: {len(formatted_lines)} speaker turns")
            else:
                print(f"⚠️ Audio is mono, skipping speaker detection")
            
            # Clean up temporary WAV file
            if 'wav_path' in locals() and os.path.exists(wav_path):
                try:
                    os.remove(wav_path)
                except Exception:
                    pass
        except Exception as e:
            print(f"⚠️ Speaker detection failed: {e}")
            # Fall back to original transcript
            formatted_transcript = transcript_text
            # Clean up temporary WAV file
            if 'wav_path' in locals() and os.path.exists(wav_path):
                try:
                    os.remove(wav_path)
                except Exception:
                    pass

    # === PHASE 1: EXTRACT CORE AUDIO METRICS ===
    audio_metrics = None
    if words and 'y' in locals() and 'sr' in locals():
        try:
            print("📊 Extracting audio metrics...")
            audio_metrics = extract_core_audio_metrics(
                audio=y,
                sample_rate=sr,
                words=words,
                labeled_words=smoothed_words if 'smoothed_words' in locals() else []
            )
            print(f"✅ Metrics extracted: {json.dumps(audio_metrics, indent=2)}")
        except Exception as e:
            print(f"⚠️ Failed to extract audio metrics: {e}")
            audio_metrics = None

    file_stat = os.stat(local_path)
    result = {
        "ok": True,
        "object_key": object_key,
        "session_id": payload.session_id,
        "scenario_id": payload.scenario_id,
        "size_bytes": file_stat.st_size,
        "transcript": formatted_transcript,
    }
    
    # Add audio metrics if available (Phase 1)
    if audio_metrics:
        result["audio_metrics_raw"] = audio_metrics

    # Add server token to result if available (most WAF-safe approach)
    if WP_SERVER_TOKEN:
        result["server_token"] = WP_SERVER_TOKEN

    callback = payload.callback_url or WP_SIDE_CAR_CALLBACK
    if callback:
        print(f"📤 Calling WordPress callback: {callback}")
        print(f"📦 Payload: object_key={object_key}, session_id={payload.session_id}, transcript length={len(transcript_text)}")
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Content-Type": "application/json",
                }
                # Optionally send token as header (redundant but safe fallback)
                if WP_SERVER_TOKEN and SEND_X_SERVER_TOKEN_HEADER:
                    headers["X-Server-Token"] = WP_SERVER_TOKEN
                    headers["Authorization"] = f"Bearer {WP_SERVER_TOKEN}"
                async with session.post(callback, json=result, headers=headers, timeout=30) as resp:
                    status = resp.status
                    text = await resp.text()
                    print(f"✅ Callback response: {status}")
                    if status != 200:
                        print(f"❌ Callback error: {text[:500]}")
        except Exception as e:
            print(f"❌ Failed to POST to WP callback: {e}")

    shutil.rmtree(tmp_dir, ignore_errors=True)
    return result


    # Add this snippet to the bottom of server.py (or paste into server.py)
from fastapi import File, UploadFile
import os
import aiofiles

@app.post("/transcribe-upload")
async def transcribe_upload(file: UploadFile = File(...)):
    """
    Local test endpoint: POST a multipart file under 'file' and get a transcript back.
    Behavior:
      - If environment var MOCK_TRANSCRIBE=1, returns a fake transcript (no OpenAI call).
      - Else requires OPENAI_API_KEY env var and will call OpenAI Whisper transcription API.
    """
    # Save upload to a temp file
    tmp_dir = os.path.join("/tmp", "st_sidecar_uploads")
    os.makedirs(tmp_dir, exist_ok=True)
    local_path = os.path.join(tmp_dir, file.filename)
    async with aiofiles.open(local_path, 'wb') as f:
        content = await file.read()
        await f.write(content)

    # Mock mode (fast, no OpenAI)
    if os.environ.get("MOCK_TRANSCRIBE", "") == "1":
        # Return a simple made-up transcript for testing
        size = os.path.getsize(local_path)
        return {"ok": True, "transcript": f"[MOCK] Received file {file.filename} ({size} bytes)"}

    # Real mode: send to OpenAI Whisper transcription endpoint
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        return {"ok": False, "error": "OPENAI_API_KEY not set. Set MOCK_TRANSCRIBE=1 to test without OpenAI."}

    import aiohttp
    try:
        async with aiohttp.ClientSession() as session:
            headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
            data = aiohttp.FormData()
            data.add_field("file", open(local_path, "rb"), filename=file.filename, content_type=file.content_type or "audio/webm")
            data.add_field("model", "whisper-1")
            async with session.post("https://api.openai.com/v1/audio/transcriptions", headers=headers, data=data, timeout=300) as resp:
                text = await resp.text()
                if resp.status != 200:
                    raise Exception(f"OpenAI error {resp.status}: {text}")
                j = await resp.json()
                transcript = j.get("text", "")
    except Exception as e:
        return {"ok": False, "error": str(e)}
    finally:
        try:
            os.remove(local_path)
        except Exception:
            pass

    return {"ok": True, "transcript": transcript}