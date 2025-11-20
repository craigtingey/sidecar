from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import os
import boto3
import aiohttp
import tempfile
import shutil
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
            async with session.post("https://api.openai.com/v1/audio/transcriptions", headers=headers, data=data, timeout=300) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise Exception(f"OpenAI transcription error {resp.status}: {text}")
                resp_json = await resp.json()
                transcript_text = resp_json.get("text", "")
    except Exception as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(status_code=502, detail=str(e))

    file_stat = os.stat(local_path)
    result = {
        "ok": True,
        "object_key": object_key,
        "session_id": payload.session_id,
        "scenario_id": payload.scenario_id,
        "size_bytes": file_stat.st_size,
        "transcript": transcript_text,
    }

    # Add server token to result if available (most WAF-safe approach)
    if WP_SERVER_TOKEN:
        result["server_token"] = WP_SERVER_TOKEN

    callback = payload.callback_url or WP_SIDE_CAR_CALLBACK
    if callback:
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Content-Type": "application/json",
                }
                # Optionally send token as header (redundant but safe fallback)
                if WP_SERVER_TOKEN and SEND_X_SERVER_TOKEN_HEADER:
                    headers["X-Server-Token"] = WP_SERVER_TOKEN
                await session.post(callback, json=result, headers=headers, timeout=30)
        except Exception as e:
            print("Failed to POST to WP callback:", e)

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