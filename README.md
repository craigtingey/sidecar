# salestrainer-sidecar

FastAPI sidecar that downloads audio from S3, transcribes with OpenAI Whisper API,
and posts structured results back to WordPress.

## Environment Variables

Required:
- `AWS_REGION` - AWS region (default: us-east-1)
- `S3_BUCKET` - S3 bucket name for audio files
- `AWS_ACCESS_KEY_ID` - AWS access key
- `AWS_SECRET_ACCESS_KEY` - AWS secret key
- `OPENAI_API_KEY` - OpenAI API key for Whisper transcription
- `SIDE_CAR_SECRET` - Secret for authenticating requests (default: change-me)
- `WP_SIDE_CAR_CALLBACK` - WordPress callback URL (e.g., https://dev.salestrainer.pro/wp-json/salestrainer/v1/upload-complete)

Optional:
- `WP_SERVER_TOKEN` - WordPress server authentication token

## Setup

1. Copy `.env.example` to `.env` and fill in your values:
   ```bash
   cp .env.example .env
   ```

2. Start with Docker Compose:
   ```bash
   docker-compose up --build
   ```

## Development

### Install dependencies

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Run tests

```bash
pytest test_server.py -v
```

### Run locally without Docker

```bash
uvicorn server:app --host 0.0.0.0 --port 9001 --reload
```

## API

### POST /process-s3

Process an audio file from S3 through OpenAI Whisper transcription.

**Request body:**
```json
{
  "object_key": "path/to/audio.webm",
  "session_id": "optional-session-id",
  "scenario_id": 123,
  "mime_type": "audio/webm",
  "callback_url": "optional-override-callback-url",
  "secret": "your-secret"
}
```

**Response:**
```json
{
  "ok": true,
  "object_key": "path/to/audio.webm",
  "session_id": "optional-session-id",
  "scenario_id": 123,
  "size_bytes": 12345,
  "transcript": "Transcribed text here..."
}
```