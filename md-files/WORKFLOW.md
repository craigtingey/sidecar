# Sidecar Development Workflow

## Overview

The sidecar is a FastAPI service that handles asynchronous audio transcription for the SalesTrainer platform. It runs independently and communicates with WordPress via REST callbacks.

## Repository Structure

```
sidecar/
├── server.py           # Main FastAPI application
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container definition
├── .env               # Environment variables (NOT committed)
├── .gitignore         # Excludes secrets and temp files
└── README.md          # Usage documentation
```

## Key Changes (Latest)

### Authentication Update
- **Old**: Sent token via `Authorization: Bearer` header
- **New**: Sends token in JSON body (primary) + optional `X-Server-Token` header
- **Why**: WAF strips `Authorization` header on `/wp-json` routes

### Implementation
```python
# JSON body (always sent)
result["server_token"] = WP_SERVER_TOKEN

# Optional header (controlled by env var)
if WP_SERVER_TOKEN and SEND_X_SERVER_TOKEN_HEADER:
    headers["X-Server-Token"] = WP_SERVER_TOKEN
```

## Development Workflow

### 1. Making Changes

```bash
cd /var/www/salestrainer-dev/sidecar

# Activate virtual environment
source venv/bin/activate

# Edit server.py or other files
# Make your changes...
```

### 2. Testing Locally

```bash
# Stop running sidecar
pkill -f "uvicorn server:app"

# Load environment variables and start
export $(grep -v '^#' .env | xargs)
uvicorn server:app --host 127.0.0.1 --port 9001 --reload

# Test endpoint
curl -X POST http://localhost:9001/process-s3 \
  -H "Content-Type: application/json" \
  -d '{"object_key":"test.webm","session_id":"test","scenario_id":1,"secret":"<SIDE_CAR_SECRET>"}'
```

### 3. Committing Changes

```bash
# Stage changes
git add server.py README.md .gitignore

# Commit with descriptive message
git commit -m "Send WP_SERVER_TOKEN in JSON body for WAF compatibility"

# Push to feature branch
git push origin feature/sidecar-fastapi
```

### 4. Deploying (Production)

The sidecar runs on the same server but as a separate service.

**Start/Restart:**
```bash
# Using systemd (if configured)
sudo systemctl restart sidecar.service

# Or manually with screen/tmux
cd /var/www/salestrainer-dev/sidecar
source venv/bin/activate
export $(grep -v '^#' .env | xargs)
uvicorn server:app --host 127.0.0.1 --port 9001
```

**Note**: Unlike wp-custom, there's no `DOCROOT` or complex deployment script. Just restart the service with updated code.

## Environment Variables

Required in `.env` (never commit this file):

```bash
# AWS S3 Configuration
AWS_REGION=us-west-2
S3_BUCKET=stp1
AWS_ACCESS_KEY_ID=your_key_here
AWS_SECRET_ACCESS_KEY=your_secret_here

# OpenAI API
OPENAI_API_KEY=sk-proj-...

# Sidecar Security
SIDE_CAR_SECRET=your_random_secret

# WordPress Integration
WP_SIDE_CAR_CALLBACK=https://dev.salestrainer.pro/wp-json/salestrainer/v1/upload-complete
WP_SERVER_TOKEN=your_wordpress_token

# Optional Settings
SEND_X_SERVER_TOKEN_HEADER=true  # Default: true
MOCK_TRANSCRIBE=1                # For testing without OpenAI
```

## Testing the Full Flow

```bash
# 1. Start sidecar
cd /var/www/salestrainer-dev/sidecar
source venv/bin/activate
export $(grep -v '^#' .env | xargs)
uvicorn server:app --host 127.0.0.1 --port 9001 --reload &

# 2. Trigger upload (simulates WordPress calling sidecar)
curl -X POST "https://dev.salestrainer.pro/wp-json/salestrainer/v1/register-upload" \
  -H "Content-Type: application/json" \
  -d '{
    "objectKey": "test.webm",
    "session_id": "test-session",
    "scenario_id": 1,
    "mimeType": "audio/webm",
    "size": 21847
  }'

# 3. Verify in WordPress database
mysql -u root wordpress_dev -e "
  SELECT ID, post_title, post_date FROM u2Boh_posts 
  WHERE post_type = 'st_upload' 
  ORDER BY post_date DESC LIMIT 3;
"

# 4. Check transcript was saved
mysql -u root wordpress_dev -e "
  SELECT post_id, meta_key, meta_value 
  FROM u2Boh_postmeta 
  WHERE post_id = <ID_FROM_ABOVE> 
  AND meta_key = 'transcript';
"
```

## Common Issues

### Sidecar Not Responding
- Check if process is running: `ps aux | grep uvicorn`
- Check logs: Look at terminal output where sidecar was started
- Verify port 9001 is not in use: `lsof -i :9001`

### Authentication Failures
- Ensure `WP_SERVER_TOKEN` matches in both `.env` and WordPress `wp-config.php`
- Check WordPress logs for token validation errors
- Verify token is being sent: Add debug print in `server.py`

### Transcription Failures
- Check `OPENAI_API_KEY` is valid
- Use `MOCK_TRANSCRIBE=1` to test without OpenAI
- Verify S3 file exists and is accessible

### WordPress Callback Fails
- Check `WP_SIDE_CAR_CALLBACK` URL is correct
- Verify WordPress is reachable from sidecar host
- Check WordPress error logs: `/var/www/vhosts/salestrainer.pro/dev/wp-content/debug.log`

## Security Notes

- **Never commit `.env`** - contains AWS keys, API tokens
- `.gitignore` now properly excludes secrets and temp files
- Keep `WP_SERVER_TOKEN` synchronized between sidecar and WordPress
- Only sidecar should know `SIDE_CAR_SECRET` (validates incoming requests)

## Next Steps

- [ ] Add systemd service file for automatic startup
- [ ] Add logging to file instead of stdout
- [ ] Add health check endpoint (`/health`)
- [ ] Add metrics endpoint for monitoring
- [ ] Consider moving to Docker Compose for easier deployment
