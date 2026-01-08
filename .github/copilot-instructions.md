# Sidecar Service - SalesTrainer Context

## Working with Copilot

**CRITICAL: Always verify before executing critical operations**

- Ask for user confirmation before running deployment commands
- Pause and ask for verification after each major step (tag, push, service restart)
- Show command output and ask "Does this look correct?" before proceeding
- Never assume success - wait for user to confirm each checkpoint
- Key moments to pause: before tagging, before pushing tags, before restarting service, after checking logs

**Better to be slow and safe than fast and broken!**

## Repository Overview
Python Flask/FastAPI service for audio processing. Downloads audio from S3, transcribes using OpenAI Whisper API, and posts structured results back to WordPress.

## Tech Stack
- **Python 3.x** with Flask/FastAPI
- **OpenAI Whisper API** for transcription
- **AWS S3** for audio file storage
- **systemd** for service management

## Service Management

### Dev/Prod Service Control
**Note:** Currently managing dev and prod versions of this service is being finalized. For now, there is one shared service.

```bash
# Check status
sudo systemctl status salestrainer-sidecar

# Start service
sudo systemctl start salestrainer-sidecar

# Stop service
sudo systemctl stop salestrainer-sidecar

# Restart service (after code changes)
sudo systemctl restart salestrainer-sidecar

# Enable on boot (already configured)
sudo systemctl enable salestrainer-sidecar
```

### View Logs

```bash
# Application log file
tail -f /var/www/salestrainer-dev/sidecar/sidecar.log

# systemd journal logs
sudo journalctl -u salestrainer-sidecar -f

# Recent errors
sudo journalctl -u salestrainer-sidecar -p err -n 50
```

### Service Configuration
- **Service File:** `/etc/systemd/system/salestrainer-sidecar.service`
- **User:** `ctingey`
- **Group:** `www-data`
- **Auto-restart:** Yes (10 second delay on failure)
- **Start on boot:** Enabled

## Deployment

### Current Process
1. Pull changes: `git pull origin main`
2. Install dependencies: `pip install -r requirements.txt`
3. Restart service: `sudo systemctl restart salestrainer-sidecar`
4. Check logs: `tail -f sidecar.log`

**Note:** No build step required (Python). Service picks up changes on restart.

## Move to Production

### Production Guardrails
- Deploy to production ONLY from prod repo checkouts using immutable tags (detached HEAD)
- Tag naming convention: `prod-YYYY-MM-DD-N` (increment N for multiple deploys same day)
- Production repo: `/var/www/salestrainer-prod/sidecar`
- Production service: `salestrainer-sidecar-prod.service`
- Production port: `127.0.0.1:9002` (dev uses different port)
- Environment file: `/var/www/salestrainer-prod/sidecar/.env`

### Separate Dev and Prod Services

**Dev Service:**
- Service: `salestrainer-sidecar` (or `salestrainer-sidecar-dev`)
- Repo: `/var/www/salestrainer-dev/sidecar`
- Config: Dev S3 bucket, dev WordPress callback URL
- Port: Different from prod

**Prod Service:**
- Service: `salestrainer-sidecar-prod.service`
- Repo: `/var/www/salestrainer-prod/sidecar`
- Config: Prod S3 bucket, `https://salestrainer.pro` callbacks
- Port: `127.0.0.1:9002`
- Environment: Uses `.venv` for isolated Python environment

### A) Dev: Merge → Tag → Push

```bash
# From dev repo
cd /var/www/salestrainer-dev/sidecar
git checkout main
git pull

# If merging a feature branch:
git merge feature/issue-XXX

# Test locally if possible
python server.py  # or use docker-compose

# Create tag
TAG="prod-$(date +%Y-%m-%d)-1"  # Increment -N if deploying multiple times today
git tag -a "$TAG" -m "Description of changes being deployed"

# Push tag
git push origin "$TAG"
```

### B) Prod: Fetch Tags → Deploy → Restart Service

```bash
# Switch to production user
su - salestrainer.pro_cht1976

# Go to prod repo
cd /var/www/salestrainer-prod/sidecar

# Fetch tags from GitHub
git fetch --tags

# Checkout the tag (detached HEAD)
git checkout prod-YYYY-MM-DD-N

# Activate virtual environment and install dependencies
source .venv/bin/activate
pip install -r requirements.txt
deactivate

# Restart production service
sudo systemctl restart salestrainer-sidecar-prod

# Verify service started
sudo systemctl status salestrainer-sidecar-prod
```

### C) Verification

```bash
# CRITICAL: Verify the correct tag is checked out
cd /var/www/salestrainer-prod/sidecar
git describe --tags
# Should show: prod-YYYY-MM-DD-N (the tag you just deployed)

# Check service is running
sudo systemctl status salestrainer-sidecar-prod

# Check logs for startup errors
sudo journalctl -u salestrainer-sidecar-prod -n 50
tail -f /var/www/salestrainer-prod/sidecar/sidecar.log

# Test health endpoint (if available)
curl http://127.0.0.1:9002/health

# Monitor for incoming requests
tail -f /var/www/salestrainer-prod/sidecar/sidecar.log | grep "POST /transcribe"
```

**Test End-to-End:**
1. Upload audio through production WordPress
2. Verify sidecar receives transcription request
3. Check Whisper API processes audio
4. Confirm callback to WordPress succeeds
5. Verify transcript appears in WordPress

### D) Rollback (If Needed)

```bash
# Go to prod repo
cd /var/www/salestrainer-prod/sidecar

# Find previous working tag
git tag --sort=-creatordate | head -10

# Checkout previous tag
git checkout prod-YYYY-MM-DD-N

# Reinstall dependencies if needed
source .venv/bin/activate
pip install -r requirements.txt
deactivate

# Restart service
sudo systemctl restart salestrainer-sidecar-prod

# Verify
sudo systemctl status salestrainer-sidecar-prod
tail -f /var/www/salestrainer-prod/sidecar/sidecar.log
```

### Production Environment Configuration

Ensure `/var/www/salestrainer-prod/sidecar/.env` contains:

```bash
# AWS S3 (production bucket)
AWS_REGION=us-east-1
S3_BUCKET=salestrainer-audio-prod
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...

# OpenAI
OPENAI_API_KEY=sk-...

# Security
SIDE_CAR_SECRET=...  # Must match wp-config.php ST_SIDECAR_SECRET

# WordPress Callback (PRODUCTION)
WP_SIDE_CAR_CALLBACK=https://salestrainer.pro/wp-json/salestrainer/v1/upload-complete
WP_SERVER_TOKEN=...  # Must match wp-config.php WP_SERVER_TOKEN

SEND_X_SERVER_TOKEN_HEADER=true
```

## Environment Variables

Required in service environment file or systemd unit:

```bash
# AWS S3 Configuration
AWS_REGION=us-east-1
S3_BUCKET=salestrainer-audio-dev  # or -prod
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...

# OpenAI Configuration
OPENAI_API_KEY=sk-...

# Security
SIDE_CAR_SECRET=...  # Shared secret for incoming requests

# WordPress Callback
WP_SIDE_CAR_CALLBACK=https://dev.salestrainer.pro/wp-json/salestrainer/v1/upload-complete
WP_SERVER_TOKEN=...  # Server-to-server auth token

# Optional
SEND_X_SERVER_TOKEN_HEADER=true  # Also send token in header (default: true)
```

## Authentication Flow

### Incoming Requests (WordPress → Sidecar)
- WordPress sends `SIDE_CAR_SECRET` in request
- Sidecar validates secret before processing

### Outgoing Callbacks (Sidecar → WordPress)
- **Primary:** `WP_SERVER_TOKEN` sent in JSON body (WAF-safe, recommended)
- **Secondary:** `X-Server-Token` header if `SEND_X_SERVER_TOKEN_HEADER=true`
- **Never uses:** `Authorization` header (gets stripped by WAF on /wp-json routes)

## How It Works

### Process Flow
1. WordPress uploads audio to S3
2. WordPress calls sidecar with S3 key
3. Sidecar downloads audio from S3
4. Sidecar transcribes with OpenAI Whisper API
5. Sidecar posts results to WordPress callback endpoint
6. WordPress processes transcript and stores in database

### Endpoints

**`POST /transcribe`** - Main transcription endpoint
- Receives S3 key from WordPress
- Downloads audio file
- Sends to Whisper API
- Posts results to WordPress callback

## Related Repositories

### wp-custom
- WordPress backend that triggers transcription
- Provides callback endpoint: `/wp-json/salestrainer/v1/upload-complete`
- Stores final transcripts and metadata
- User interface for viewing results

### convai-client
- React app capturing audio in browser
- Uploads recorded audio to WordPress
- Displays transcription results
- Frontend for conversation interface

## Data Flow

```
User speaks → convai-client (browser)
                    ↓
              WordPress API (wp-custom)
                    ↓
              Upload to S3
                    ↓
              Trigger Sidecar ← (this service)
                    ↓
              Download from S3
                    ↓
              OpenAI Whisper API
                    ↓
              Callback to WordPress API
                    ↓
              Store in database
                    ↓
              Display to user (convai-client)
```

## Development

### Local Development (Docker)

```bash
# Build and run with Docker Compose
docker-compose up --build

# Test endpoint locally
curl -X POST http://localhost:5001/transcribe \
  -H "Content-Type: application/json" \
  -d '{"s3_key": "test-audio.mp3"}'
```

### Testing Production Service

```bash
# Test from command line
./test-s3-flow.sh

# Check service health
curl http://localhost:5001/health

# View recent requests
grep "POST /transcribe" /var/www/salestrainer-dev/sidecar/sidecar.log | tail -20
```

## Key Files

- **`server.py`** - Main Flask/FastAPI application
- **`requirements.txt`** - Python dependencies
- **`salestrainer-sidecar.service`** - systemd service definition
- **`Dockerfile`** - For local Docker development
- **`test-s3-flow.sh`** - End-to-end test script

## Common Issues & Solutions

### Issue: Service Won't Start
```bash
# Check service status and logs
sudo systemctl status salestrainer-sidecar
sudo journalctl -u salestrainer-sidecar -n 50

# Common causes:
# - Missing environment variables
# - Python dependencies not installed
# - Port already in use
# - Permission issues on log file
```

### Issue: Transcription Fails
- Check S3 permissions (can sidecar download?)
- Verify OpenAI API key is valid and has credits
- Check audio file format (supported: mp3, wav, m4a, etc.)
- Review Whisper API error in logs

### Issue: WordPress Doesn't Receive Callback
- Verify `WP_SIDE_CAR_CALLBACK` URL is correct
- Check `WP_SERVER_TOKEN` matches WordPress configuration
- Look for network/firewall blocking callback
- Check WordPress error logs for rejected callbacks

### Issue: Port Conflicts
```bash
# Check if port is in use
sudo lsof -i :5001

# Kill process if needed
sudo kill -9 <PID>

# Or change port in service configuration
```

## Security Notes

- Never commit API keys or secrets to repository
- All secrets should be in systemd environment or `.env` file (not in git)
- Validate `SIDE_CAR_SECRET` on all incoming requests
- Use HTTPS for all WordPress callbacks in production
- Rotate secrets periodically
- S3 bucket should have restricted IAM policies

## Performance Considerations

- Whisper API calls can take 10-60 seconds depending on audio length
- Service handles requests asynchronously
- Failed transcriptions are retried (implement retry logic if not present)
- Monitor S3 bandwidth costs for large audio files

## Monitoring

### What to Monitor
- Service uptime: `sudo systemctl status salestrainer-sidecar`
- Error rate: `grep ERROR sidecar.log | wc -l`
- Transcription success rate
- API response times
- S3 download/upload success

### Logs to Watch
- Application: `/var/www/salestrainer-dev/sidecar/sidecar.log`
- System: `sudo journalctl -u salestrainer-sidecar`
- S3 errors in application log
- OpenAI API errors in application log

## Future Improvements

- [ ] Separate dev and prod service instances
- [ ] Implement retry logic for failed transcriptions
- [ ] Add health check endpoint monitoring
- [ ] Structured logging (JSON format)
- [ ] Metrics collection (Prometheus/Grafana)
- [ ] Queue system for high-volume processing
- [ ] Support for multiple audio formats
- [ ] Batch processing for efficiency
