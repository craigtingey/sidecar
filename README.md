```markdown
salestrainer-sidecar

FastAPI sidecar that downloads audio from S3, transcribes with OpenAI Whisper API,
and posts structured results back to WordPress.

Env vars:
- AWS_REGION, S3_BUCKET, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
- OPENAI_API_KEY
- SIDE_CAR_SECRET (shared secret for authenticating incoming requests)
- WP_SIDE_CAR_CALLBACK (e.g., https://dev.salestrainer.pro/wp-json/salestrainer/v1/upload-complete)
- WP_SERVER_TOKEN (server-to-server auth token for WordPress callback)
- SEND_X_SERVER_TOKEN_HEADER (optional, default: true) - also send token as X-Server-Token header

Authentication:
- Always sends WP_SERVER_TOKEN in JSON body (most WAF-safe, recommended)
- Optionally sends X-Server-Token header if SEND_X_SERVER_TOKEN_HEADER=true (redundant but safe)
- Does NOT use Authorization header (gets stripped by WAF on /wp-json routes)

Start:
docker-compose up --build

Test endpoint locally:
curl -X POST http://localhost:9001/process-s3 \
  -H "Content-Type: application/json" \
  -d '{"object_key":"test.webm","session_id":"test","secret":"<SIDE_CAR_SECRET>"}'

Test WordPress callback manually:
curl -v -X POST "https://dev.salestrainer.pro/wp-json/salestrainer/v1/upload-complete" \
  -H "Content-Type: application/json" \
  -d '{"server_token":"<WP_SERVER_TOKEN>","object_key":"test.webm","session_id":"test","scenario_id":0,"size_bytes":100,"transcript":"test","ok":true}'
```
