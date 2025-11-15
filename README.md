```markdown
salestrainer-sidecar

FastAPI sidecar that downloads audio from S3, transcribes with OpenAI Whisper API,
and posts structured results back to WordPress.

Env vars:
- AWS_REGION, S3_BUCKET, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
- OPENAI_API_KEY
- SIDE_CAR_SECRET
- WP_SIDE_CAR_CALLBACK (e.g., https://dev.salestrainer.pro/wp-json/salestrainer/v1/upload-complete)
- WP_SERVER_TOKEN (optional)

Start:
docker-compose up --build
```