# Deploy Procedure

## Local Postgres

Start the database with docker compose up -d. Credentials come from the .env file: POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB.

## Smoke Test

After containers are healthy, run the integration ingest against a small fixture folder, then issue a known query and confirm hybrid search returns the expected chunk.

## Production Notes

Never force-push to main. Do not commit .env. Prefer creating a PR with gh pr create after pushing the feature branch with -u.
