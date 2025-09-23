# Streamlit App for the birthd.ai [バースデイ]

This directory contains the Streamlit UI that talks to the Vertex AI Agent Engine deployment. The app runs locally with `uv` and deploys to Cloud Run using buildpacks (via the `Procfile`).

## Prerequisites
- Python 3.10+ managed by [uv](https://github.com/astral-sh/uv)
- Google Cloud SDK (`gcloud` CLI) with an authenticated project
- `jq` (used by the `generate_env_yaml.sh` helper)
- A Vertex AI Agent Engine ID and a service-account JSON key with the required permissions

## 1. Install dependencies
```bash
cd streamlit-app
uv sync
```

## 2. Prepare environment variables
Use the helper script to convert your service-account JSON into the Cloud Run friendly `.env.yaml` file. Replace the paths as needed.
```bash
./generate_env_yaml.sh ../birthday-concierge-867d0e6b216a.json streamlit-app/.env.yaml
```
The script keeps existing values from your shell for `SERPAPI_API_KEY`, `VERTEX_AI_AGENT_ENGINE_ID`, `GOOGLE_CLOUD_PROJECT`, and `GOOGLE_CLOUD_LOCATION`, falling back to sensible placeholders or the JSON `project_id`.

### Environment variables expected
| Key | Description |
| --- | --- |
| `SERPAPI_API_KEY` | API key used by the shopping tool. |
| `VERTEX_AI_AGENT_ENGINE_ID` | Target Agent Engine resource name (`projects/.../locations/.../agentEngines/...`). |
| `GOOGLE_CLOUD_PROJECT` | Project that hosts Vertex AI resources. |
| `GOOGLE_CLOUD_LOCATION` | Region for Vertex AI (e.g. `us-central1`). |
| `VERTEXAI_SERVICE_ACCOUNT_*` | Fields derived from the service-account JSON. The private key is stored with literal `\n` escapes. |
| `VERTEXAI_SERVICE_ACCOUNT_UNIVERSE_DOMAIN` | Optional; included when present in the JSON. |

## 3. Run locally
```bash
uv run streamlit run main.py --server.address 0.0.0.0 --server.port 8501
```
Set the environment variables in your shell (or load them from `.env`) before running the command if you want to mirror production locally.

## 4. Deploy to Cloud Run
1. Export a `requirements.txt` for the buildpack:
   ```bash
   uv export --format requirements.txt > requirements.txt
   ```
2. Deploy:
   ```bash
   gcloud run deploy birthday-present-agent \
     --source . \
     --env-vars-file .env.yaml \
     --region us-central1 \
     --allow-unauthenticated
   ```
The `Procfile` in this directory ensures Cloud Run starts the Streamlit server with the correct `PORT` binding.

## 5. Post-deployment check
After deployment finishes, open the Cloud Run service URL. The Streamlit UI should load and display responses from the remote agent. Inspect Cloud Run logs if the page does not load or if the agent calls fail.

## Troubleshooting
- **`Failed to find attribute 'app' in 'main'`**: Cloud Run did not see the `Procfile`. Make sure it exists at the repo root for the deploy (`streamlit-app/Procfile`).
- **Permissions errors calling Vertex AI**: Verify all `VERTEXAI_SERVICE_ACCOUNT_*` vars are present and the private key retains the literal `\n` sequences.
- **Missing SERPAPI results**: Confirm `SERPAPI_API_KEY` is set and the associated API quota is sufficient.
