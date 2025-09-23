#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <service-account-json> [output-yaml]" >&2
  exit 1
fi

SERVICE_ACCOUNT_JSON=$1
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
OUTPUT_YAML=${2:-"$SCRIPT_DIR/.env.yaml"}

if ! command -v jq >/dev/null 2>&1; then
  echo "Error: jq is required but not installed." >&2
  exit 1
fi

if [[ ! -f $SERVICE_ACCOUNT_JSON ]]; then
  echo "Error: service account JSON not found at $SERVICE_ACCOUNT_JSON" >&2
  exit 1
fi

jq_filter='{
  type,
  project_id,
  private_key_id,
  private_key,
  client_email,
  client_id,
  auth_uri,
  token_uri,
  auth_provider_x509_cert_url,
  client_x509_cert_url,
  universe_domain
}'

service_account=$(jq "$jq_filter" "$SERVICE_ACCOUNT_JSON")

get_value() {
  local key=$1
  jq -r --arg key "$key" '.[$key]' <<<"$service_account"
}

quote_if_not_null() {
  local value=$1
  if [[ $value == null ]]; then
    echo ""
  else
    printf '"%s"' "$value"
  fi
}

project_id_value=$(get_value project_id)
if [[ $project_id_value == null ]]; then
  project_id_value=""
fi

serpapi_value=${SERPAPI_API_KEY:-replace-with-serpapi-key}
agent_engine_value=${VERTEX_AI_AGENT_ENGINE_ID:-projects/your-project/locations/your-location/agentEngines/your-agent-engine-id}
location_value=${GOOGLE_CLOUD_LOCATION:-us-central1}
project_line_value=${GOOGLE_CLOUD_PROJECT:-$project_id_value}

cat > "$OUTPUT_YAML" <<EOF2
# Environment variables for deploying the Streamlit app to Cloud Run.
SERPAPI_API_KEY: "$serpapi_value"
VERTEX_AI_AGENT_ENGINE_ID: "$agent_engine_value"
GOOGLE_CLOUD_PROJECT: "$project_line_value"
GOOGLE_CLOUD_LOCATION: "$location_value"

# Service account credentials (each field from the JSON key is mapped to an env var).
VERTEXAI_SERVICE_ACCOUNT_TYPE: $(quote_if_not_null "$(get_value type)")
VERTEXAI_SERVICE_ACCOUNT_PROJECT_ID: $(quote_if_not_null "$(get_value project_id)")
VERTEXAI_SERVICE_ACCOUNT_PRIVATE_KEY_ID: $(quote_if_not_null "$(get_value private_key_id)")
VERTEXAI_SERVICE_ACCOUNT_PRIVATE_KEY: $(jq -r '.private_key | @json' "$SERVICE_ACCOUNT_JSON")
VERTEXAI_SERVICE_ACCOUNT_CLIENT_EMAIL: $(quote_if_not_null "$(get_value client_email)")
VERTEXAI_SERVICE_ACCOUNT_CLIENT_ID: $(quote_if_not_null "$(get_value client_id)")
VERTEXAI_SERVICE_ACCOUNT_AUTH_URI: $(quote_if_not_null "$(get_value auth_uri)")
VERTEXAI_SERVICE_ACCOUNT_TOKEN_URI: $(quote_if_not_null "$(get_value token_uri)")
VERTEXAI_SERVICE_ACCOUNT_AUTH_PROVIDER_X509_CERT_URL: $(quote_if_not_null "$(get_value auth_provider_x509_cert_url)")
VERTEXAI_SERVICE_ACCOUNT_CLIENT_X509_CERT_URL: $(quote_if_not_null "$(get_value client_x509_cert_url)")
EOF2

universe_domain=$(get_value universe_domain)
if [[ $universe_domain != null && -n $universe_domain ]]; then
  printf 'VERTEXAI_SERVICE_ACCOUNT_UNIVERSE_DOMAIN: "%s"\n' "$universe_domain" >> "$OUTPUT_YAML"
else
  printf '# VERTEXAI_SERVICE_ACCOUNT_UNIVERSE_DOMAIN: "googleapis.com"\n' >> "$OUTPUT_YAML"
fi

chmod 600 "$OUTPUT_YAML"

echo "Wrote $OUTPUT_YAML"
