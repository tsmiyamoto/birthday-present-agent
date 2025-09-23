# 🎁 birthd.ai [バースデイ]

<div align="center">
  <img src="public/birthd.ai logo.png" alt="birthd.ai logo" width="600">
</div>

Vertex AI Agent Engine と Google Agent Development Kit (ADK) を組み合わせて、誕生日プレゼントを提案するAIエージェントを構築です。3 つのモジュールを中心に運用しています。

## 主な構成要素
- `birthday_present_agent/` — Gemini 2.5 Pro を用いたルートエージェント、ツール実装、システムプロンプトを管理。
- `streamlit_app.py` — ADK の `InMemoryRunner` でエージェントを起動し、ローカル開発用の Streamlit チャット UI を提供。
- `streamlit-app/` — Vertex AI Agent Engine にデプロイ済みのエージェントと通信する本番想定の Streamlit UI（Cloud Run などにデプロイ可能）。

## セットアップ
1. [uv](https://github.com/astral-sh/uv) をインストールします。
   ```bash
   pip install uv
   ```
2. ルートディレクトリで依存関係を同期します。
   ```bash
   uv sync
   ```
3. `.env.example` を `.env` にコピーし、必要なキーを設定します。
   - `GOOGLE_API_KEY` — Gemini API キー（ADK からの呼び出しに使用）
   - `SERPAPI_API_KEY` — Google Shopping 検索用の SerpApi キー
   - `GROK_API_KEY` — （任意）xAI Grok を直接叩く際に利用。現在は手動確認を促す運用です。
   - `GOOGLE_CLOUD_PROJECT` / `GOOGLE_CLOUD_LOCATION` / `GOOGLE_CLOUD_STORAGE_BUCKET` / `GOOGLE_APPLICATION_CREDENTIALS` — Agent Engine へデプロイ・接続する場合に必須
   - `VERTEX_AI_AGENT_ENGINE_ID` — デプロイ済み Agent Engine のリソース名

Python 3.11.10 以上が必要です。`uv` は自動で仮想環境を管理します。

## ローカル開発（ADK + Streamlit）
1. `.env` で API キーを読み込める状態にします（`uv run` は自動で `.env` を反映）。
2. ローカルで Streamlit を起動します。
   ```bash
   uv run streamlit run streamlit_app.py
   ```
3. ブラウザで表示されるチャット UI から、贈る相手の情報（職業・年齢・X プロフィール URL など）を入力すると候補が提示されます。気になる候補を選択すると、`fetch_product_details` ツールが追加情報を取得して詳細を返します。

## Vertex AI Agent Engine 連携 UI（`streamlit-app/`）
Agent Engine にデプロイ済みのエージェントへアクセスする UI です。Cloud Run などにデプロイする前提で、サービスアカウント情報を環境変数に展開します。

1. ディレクトリを移動して依存関係を同期します。
   ```bash
   cd streamlit-app
   uv sync
   ```
2. `generate_env_yaml.sh` を使うと、サービスアカウント JSON から Cloud Run 用 `.env.yaml` を生成できます。
   ```bash
   ./generate_env_yaml.sh ../<service-account>.json .env.yaml
   ```
3. ローカルで動作確認する場合は、必要な環境変数を読み込んだ上で実行します。
   ```bash
   uv run streamlit run main.py --server.address 0.0.0.0 --server.port 8501
   ```
4. Cloud Run へデプロイする際は `uv export --format requirements.txt > requirements.txt` を実行し、`gcloud run deploy` で `Procfile` に従って起動します。

期待する主な環境変数（`streamlit_app.py` と共通のものを含む）：
- `SERPAPI_API_KEY`
- `VERTEX_AI_AGENT_ENGINE_ID`
- `GOOGLE_CLOUD_PROJECT`
- `GOOGLE_CLOUD_LOCATION`
- `VERTEXAI_SERVICE_ACCOUNT_*`（サービスアカウント JSON 由来の項目。`private_key` は `\n` を含む形で保持します）
- `VERTEXAI_SERVICE_ACCOUNT_UNIVERSE_DOMAIN`（存在する場合）

## エージェントの仕組み
- ルートエージェント（`birthday_present_agent/agent.py`）は Gemini 2.5 Pro を使用し、ショッピング検索・商品の詳細取得・X プロフィール調査（現在は手動ガイダンス）の 3 つのツールを動的に選択します。
- システムプロンプト（`birthday_present_agent/prompt.py`）では、ペルソナヒアリングからプレゼント候補提示までの会話フローと、各ツールの利用条件を明示。
- `streamlit_app.py` / `streamlit-app/main.py` はツール呼び出しログを展開表示し、候補のカード表示や詳細パネルなどリッチな UI を提供します。

## よくある確認ポイント
- SerpApi のクォータを使い切ると候補が取得できません。`SERPAPI_API_KEY` を再確認してください。
- Vertex AI 関連の環境変数が揃っていない場合、`streamlit-app/main.py` が起動時に例外を投げます。`VERTEX_AI_AGENT_ENGINE_ID`、`GOOGLE_CLOUD_PROJECT`、`GOOGLE_CLOUD_LOCATION` とサービスアカウント情報を再設定してください。
- Grok の自動連携は未整備のため、X プロフィール解析は UI 経由で手動入力してもらう想定です。

