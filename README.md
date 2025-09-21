# 🎁 Birthday Present Agent

Google Agent Development Kit (ADK) を使って実装した誕生日プレゼント提案エージェントです。Gemini をコアモデルに据え、Grok で X(旧Twitter) の公開プロフィールを要約し、SerpApi で Google Shopping の商品情報を取得します。Streamlit UI から対話的に利用できます。

## 主要コンポーネント
- `birthday_present_agent/agent.py` — Gemini 2.5 Flash を使ったルートエージェント定義。
- `birthday_present_agent/prompt.py` — 会話フローとツール利用ポリシーをまとめたシステムプロンプト。
- `birthday_present_agent/tools/` — Grok/SerpApi を呼び出すカスタムツール群。
- `streamlit_app.py` — ADK の `InMemoryRunner` を用いたチャット UI。初回アクセス時にエージェントが会話を開始します。

## セットアップ (uv を利用)
1. [uv](https://github.com/astral-sh/uv) をインストールします。
   ```bash
   pip install uv
   ```
2. 依存関係を同期します (プロジェクト直下で実行)。
   ```bash
   uv sync
   ```
   ※ 既存の仮想環境がある場合は `UV_ACTIVE=1` を設定するか、`uv venv` で新規に作成してから `uv pip install -r` する方法でも構いません。
3. `.env.example` を `.env` にコピーし、以下の値を設定します。
   - `GOOGLE_API_KEY` — Gemini API キー
   - `GROK_API_KEY` — xAI Grok API キー
   - `SERPAPI_API_KEY` — SerpApi キー
   そのほか地域やモデルを変更したい場合はコメントアウトされた環境変数を必要に応じて有効化してください。

## 使い方
### Streamlit UI
```bash
uv run streamlit run streamlit_app.py
```
ブラウザで表示されるチャットから、贈る相手の職業や年齢、X プロフィールなどを伝えると候補が提示されます。興味のあるプレゼントを選ぶと詳細説明が返ってきます。

### CLI (オプション)
ADK の CLI から直接会話することもできます。
```bash
uv run adk run birthday_present_agent
```

## 実装メモ
- Grok への問い合わせは JSON 形式で人物像を返すよう指示しています。取得できなかった場合は人間に `.env` の設定を促すようエージェントに伝えています。
- SerpApi の検索結果は最大 10 件までサマリ化し、価格・画像 URL・詳細 API のエンドポイントを返します。詳細説明時には `fetch_product_details` ツールで追加情報を取得できます。
- Streamlit 側では ADK の `InMemoryRunner` とセッションを共有し、ツール呼び出しログを展開表示できるようにしています。

## 今後の拡張アイデア
1. SerpApi のレスポンスをキャッシュし、同じ商品へのアクセスを高速化する。
2. 予算帯やカテゴリを UI で直接指定できる補助フォームを追加する。
3. 選んだギフトを Google カレンダーやリマインダーに登録する追加ツールを実装する。
