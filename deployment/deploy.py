"""Utility to deploy the birthday present agent to Vertex AI Agent Engine."""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
import vertexai
from vertexai import agent_engines
from vertexai.preview.reasoning_engines import AdkApp

from birthday_present_agent import root_agent

ROOT_DIR = Path(__file__).resolve().parents[1]
DIST_DIR = ROOT_DIR / "dist"


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Environment variable '{name}' is required.")
    return value


def _resolve_staging_bucket(raw: str) -> str:
    return raw if raw.startswith("gs://") else f"gs://{raw}"


def _find_wheel(dist_dir: Path) -> Path:
    wheels = sorted(dist_dir.glob("birthday_present_agent-*.whl"))
    if not wheels:
        raise FileNotFoundError(
            "No wheel found in 'dist'. Build the package first "
            "(e.g. `uv build` or `python -m build`)."
        )
    return wheels[-1]


def _wheel_requirement_path(wheel_path: Path) -> str:
    try:
        rel = wheel_path.relative_to(Path.cwd())
        rel_str = rel.as_posix()
        return f"./{rel_str}" if not rel_str.startswith(".") else rel_str
    except ValueError:
        # Fallback to absolute path if deployment is run from outside the repo.
        return str(wheel_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dist-dir",
        type=Path,
        default=DIST_DIR,
        help="Directory containing the built wheel (defaults to ./dist).",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip sending a test message after deployment.",
    )
    parser.add_argument(
        "--test-message",
        default="Hi!",
        help="Message to send for post-deployment smoke test.",
    )
    parser.add_argument(
        "--user-id",
        default="local-user",
        help="User id to use for the optional smoke test session.",
    )
    args = parser.parse_args()

    load_dotenv()

    project = _require_env("GOOGLE_CLOUD_PROJECT")
    location = _require_env("GOOGLE_CLOUD_LOCATION")
    bucket = _resolve_staging_bucket(_require_env("GOOGLE_CLOUD_STORAGE_BUCKET"))

    try:
        application_credentials = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
    except KeyError:
        application_credentials = None
    if application_credentials:
        print(f"Using service account credentials at: {application_credentials}")

    print(f"Deploying to project={project}, location={location}, staging_bucket={bucket}")

    vertexai.init(project=project, location=location, staging_bucket=bucket)

    app = AdkApp(agent=root_agent, enable_tracing=True)

    wheel_path = _find_wheel(args.dist_dir)
    requirement_path = _wheel_requirement_path(wheel_path)
    print(f"Uploading wheel: {wheel_path}")
    print(f"Requirement path passed to Agent Engine: {requirement_path}")

    remote_app = agent_engines.create(
        app,
        requirements=[requirement_path],
        extra_packages=[requirement_path],
    )
    print("Agent deployment finished.")

    if args.skip_test:
        return

    print("Running smoke test conversation...")
    session = remote_app.create_session(user_id=args.user_id)
    for event in remote_app.stream_query(
        user_id=args.user_id,
        session_id=session["id"],
        message=args.test_message,
    ):
        print(event)
    print("Smoke test complete.")


if __name__ == "__main__":
    main()
