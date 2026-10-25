"""Launch explicitly on loopback; existing boxing-motion-ai CLI is unchanged."""

import argparse
from pathlib import Path

import uvicorn

from .app import create_app


def main():
    parser = argparse.ArgumentParser(
        description="Local Boxing Motion AI annotation-review dashboard"
    )
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Trusted project root")
    args = parser.parse_args()
    uvicorn.run(create_app(args.root), host="127.0.0.1", port=8000, access_log=False)


if __name__ == "__main__":
    main()
