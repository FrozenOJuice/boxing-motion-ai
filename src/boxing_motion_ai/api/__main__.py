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
    parser.add_argument("--port", type=int, default=8000, choices=(8000, 8001))
    args = parser.parse_args()
    uvicorn.run(create_app(args.root), host="127.0.0.1", port=args.port, access_log=False)


if __name__ == "__main__":
    main()
