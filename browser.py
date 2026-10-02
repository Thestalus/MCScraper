#!/usr/bin/env python3
"""
browser.py — Flask story browser entry point.

Usage:
  .venv/bin/python3 browser.py
  .venv/bin/python3 browser.py --index ./output/stories_index.json --port 5003

Then open:  http://localhost:5003
"""
from __future__ import annotations
import argparse
import sys

from app_factory import AppState, create_app
from services.data_store import load_data


def main():
    parser = argparse.ArgumentParser(description="MC Archive local browser")
    parser.add_argument("--index", default="./output/stories_index.json",
                        help="Path to stories_index.json (default: ./output/stories_index.json)")
    parser.add_argument("--port",  type=int, default=5000,
                        help="Port to listen on (default: 5000)")
    parser.add_argument("--host",  default="127.0.0.1",
                        help="Host to bind (default: 127.0.0.1)")
    args = parser.parse_args()

    state = AppState(index_path=args.index)
    load_data(state)

    app = create_app(state)

    print(f"\n  MC Archive Browser")
    print(f"  Open: http://{args.host}:{args.port}\n")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
