"""Start the local demo; implementation lives in demo.server."""

from demo.server import create_app, main

__all__ = ["create_app", "main"]


if __name__ == "__main__":
    main()
