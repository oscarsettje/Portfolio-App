"""
Portfolio Tracker - Main Entry Point
=====================================
Run this file to start the portfolio tracker CLI.
Usage: python main.py
"""

from tracker.cli import CLI


def main():
    cli = CLI()
    cli.run()


if __name__ == "__main__":
    main()
