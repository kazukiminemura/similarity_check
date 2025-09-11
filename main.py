from similarity_check.cli import main as cli_main


if __name__ == "__main__":
    # Delegate to the CLI entry so you can run from repo root:
    #   python main.py --target <video> --candidates <dir>
    cli_main()

