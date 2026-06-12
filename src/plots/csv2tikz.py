import subprocess
from pathlib import Path


def CompileLatex(tex_file_path: str) -> None:
    """Compile LaTeX file using platex, dvipdfmx, and dvisvgm."""
    tex_path = Path(tex_file_path).resolve()
    if not tex_path.exists():
        print(f"Error: {tex_path} does not exist.")
        return

    cwd = tex_path.parent
    base_name = tex_path.stem

    print(f"Compiling {tex_path.name} in {cwd}...")

    # 1. platex
    print("Running platex...")
    res = subprocess.run(
        ["platex", "-interaction=nonstopmode", f"{base_name}.tex"],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        print("platex failed!")
        print(res.stdout)
        print(res.stderr)
        return

    # 2. dvipdfmx
    print("Running dvipdfmx...")
    res = subprocess.run(
        ["dvipdfmx", f"{base_name}.dvi"],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        print("dvipdfmx failed!")
        print(res.stdout)
        print(res.stderr)
        return

    # 3. dvisvgm
    print("Running dvisvgm...")
    res = subprocess.run(
        ["dvisvgm", "--no-fonts", f"{base_name}.dvi"],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        print("dvisvgm failed!")
        print(res.stdout)
        print(res.stderr)
        return

    print("Compilation completed successfully!")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Compile LaTeX documents.")
    parser.add_argument(
        "tex_file",
        help="Path to the .tex file to compile.",
    )
    args = parser.parse_args()

    CompileLatex(args.tex_file)


if __name__ == "__main__":
    main()
