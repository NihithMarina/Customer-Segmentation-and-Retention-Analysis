from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "screenshots" / "terminal_outputs"


def render_text_to_png(txt_path: Path, png_path: Path, title: str) -> None:
    lines = txt_path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        lines = ["(no output)"]

    # Keep image readable while preserving terminal-like layout.
    n_lines = max(len(lines), 6)
    fig_h = min(0.45 * n_lines + 1.8, 12)

    fig, ax = plt.subplots(figsize=(13, fig_h), dpi=160)
    fig.patch.set_facecolor("#101418")
    ax.set_facecolor("#101418")
    ax.axis("off")

    ax.text(
        0.02,
        0.96,
        title,
        transform=ax.transAxes,
        fontsize=14,
        color="#9be9a8",
        fontfamily="DejaVu Sans Mono",
        va="top",
        weight="bold",
    )

    body = "\n".join(lines)
    ax.text(
        0.02,
        0.90,
        body,
        transform=ax.transAxes,
        fontsize=11.5,
        color="#e6edf3",
        fontfamily="DejaVu Sans Mono",
        va="top",
    )

    plt.tight_layout()
    plt.savefig(png_path, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    render_text_to_png(
        OUT_DIR / "model_testing_terminal.txt",
        OUT_DIR / "model_testing_terminal.png",
        "Terminal Output - Model Testing",
    )

    render_text_to_png(
        OUT_DIR / "input_validation_terminal.txt",
        OUT_DIR / "input_validation_terminal.png",
        "Terminal Output - Input Validation",
    )

    print("Created terminal output images in:")
    print(OUT_DIR)


if __name__ == "__main__":
    main()
