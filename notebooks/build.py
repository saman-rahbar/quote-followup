"""Turn the notebook sources in notebooks/src into executed .ipynb files.

The sources are plain Python in the "percent" format (`# %%` starts a code cell, `# %% [markdown]`
a text cell), so they diff cleanly and can be read without Jupyter. Each build runs every cell top
to bottom in a fresh kernel, so a committed notebook always matches the code and data.

    python notebooks/build.py            # all of them
    python notebooks/build.py 03         # only those whose name starts with 03
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent


def to_notebook(source: str) -> nbformat.NotebookNode:
    cells = []
    for block in source.split("\n# %%")[1:]:
        header, _, body = block.partition("\n")
        body = body.strip("\n")
        if header.strip() == "[markdown]":
            text = "\n".join(line[2:] if line.startswith("# ") else line.lstrip("#") for line in body.splitlines())
            cells.append(nbformat.v4.new_markdown_cell(text))
        else:
            cells.append(nbformat.v4.new_code_cell(body))
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return notebook


def main() -> None:
    wanted = sys.argv[1] if len(sys.argv) > 1 else ""
    for source in sorted((HERE / "src").glob(f"{wanted}*.py")):
        notebook = to_notebook("\n" + source.read_text())
        NotebookClient(notebook, timeout=1800, kernel_name="python3",
                       resources={"metadata": {"path": str(HERE)}}).execute()
        target = HERE / f"{source.stem}.ipynb"
        nbformat.write(notebook, target)
        print(f"built {target.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()
