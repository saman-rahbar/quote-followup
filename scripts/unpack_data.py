"""Put the supplied CSV files in data/, straight from data.zip. Does nothing if they are already there.

    python scripts/unpack_data.py ../data.zip
"""

from __future__ import annotations

import os
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    archive, target = Path(sys.argv[1]), ROOT / "data"
    if (target / "quotes.csv").exists():
        print("data/ already has the CSV files, leaving it alone")
        return
    target.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as packed:
        for member in packed.namelist():
            # Take only the CSV files, by file name alone, so nothing can land outside data/.
            if member.endswith(".csv") and "__MACOSX" not in member:
                final = target / Path(member).name
                # Write to a temporary name, then rename, so another process never sees half a file.
                partial = final.with_suffix(f".partial-{os.getpid()}")
                partial.write_bytes(packed.read(member))
                os.replace(partial, final)
    print(f"unpacked {len(list(target.glob('*.csv')))} CSV files into data/")


if __name__ == "__main__":
    main()
