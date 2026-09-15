"""Run DemoCodeLive scripts that do not need Postgres, env, or HuggingFace downloads."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEMOS = ROOT / "demos"

ALWAYS = [
    "01_hello_memory.py",
    "02_ingest_files.py",
    "03_memory_types.py",
    "04_sessions_and_filters.py",
    "05_search_expansion.py",
    "06_extract_only.py",
    "07_custom_extractor.py",
    "08_list_and_delete.py",
    "09_rag_loop.py",
    "10_file_types.py",
    "11_postgres.py",
    "12_from_env.py",
]


def main() -> int:
    import memotrix

    print("memotrix file:", memotrix.__file__)
    if "site-packages" not in str(memotrix.__file__).replace("\\", "/"):
        print(
            "WARNING: memotrix is not loading from site-packages. "
            "Activate DemoCodeLive/.venv after pip install memotrix[memory]."
        )

    failed = []
    for name in ALWAYS:
        path = DEMOS / name
        print(f"\n===== {name} =====")
        try:
            runpy.run_path(str(path), run_name="__main__")
        except Exception as exc:  # noqa: BLE001 — surface demo failures
            print(f"FAILED {name}: {exc}")
            failed.append(name)

    if failed:
        print("failed:", failed)
        return 1
    print("\nAll listed demos finished.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
