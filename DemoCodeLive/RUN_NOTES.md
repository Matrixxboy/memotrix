---
title: "Demo run notes"
---

# DemoCodeLive run notes

**Date:** 2026-09-15  
**Host:** Windows 10 (win32 10.0.26200)  
**Goal:** `pip install memotrix` then run `DemoCodeLive` against **site-packages**, not repo `src/`.

## Commands

```text
py -3.11 -m venv --clear .venv
.venv/Scripts/python -m pip install memotrix==0.2.0
.venv/Scripts/python -c "import memotrix; print(memotrix.__file__); print(memotrix.__version__)"
.venv/Scripts/python -m pip install pillow numpy rank-bm25
.venv/Scripts/python demos/01_hello_memory.py
.venv/Scripts/python -m pip install "memotrix[local]==0.2.0"
.venv/Scripts/python -m pip install "memotrix[memory]"
```

## Results (this machine)

| Step | Result |
|---|---|
| `pip install memotrix==0.2.0` | Success. Wheel `memotrix-0.2.0-py3-none-any.whl` (94 kB) |
| Import path | `DemoCodeLive/.venv/Lib/site-packages/memotrix/__init__.py` |
| `memotrix.__version__` | `0.1.0` (mismatch vs distribution 0.2.0) |
| `importlib.metadata.version("memotrix")` | `0.2.0` |
| `pip install "memotrix[local]"` / `[memory]` | **Failed** building `hnswlib` — `Microsoft Visual C++ 14.0 or greater is required` |
| `demos/01_hello_memory.py` after bare install | `ModuleNotFoundError: PIL` |
| Same after `pillow` | `ModuleNotFoundError: hnswlib` |

Python 3.14 and 3.11 both failed the hnswlib compile. `pip download hnswlib --only-binary=:all:` found **no** Windows wheels.

Demos **01–13 are complete in source**. They will run after `hnswlib` installs (Linux/macOS wheels, or Windows + MSVC).

## Not fabricated

No search latency, RPS, or RAG quality numbers were collected because `Memory()` could not be constructed in this environment.
