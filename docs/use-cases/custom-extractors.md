---
title: "Custom extractors"
---

# Use case: custom extractors

Two extension points exist in the shipped code.

## 1. Registry (per extension)

```python
from pathlib import Path
from memotrix.filetypes import register_extractor
from memotrix.utils.outputSturcture import build_document

def extract_myext(path: Path, **kwargs):
    text = path.read_text(encoding="utf-8")
    return build_document(path, text, extra_metadata={"file_type": "myext"})

register_extractor(".myext", extract_myext)
memory.add("data/file.myext")
```

A class with `.extract(path)` also works. Registry is checked **before** built-in routers, so you can override `.pdf`.

## 2. Constructor router

```python
memory = Memory(embeddings=embeddings, extract_file=my_full_router)
```

`my_full_router` replaces **all** routing for that `Memory` instance. It must accept `describe_images` and `generate_srt` keyword arguments (see `Memory.extract`).

Extractors must return `DocumentData` (`memotrix.utils.models`).

Demo: `DemoCodeLive/demos/07_custom_extractor.py`.
