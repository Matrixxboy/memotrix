---
title: "Optional extras"
---

# Optional extras

Copied from `pyproject.toml` extras. Install only what you use.

```bash
pip install memotrix
pip install "memotrix[memory]"
pip install "memotrix[postgres,memory,extractors]"
pip install "memotrix[all]"
```

See the table in [Installation](../installation.md).

`extractors` is an alias of `all-extractors`. Granular extras: `pdf`, `docx`, `pptx`, `excel`, `images`, `html`, `knowledge`, `yaml`.
