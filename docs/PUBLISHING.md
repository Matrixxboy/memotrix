# Publishing Memotrix to PyPI

This is the release checklist for the `memotrix` package (the `memory/` directory). Do not upload `.env` files, `__pycache__`, or private test fixtures.

Current version is declared in two places and **must stay in sync**:

- [`pyproject.toml`](../pyproject.toml) → `[project] version`
- [`src/memotrix/__init__.py`](../src/memotrix/__init__.py) → `__version__`

---

## 1. Preconditions

1. A [PyPI](https://pypi.org) account (and a [TestPyPI](https://test.pypi.org) account for the dry run).
2. Confirm the name is free:

   ```bash
   pip index versions memotrix
   ```

   If the name is taken, change `[project] name` in `pyproject.toml` before the first upload. You cannot rename a project after it exists on PyPI.

3. Homepage / source URL in `pyproject.toml` (`[project.urls] Homepage`) must be a real repository. Today it is `https://github.com/memotrix/memotrix` — replace it if that org/repo is not yours.
4. Set a real author (and email if you want) under `[project] authors`.
5. `LICENSE` (MIT) lives next to `pyproject.toml`. Keep `license = { file = "LICENSE" }`.
6. Python `>=3.10` (`requires-python`). Users must pass extras (`[memory]`, `[postgres]`, …); a bare `pip install memotrix` only installs `python-dotenv`.

---

## 2. What gets published

The sdist / wheel is built from `memory/` with `setuptools` (`packages.find` where `src/`). Included:

- `memotrix` package under `src/`
- `py.typed`
- `pypi.md` (as the PyPI long description / About panel)
- `LICENSE`

Not included (and should stay out): `.env`, test files, eval JSON, `__pycache__`, local media folders (`.memotrix_media`).

Users then install extras:

```bash
pip install memotrix[memory]
pip install memotrix[postgres,memory,extractors]
pip install memotrix[all]
```

---

## 3. Version bump

Before each release, increment **both**:

```toml
# pyproject.toml
version = "0.2.1"
```

```python
# src/memotrix/__init__.py
__version__ = "0.2.1"
```

Use [semver](https://semver.org): patch for bugfixes, minor for compatible API additions, major for breaking changes. You cannot reuse a version that was already uploaded to PyPI.

---

## 4. Build locally

From `memory/` (the directory that contains `pyproject.toml`):

```bash
python -m pip install --upgrade build twine
python -m build
python -m twine check dist/*
```

Expect `dist/memotrix-<version>.tar.gz` and `dist/memotrix-<version>-py3-none-any.whl`.

Install the wheel in a clean venv to sanity-check:

```bash
python -m venv .venv-release
# Windows: .venv-release\Scripts\activate
# Unix:    source .venv-release/bin/activate
pip install dist/memotrix-*.whl
python -c "import memotrix; print(memotrix.__version__)"
```

---

## 5. TestPyPI first (required dry run)

Create an API token on TestPyPI (Account → API tokens). Prefer Trusted Publishing (section 7) over long-lived tokens.

```bash
python -m twine upload --repository testpypi dist/*
```

Install from TestPyPI (dependencies that live only on real PyPI may need `--extra-index-url`):

```bash
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple memotrix[memory]
```

Smoke-test:

```python
from memotrix import Memory
from memotrix.embeddings import FakeEmbeddings
m = Memory(embeddings=FakeEmbeddings(dim=8), backend="memory")
m.add_text("hello", source_id="t")
print(m.search("hello"))
m.close()
```

---

## 6. Production PyPI

When TestPyPI looks right:

```bash
python -m twine upload dist/*
```

Do **not** pass `--skip-existing` to hide a failed upload of a new version. If a version is broken on PyPI, yank it in the web UI and ship a new version number.

After upload:

```bash
pip install memotrix[memory]
```

---

## 7. Trusted Publishing (recommended) vs API tokens

### API token (manual)

- Create a **scoped** token on pypi.org (project: `memotrix` once it exists, or account-wide for the first upload).
- Store it in a password manager. Never commit it. Use:

  ```bash
  python -m twine upload dist/*
  # prompts, or set TWINE_USERNAME=__token__ and TWINE_PASSWORD=<pypi-token>
  ```

### Trusted Publishing (GitHub Actions OIDC)

PyPI can accept uploads from GitHub without a stored token:

1. On pypi.org → your project → Publishing → add a GitHub publisher (`owner`, `repository`, `workflow` filename, optional `environment`).
2. Add a workflow at the **repo root** (this repo has no `memory/.github`; a root workflow that `cd memory` and builds is enough). Example sketch — do not invent an org name; fill in yours:

```yaml
# .github/workflows/publish-memotrix.yml
name: Publish memotrix
on:
  release:
    types: [published]
permissions:
  id-token: write
  contents: read
jobs:
  pypi:
    runs-on: ubuntu-latest
    environment: pypi
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install build
      - run: python -m build
        working-directory: memory
      - uses: pypa/gh-action-pypi-publish@release/v1
        with:
          packages-dir: memory/dist
```

3. First production upload can use a one-time token if the project does not exist yet; subsequent releases use OIDC.

Never put `TWINE_PASSWORD` in the workflow YAML or in git.

---

## 8. Release checklist

- [ ] Tests pass from `memory/`: `pytest tests/test_memory_sdk.py tests/test_config.py tests/test_hybrid_search.py tests/test_neighbor_expansion.py tests/test_domain_extractors.py -q`
- [ ] `version` in `pyproject.toml` equals `__version__`
- [ ] Homepage URL is correct; `LICENSE` present
- [ ] `python -m build` + `twine check dist/*`
- [ ] TestPyPI install + import smoke test
- [ ] Production upload (Trusted Publishing or token)
- [ ] Git tag `vX.Y.Z` on the commit you released
- [ ] Delete local `dist/` / `*.egg-info` if you do not want artifacts in the working tree

---

## 9. What not to publish

- `.env`, credentials, `OPENAI_API_KEY`
- `tests/filesForTests` and eval dumps (they are not in the package, keep it that way)
- `__pycache__`, `.pytest_cache`
- A second project name from this repo (`agent/` is not this package)
