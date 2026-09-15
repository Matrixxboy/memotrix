---
title: "GitHub Pages"
---

# GitHub Pages (documentation site)

The MkDocs site is built by [`.github/workflows/docs.yml`](../../.github/workflows/docs.yml) and published to the `gh-pages` branch.

Live URL (after Pages is pointed at that branch): [https://matrixxboy.github.io/memotrix/](https://matrixxboy.github.io/memotrix/)

## Why `actions/deploy-pages` returned 404

```text
Failed to create deployment (status: 404)
Ensure GitHub Pages has been enabled:
https://github.com/Matrixxboy/memotrix/settings/pages
```

`actions/deploy-pages@v4` talks to the Pages **deployment API**. That API returns **404** until Pages is turned on for the repo. The MkDocs **build** can succeed (you already got an artifact) while **deploy** still fails.

The workflow now **pushes `site/` to `gh-pages`**. That does not use the Pages deployment API.

## One-time GitHub setting

1. Wait for **Deploy docs** to finish green (it will create/update branch `gh-pages`).
2. Open [Settings → Pages](https://github.com/Matrixxboy/memotrix/settings/pages).
3. **Build and deployment → Source:** Deploy from a branch.
4. **Branch:** `gh-pages` / folder `/ (root)`.
5. Save.

GitHub will serve [https://matrixxboy.github.io/memotrix/](https://matrixxboy.github.io/memotrix/) after a minute or two.

If you prefer **Source: GitHub Actions** instead of a branch, you must enable that source **before** using `actions/deploy-pages`. This repo’s workflow uses the branch method so the first push does not 404.

## Local preview

```bash
pip install -r requirements-docs.txt
mkdocs serve
```

Open http://127.0.0.1:8000
