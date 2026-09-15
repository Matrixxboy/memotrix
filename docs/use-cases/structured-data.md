---
title: "Structured data"
---

# Use case: structured data

Spreadsheets, JSON, YAML, SQL dumps, and XML are extracted into text sections, then chunked like any other document.

| Format | Extra | Notes |
|---|---|---|
| `.csv` | none | Stdlib `csv`; rows batched (8 per section) |
| `.xlsx` | `excel` | `openpyxl` + `pandas`; `.xls` is rejected |
| `.json` | none | Pretty-printed unless sniffed as FHIR / GeoJSON / chat / JSON-LD |
| `.yaml` / `.yml` | `yaml` | `PyYAML.safe_load` |
| `.sql` | none | Raw SQL text |
| `.xml` | none | Parsed to text |

```python
memory.add("exports.csv")
hits = memory.search("which donors are in region north?")
```

**Limitation.** Tabular Q&A is still retrieval over **text**, not a SQL engine. Precise aggregations (“sum of column X”) are not computed by Memotrix.

Demo samples: `DemoCodeLive/samples/data.csv`.
