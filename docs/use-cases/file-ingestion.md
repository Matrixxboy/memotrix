---
title: "File ingestion"
---

# Use case: file ingestion

`Memory.add(path)` is the single entry for files. Routing is `memotrix.filetypes.extract_file`.

## Wired extensions

| Group | Extensions |
|---|---|
| Documents | `.pdf` `.docx` `.pptx` `.txt` `.md` `.markdown` `.html` `.htm` `.epub` |
| Data | `.csv` `.xlsx` `.json` `.sql` `.xml` `.yaml` `.yml` |
| Images | `.png` `.jpg` `.jpeg` `.webp` `.bmp` `.gif` `.tiff` `.tif` `.svg` `.heic` |
| Video | `.mp4` `.mkv` `.avi` `.mov` `.webm` `.m4v` `.wmv` |
| Audio | `.mp3` `.wav` `.aac` `.m4a` `.ogg` `.flac` `.wma` |
| Code | `.py` `.js` `.ts` `.java` `.cpp` `.cc` `.cxx` `.go` `.rs` |
| Email | `.eml` `.mbox` |
| Geo | `.geojson` (and GeoJSON sniffed from `.json`) |
| Knowledge | `.ttl` `.nt` `.nq` `.rdf` `.owl` `.trig` `.jsonld` `.graphml` |
| Learning | `.scorm` or `.zip` **only if** it contains `imsmanifest.xml` |
| Other | `.jsonl` `.log` |

`.txt` that looks like WhatsApp (`[timestamp] Name: ...`) is routed to the chat extractor. `.json` is sniffed: FHIR, GeoJSON, chat, JSON-LD graph, else generic JSON.

## `add` kwargs

- `describe_images` — default from `IngestConfig.describe_images` (`True`). Set `False` to skip vision captions.
- `generate_srt` — audio/video extractors may write an SRT next to the source when `True`.

## Return value

```python
{
  "filename": "notes.txt",
  "path": "/abs/path/notes.txt",
  "chunks": 1,
  "inserted": 1,
  "merged": 0,
  "extracted_images": 0,
}
```

(`path` is the resolved filesystem path.)

## Extract only

`memory.extract(path)` returns `DocumentData` without indexing — useful to debug parsers.

## Unsupported

- Generic zip archives → `UnsupportedDocumentTypeError`
- `.xls` (BIFF) → convert to `.xlsx`

Demo: `DemoCodeLive/demos/10_file_types.py`.
