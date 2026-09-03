CLI is ready in `tests/context_cli.py` (also via `main.py`). It uses Memotrix ingest/retrieval and OpenRouter to build tighter context.

### Setup
Add to `.env`:
```env
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_MODEL=openai/gpt-4o-mini
```
Use a **vision-capable** model so PDF/DOCX/PPTX image OCR captions work.

Then: `pip install openai pymupdf python-docx python-pptx pillow`

### Commands
```bash
# Ingest docs (extracts embedded images + OCR-captions scanned pages)
python -m tests.context_cli ingest --path ./my_docs

# Raw retrieved chunks (no LLM synthesis)
python -m tests.context_cli query "What UI theme does the user prefer?"

# Retrieve + OpenRouter → better agent context
python -m tests.context_cli ask "What UI theme does the user prefer?"

# Interactive loop
python -m tests.context_cli chat

python main.py ask "What timezone should the agent assume?"

# Postgres backend
python -m tests.context_cli --backend postgres ingest --path ./my_docs --reset
```

### Image / scanned document behavior
- **Image-only PDF pages**: full-page render → vision OCR + description (searchable)
- **PDF/DOCX/PPTX with images between text**: images extracted to `.memotrix_media/<doc>/`, captioned, indexed
- Captions prioritize **text in the image**, then structure, then visual summary
