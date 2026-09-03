# Procedural — How to Add a File Type

## Steps

1. Create an extractor under src/filetypes for the format.
2. Normalize output through build_document() into DocumentData.
3. Ensure sections, tables, and images land in the shared schema.
4. Ingest via IngestionPipeline so chunks hit dense and sparse indexes.

## Validation

Add at least one fixture file and one eval query that must retrieve a distinctive phrase from the new format. If Recall@5 drops on the shared harness, investigate before merging.
