# Known Bugs and Caveats

## CSV Ingestion

CSV extraction used to index headers-only. It now embeds row batches (25 rows per batch, headers repeated) so cell values are retrieval-visible like Excel.

## Orphan Chunks

Re-ingesting an updated file now deletes prior chunks for that source_path before insert. Without that cleanup, stale memories kept surfacing after facts changed.

## Token Waste

Returning five full-file-expanded results blows the agent context window. Target two to three tightly scoped chunks with neighbor windows instead.
