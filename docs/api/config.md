---
title: "Config API"
---

# Config API

```python
from memotrix import MemoryConfig
from memotrix.config import (
    ChunkingConfig,
    RetrievalConfig,
    IngestConfig,
    resolve_database_url,
    require_env,
)
```

See [Configuration](../configuration.md) for defaults and environment variables.

`resolve_database_url(connection=None)` uses `connection`, else `DATABASE_URL`, else discrete `DATABASE_*` / `POSTGRES_*` parts. It never invents a password.
