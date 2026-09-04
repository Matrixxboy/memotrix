# How to Add Custom File Extractors

Memotrix supports over 20 file formats out of the box, but you may need to parse a proprietary format (like `.myext` or an internal XML structure). 

With the **Extractor Plugin Registry**, you can register custom extractors dynamically without forking the SDK.

## 1. Create a Custom Extractor

All extractors must return a `DocumentData` object from the `memotrix.utils.models` module. 
We provide a helper function `build_document` to make this easy.

```python
from pathlib import Path
from memotrix.utils.outputSturcture import build_document

class MyCustomExtractor:
    def extract(self, path: Path, **kwargs):
        # 1. Read your custom file
        content = path.read_text(encoding="utf-8")
        
        # 2. Parse the content
        parsed_text = f"Parsed custom file: {content.strip()}"
        
        # 3. Build and return the DocumentData
        return build_document(
            path=path,
            text=parsed_text,
            extra_metadata={"file_type": "myext", "parser": "MyCustomExtractor"}
        )
```

You can also use a simple function instead of a class:

```python
def extract_myext(path: Path, **kwargs):
    text = path.read_text()
    return build_document(path, text, extra_metadata={"file_type": "myext"})
```

## 2. Register the Extractor

Use the `register_extractor` function from `memotrix.filetypes` to map your custom extension to your extractor.

```python
from memotrix.filetypes import register_extractor

# Register the class instance
register_extractor(".myext", MyCustomExtractor())

# Or register the function
# register_extractor(".myext", extract_myext)
```

## 3. Use Memotrix as Normal

Once registered, Memotrix's `Memory.add()` and `Memory.extract()` functions will automatically route files with your custom extension to your logic.

```python
from memotrix import Memory
from memotrix.embeddings import FakeEmbeddings

memory = Memory(embeddings=FakeEmbeddings(dim=8), backend="memory")

# The custom extractor will be invoked automatically!
memory.add("data/custom_file.myext")
```

## Fallbacks and Overrides

You can override built-in extractors by registering a custom one for an existing extension.

```python
# Override the default PDF extractor
register_extractor(".pdf", MyBetterPDFExtractor())
```
