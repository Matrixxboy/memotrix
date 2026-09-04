import pytest
from pathlib import Path
from memotrix.filetypes.registry import ExtractorRegistry, register_extractor
from memotrix.filetypes import extract_file

class DummyExtractor:
    def extract(self, path: Path, **kwargs):
        return {"extracted": True, "path": str(path), "type": "dummy_class"}

def dummy_extractor_func(path: Path, **kwargs):
    return {"extracted": True, "path": str(path), "type": "dummy_func"}

def test_registry_class_extractor(tmp_path):
    test_file = tmp_path / "test.dummy1"
    test_file.touch()

    register_extractor(".dummy1", DummyExtractor())
    assert ".dummy1" in ExtractorRegistry.supported_extensions()
    
    result = extract_file(test_file)
    assert result["extracted"] is True
    assert result["type"] == "dummy_class"

def test_registry_func_extractor(tmp_path):
    test_file = tmp_path / "test.dummy2"
    test_file.touch()

    register_extractor(".dummy2", dummy_extractor_func)
    assert ".dummy2" in ExtractorRegistry.supported_extensions()
    
    result = extract_file(test_file)
    assert result["extracted"] is True
    assert result["type"] == "dummy_func"
