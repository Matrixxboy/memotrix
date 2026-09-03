import sys
from pathlib import Path
from pprint import pprint

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
for _p in (str(SRC), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from memotrix.filetypes.structuredData import StructuredDataExtractor


def main():
    file_path = PROJECT_ROOT / "tests" / "filesForTests" / "SignalHire_exports.csv"

    if not file_path.exists():
        raise FileNotFoundError(f"Test file not found: {file_path}")

    document = StructuredDataExtractor.extract(file_path)

    pprint(document)


if __name__ == "__main__":
    main()