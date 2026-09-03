import os
import sys
from pathlib import Path
from pprint import pprint

from dotenv import load_dotenv
load_dotenv()
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
for _p in (str(SRC), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from memotrix.services import DescriptionService
from memotrix.utils.ai_integration import GoogleProvider, configure_ai_provider


def main() -> None:
    if os.getenv("GOOGLE_API_KEY"):
        configure_ai_provider(GoogleProvider())

    service = DescriptionService()
    description = service.describe_image("./filesForTests/test.jpg")
    print("Image description:")
    pprint(description)

if __name__ == "__main__":
    main()