import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

from memotrix.filetypes.document.base import BaseExtractor
from memotrix.filetypes.document.html import extract_text_from_html
from memotrix.utils.outputSturcture import build_document
from memotrix.utils.exceptions import DocumentExtractionError

class ScormExtractor(BaseExtractor):
    supported_extensions = (".scorm",)

    def extract(self, path: Path):
        try:
            with zipfile.ZipFile(path, 'r') as zf:
                # Check for manifest
                if 'imsmanifest.xml' not in zf.namelist():
                    raise DocumentExtractionError(f"Missing imsmanifest.xml in {path.name}. Is this a valid SCORM package?")
                
                # Try to extract a title from the manifest
                manifest_content = zf.read('imsmanifest.xml')
                title = path.name
                try:
                    root = ET.fromstring(manifest_content)
                    # Use a generic namespace search to find the organizations/organization/title
                    for elem in root.iter():
                        if elem.tag.endswith('organization'):
                            for child in elem:
                                if child.tag.endswith('title') and child.text:
                                    title = child.text.strip()
                                    break
                            break
                except ET.ParseError:
                    pass  # Title extraction failure shouldn't fail the whole document

                extracted_text = []
                extracted_text.append(f"# {title}")
                extracted_text.append("")

                # Iterate through all html files
                html_files = [f for f in zf.namelist() if f.lower().endswith(('.html', '.htm'))]
                
                for html_file in html_files:
                    content = zf.read(html_file)
                    text = extract_text_from_html(content)
                    if text:
                        extracted_text.append(text)

                full_text = "\n\n".join(extracted_text)

                metadata = {
                    "file_type": "scorm",
                    "kind": "text",
                    "title": title,
                    "html_file_count": len(html_files)
                }

                return build_document(path, full_text, extra_metadata=metadata)

        except zipfile.BadZipFile as e:
            raise DocumentExtractionError(f"Failed to open zip file {path.name}: {str(e)}") from e
