import re
from pathlib import Path

from memotrix.filetypes.document.base import BaseExtractor
from memotrix.utils.outputSturcture import build_document

class ProgrammingFileExtractor(BaseExtractor):
    supported_extensions = (".py", ".js", ".ts", ".java", ".cpp", ".cc", ".cxx", ".go", ".rs")
    
    EXTENSION_TO_LANG = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".java": "java",
        ".cpp": "cpp",
        ".cc": "cpp",
        ".cxx": "cpp",
        ".go": "go",
        ".rs": "rust",
    }

    # Basic regex patterns to identify class and function definitions across languages
    BLOCK_PATTERNS = {
        "python": re.compile(r"^\s*(?:async\s+)?(?:def|class)\s+([a-zA-Z_]\w*)\b", re.MULTILINE),
        "javascript": re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?(?:function|class)\s+([a-zA-Z_$][\w$]*)\b|^\s*(?:const|let|var)\s+([a-zA-Z_$][\w$]*)\s*=\s*(?:async\s+)?(?:\([^)]*\)|[a-zA-Z_$][\w$]*)\s*=>", re.MULTILINE),
        "typescript": re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?(?:function|class|interface|type)\s+([a-zA-Z_$][\w$]*)\b|^\s*(?:const|let|var)\s+([a-zA-Z_$][\w$]*)\s*=\s*(?:async\s+)?(?:\([^)]*\)|[a-zA-Z_$][\w$]*)\s*=>", re.MULTILINE),
        "java": re.compile(r"^\s*(?:public|protected|private)?\s*(?:static\s+)?(?:final\s+)?(?:class|interface|enum)\s+([a-zA-Z_$][\w$]*)\b|^\s*(?:public|protected|private)\s+(?:static\s+)?[a-zA-Z_$\<>\.,\s]+\s+([a-zA-Z_$][\w$]*)\s*\(", re.MULTILINE),
        "cpp": re.compile(r"^\s*(?:class|struct|namespace)\s+([a-zA-Z_]\w*)\b|^\s*(?:virtual\s+)?[a-zA-Z_][a-zA-Z0-9_:\<\>\*&\s]+\s+([a-zA-Z_]\w*)\s*\(", re.MULTILINE),
        "go": re.compile(r"^\s*func\s+(?:\([^)]+\)\s+)?([a-zA-Z_]\w*)\b|^\s*type\s+([a-zA-Z_]\w*)\s+(?:struct|interface)", re.MULTILINE),
        "rust": re.compile(r"^\s*(?:pub\s+)?(?:async\s+)?fn\s+([a-zA-Z_]\w*)\b|^\s*(?:pub\s+)?(?:struct|enum|trait|impl)\s+(?:[a-zA-Z_]\w*\s+for\s+)?([a-zA-Z_]\w*)", re.MULTILINE),
        "html": re.compile(r"^\s*<(script|style|main|section|article|nav|header|footer)\b", re.IGNORECASE | re.MULTILINE),
    }

    def extract(self, path: Path):
        ext = path.suffix.lower()
        lang_name = self.EXTENSION_TO_LANG.get(ext)
        if not lang_name:
            raise ValueError(f"Unsupported language extension: {ext}")
            
        source_text = self.read_text(path)
        pattern = self.BLOCK_PATTERNS.get(lang_name)
        
        blocks = []
        if pattern:
            # We will chunk the file by splitting at the start of each block pattern
            matches = list(pattern.finditer(source_text))
            
            if matches:
                # Add any preamble (imports, global variables) before the first block
                first_match_start = matches[0].start()
                if first_match_start > 0:
                    preamble = source_text[:first_match_start].strip()
                    if preamble:
                        blocks.append(f"# Global context (Imports, Variables)\n```\n{preamble}\n```")
                
                # Add each block
                for i, match in enumerate(matches):
                    start_idx = match.start()
                    end_idx = matches[i+1].start() if i + 1 < len(matches) else len(source_text)
                    
                    block_content = source_text[start_idx:end_idx].strip()
                    # The name is either group 1 or group 2 depending on the regex
                    block_name = match.group(1) or match.group(2) or "Block"
                    
                    blocks.append(f"# {lang_name.title()} Block: {block_name}\n```\n{block_content}\n```")
            else:
                # No blocks found, treat whole file as one block
                blocks.append(f"# Source File: {path.name}\n```\n{source_text}\n```")
        else:
            blocks.append(f"# Source File: {path.name}\n```\n{source_text}\n```")

        full_text = "\n\n".join(blocks)

        metadata = {
            "file_type": "source_code",
            "language": lang_name,
            "block_count": len(blocks)
        }
        
        return build_document(path, full_text, extra_metadata=metadata, language=lang_name)
