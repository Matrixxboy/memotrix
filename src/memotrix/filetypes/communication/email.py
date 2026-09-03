"""Email messages as episodic memory text."""

from __future__ import annotations

import mailbox
from email import policy
from email.message import Message
from email.parser import BytesParser
from pathlib import Path
from typing import List

from memotrix.filetypes.document.base import BaseExtractor
from memotrix.filetypes.document.html import extract_text_from_html
from memotrix.utils.exceptions import EmptyDocumentError
from memotrix.utils.outputSturcture import build_document


def _decode_body(message: Message) -> str:
    if message.is_multipart():
        texts: List[str] = []
        htmls: List[str] = []
        for part in message.walk():
            if part.is_multipart():
                continue
            ctype = (part.get_content_type() or "").lower()
            payload = part.get_payload(decode=True)
            if payload is None:
                continue
            charset = part.get_content_charset() or "utf-8"
            decoded = payload.decode(charset, errors="replace")
            if ctype == "text/plain":
                texts.append(decoded)
            elif ctype == "text/html":
                htmls.append(extract_text_from_html(decoded))
        return "\n\n".join(texts or htmls)
    payload = message.get_payload(decode=True)
    if payload is None:
        return str(message.get_payload() or "")
    charset = message.get_content_charset() or "utf-8"
    body = payload.decode(charset, errors="replace")
    if (message.get_content_type() or "").lower() == "text/html":
        return extract_text_from_html(body)
    return body


def format_message(message: Message, index: int | None = None) -> str:
    subject = message.get("Subject") or "(no subject)"
    sender = message.get("From") or ""
    to = message.get("To") or ""
    date = message.get("Date") or ""
    heading = f"# Email {index}" if index is not None else "# Email"
    lines = [
        heading,
        f"Subject: {subject}",
        f"From: {sender}",
        f"To: {to}",
        f"Date: {date}",
        "",
        _decode_body(message).strip(),
    ]
    return "\n".join(lines).strip()


class EmailExtractor(BaseExtractor):
    supported_extensions = (".eml", ".mbox")

    def extract(self, path: Path):
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".mbox":
            box = mailbox.mbox(path)
            try:
                parts: List[str] = []
                for index, message in enumerate(box, start=1):
                    parts.append(format_message(message, index))
                    if index >= 500:
                        break
                text = "\n\n".join(parts)
            finally:
                box.close()
        else:
            with path.open("rb") as handle:
                message = BytesParser(policy=policy.default).parse(handle)
            text = format_message(message)
        if not text.strip():
            raise EmptyDocumentError(f"No email content in {path.name}")
        return build_document(
            path,
            text,
            extra_metadata={"file_type": "email", "kind": "episodic"},
        )
