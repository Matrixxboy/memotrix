class ConfigurationError(Exception):
    """Raised when required Memotrix configuration is missing or invalid."""


class DocumentExtractionError(Exception):
    """Base class for document extraction failures."""


class UnsupportedDocumentTypeError(DocumentExtractionError):
    """Raised when the file extension is not supported."""


class EmptyDocumentError(DocumentExtractionError):
    """Raised when the extracted content is empty or whitespace-only."""


class ExtractionBackendError(DocumentExtractionError):
    """Raised when an optional backend for a document type is unavailable."""
