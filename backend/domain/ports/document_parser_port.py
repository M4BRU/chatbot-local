"""Document Parser Port - Interface for document parsing operations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ParsedChunk:
    """Represents a parsed document chunk."""

    content: str
    metadata: dict
    chunk_index: int


class DocumentParserPort(ABC):
    """Port interface for document parsing operations."""

    @abstractmethod
    async def parse(self, file_path: str, file_type: str) -> list[ParsedChunk]:
        """Parse a document into chunks.

        Args:
            file_path: Path to the document file
            file_type: MIME type or file extension

        Returns:
            List of parsed chunks with metadata
        """
        pass

    @abstractmethod
    def supported_types(self) -> list[str]:
        """Get list of supported file types.

        Returns:
            List of MIME types or extensions
        """
        pass
