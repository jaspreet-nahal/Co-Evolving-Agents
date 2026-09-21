import re
import json
import uuid
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from abc import ABC, abstractmethod


@dataclass
class CorpusDocument:
    doc_id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunks: List['CorpusChunk'] = field(default_factory=list)


@dataclass
class CorpusChunk:
    chunk_id: str
    doc_id: str
    content: str
    start_char: int = 0
    end_char: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None


class CorpusIndex(ABC):

    @abstractmethod
    def search(self, query: str, top_k: int, exclude_chunk_ids: Set[str]) -> List[CorpusChunk]:
        pass

    @abstractmethod
    def grep(self, pattern: str, max_results: int, exclude_chunk_ids: Set[str]) -> List[CorpusChunk]:
        pass

    @abstractmethod
    def get_document(self, doc_id: str) -> Optional[CorpusDocument]:
        pass

    @abstractmethod
    def get_chunk(self, chunk_id: str) -> Optional[CorpusChunk]:
        pass


class InMemoryCorpusIndex(CorpusIndex):

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.documents: Dict[str, CorpusDocument] = {}
        self.chunks: Dict[str, CorpusChunk] = {}
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._bm25_index = None

    def add_document(self, doc: CorpusDocument):
        self.documents[doc.doc_id] = doc
        for chunk in doc.chunks:
            self.chunks[chunk.chunk_id] = chunk

    def load_from_jsonl(self, file_path: str, text_field: str = "text", id_field: str = "id"):
        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                doc_id = data.get(id_field, str(uuid.uuid4()))
                content = data.get(text_field, "")
                metadata = {k: v for k, v in data.items() if k not in [text_field, id_field]}

                doc = CorpusDocument(doc_id=doc_id, content=content, metadata=metadata)
                self._chunk_document(doc)
                self.add_document(doc)

    def _chunk_document(self, doc: CorpusDocument):
        content = doc.content
        chunks = []
        start = 0
        chunk_idx = 0

        while start < len(content):
            end = min(start + self.chunk_size, len(content))
            chunk_content = content[start:end]

            chunk = CorpusChunk(
                chunk_id=f"{doc.doc_id}_chunk_{chunk_idx}",
                doc_id=doc.doc_id,
                content=chunk_content,
                start_char=start,
                end_char=end,
                metadata=doc.metadata.copy()
            )
            chunks.append(chunk)
            doc.chunks.append(chunk)

            start += self.chunk_size - self.chunk_overlap
            chunk_idx += 1

    def search(self, query: str, top_k: int = 10, exclude_chunk_ids: Set[str] = None) -> List[CorpusChunk]:
        exclude_chunk_ids = exclude_chunk_ids or set()

        query_terms = set(query.lower().split())
        scored_chunks = []

        for chunk in self.chunks.values():
            if chunk.chunk_id in exclude_chunk_ids:
                continue

            chunk_terms = set(chunk.content.lower().split())
            overlap = len(query_terms & chunk_terms)
            if overlap > 0:
                dense_score = np.random.random() * 0.3
                bm25_score = overlap / len(query_terms) * 0.7
                total_score = bm25_score + dense_score
                scored_chunks.append((chunk, total_score))

        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        results = []
        for chunk, score in scored_chunks[:top_k]:
            chunk.score = score
            results.append(chunk)

        return results

    def grep(self, pattern: str, max_results: int = 5, exclude_chunk_ids: Set[str] = None) -> List[CorpusChunk]:
        exclude_chunk_ids = exclude_chunk_ids or set()
        regex = re.compile(pattern, re.IGNORECASE | re.MULTILINE)

        matches = []
        for chunk in self.chunks.values():
            if chunk.chunk_id in exclude_chunk_ids:
                continue
            if regex.search(chunk.content):
                matches.append(chunk)
                if len(matches) >= max_results:
                    break

        return matches

    def get_document(self, doc_id: str) -> Optional[CorpusDocument]:
        return self.documents.get(doc_id)

    def get_chunk(self, chunk_id: str) -> Optional[CorpusChunk]:
        return self.chunks.get(chunk_id)


class SearchReadTools:

    def __init__(self, corpus_index: CorpusIndex):
        self.corpus_index = corpus_index
        self.seen_chunk_ids: Set[str] = set()
        self.all_search_results: List[Dict[str, Any]] = []

    def search_corpus(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        results = self.corpus_index.search(query, top_k, self.seen_chunk_ids)

        new_chunk_ids = [c.chunk_id for c in results]
        self.seen_chunk_ids.update(new_chunk_ids)

        result = {
            "tool": "search_corpus",
            "query": query,
            "top_k": top_k,
            "results": [
                {
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "content": c.content[:500],
                    "score": c.score,
                    "metadata": c.metadata
                }
                for c in results
            ],
            "new_chunk_ids": new_chunk_ids,
            "total_seen_chunks": len(self.seen_chunk_ids)
        }

        self.all_search_results.append(result)
        return result

    def grep_corpus(self, pattern: str, max_results: int = 5) -> Dict[str, Any]:
        results = self.corpus_index.grep(pattern, max_results, self.seen_chunk_ids)

        new_chunk_ids = [c.chunk_id for c in results]
        self.seen_chunk_ids.update(new_chunk_ids)

        result = {
            "tool": "grep_corpus",
            "pattern": pattern,
            "max_results": max_results,
            "results": [
                {
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "content": c.content[:500],
                    "metadata": c.metadata
                }
                for c in results
            ],
            "new_chunk_ids": new_chunk_ids,
            "total_seen_chunks": len(self.seen_chunk_ids)
        }

        self.all_search_results.append(result)
        return result

    def read_document(self, doc_id: str) -> Dict[str, Any]:
        doc = self.corpus_index.get_document(doc_id)

        if not doc:
            return {
                "tool": "read_document",
                "doc_id": doc_id,
                "error": "Document not found",
                "results": []
            }

        chunk_ids = [c.chunk_id for c in doc.chunks]
        new_chunk_ids = [cid for cid in chunk_ids if cid not in self.seen_chunk_ids]
        self.seen_chunk_ids.update(new_chunk_ids)

        result = {
            "tool": "read_document",
            "doc_id": doc_id,
            "results": [
                {
                    "chunk_id": c.chunk_id,
                    "content": c.content,
                    "start_char": c.start_char,
                    "end_char": c.end_char,
                    "metadata": c.metadata
                }
                for c in doc.chunks
            ],
            "new_chunk_ids": new_chunk_ids,
            "total_seen_chunks": len(self.seen_chunk_ids)
        }

        self.all_search_results.append(result)
        return result

    def prune_chunks(self, chunk_ids: List[str]) -> Dict[str, Any]:
        pruned = [cid for cid in chunk_ids if cid in self.seen_chunk_ids]

        result = {
            "tool": "prune_chunks",
            "requested_chunk_ids": chunk_ids,
            "pruned_chunk_ids": pruned,
            "remaining_seen_chunks": len(self.seen_chunk_ids)
        }

        self.all_search_results.append(result)
        return result

    def get_seen_chunk_ids(self) -> List[str]:
        return list(self.seen_chunk_ids)

    def reset(self):
        self.seen_chunk_ids = set()
        self.all_search_results = []