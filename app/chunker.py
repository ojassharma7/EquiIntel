"""
Document Chunker for EquIntel
Splits documents into semantic chunks for embedding
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Optional
from pydantic import BaseModel


class TextChunk(BaseModel):
    """A chunk of text with metadata"""
    chunk_id: str
    text: str
    metadata: Dict
    start_char: int
    end_char: int
    token_count: Optional[int] = None


class DocumentChunker:
    """Chunks documents into semantic units for embedding"""
    
    def __init__(self, chunk_size: int = 500, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        # Common sentence endings
        self.sentence_endings = r'[.!?]\s+'
        
        # Common paragraph breaks
        self.paragraph_breaks = r'\n\s*\n'
        
        # Financial document specific patterns
        self.section_headers = r'^[A-Z][A-Z\s]{2,50}$'
        self.subsection_headers = r'^\d+\.\s+[A-Z]'
    
    def estimate_tokens(self, text: str) -> int:
        """Rough estimate of token count (4 chars per token)"""
        return len(text) // 4
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Split on sentence endings, but preserve the endings
        sentences = re.split(f'({self.sentence_endings})', text)
        
        # Recombine sentences with their endings
        result = []
        current_sentence = ""
        
        for i, part in enumerate(sentences):
            current_sentence += part
            if re.match(self.sentence_endings, part):
                result.append(current_sentence.strip())
                current_sentence = ""
        
        # Add any remaining text
        if current_sentence.strip():
            result.append(current_sentence.strip())
        
        return [s for s in result if s.strip()]
    
    def split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs"""
        paragraphs = re.split(self.paragraph_breaks, text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def is_section_header(self, text: str) -> bool:
        """Check if text looks like a section header"""
        return bool(re.match(self.section_headers, text.strip()))
    
    def is_subsection_header(self, text: str) -> bool:
        """Check if text looks like a subsection header"""
        return bool(re.match(self.subsection_headers, text.strip()))
    
    def create_chunks_from_sentences(self, sentences: List[str], metadata: Dict) -> List[TextChunk]:
        """Create chunks from sentences with overlap"""
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_id = 0
        
        for i, sentence in enumerate(sentences):
            sentence_tokens = self.estimate_tokens(sentence)
            
            # If adding this sentence would exceed chunk size
            if current_size + sentence_tokens > self.chunk_size and current_chunk:
                # Create chunk
                chunk_text = " ".join(current_chunk)
                chunks.append(TextChunk(
                    chunk_id=f"{metadata.get('file_path', 'doc')}_{chunk_id}",
                    text=chunk_text,
                    metadata=metadata.copy(),
                    start_char=0,  # Will be calculated later
                    end_char=len(chunk_text),
                    token_count=current_size
                ))
                chunk_id += 1
                
                # Start new chunk with overlap
                overlap_sentences = []
                overlap_size = 0
                
                # Add sentences from the end for overlap
                for j in range(len(current_chunk) - 1, -1, -1):
                    if overlap_size >= self.overlap:
                        break
                    overlap_sentences.insert(0, current_chunk[j])
                    overlap_size += self.estimate_tokens(current_chunk[j])
                
                current_chunk = overlap_sentences
                current_size = overlap_size
            
            current_chunk.append(sentence)
            current_size += sentence_tokens
        
        # Add the last chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(TextChunk(
                chunk_id=f"{metadata.get('file_path', 'doc')}_{chunk_id}",
                text=chunk_text,
                metadata=metadata.copy(),
                start_char=0,
                end_char=len(chunk_text),
                token_count=current_size
            ))
        
        return chunks
    
    def create_chunks_from_paragraphs(self, paragraphs: List[str], metadata: Dict) -> List[TextChunk]:
        """Create chunks from paragraphs with overlap"""
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_id = 0
        
        for i, paragraph in enumerate(paragraphs):
            paragraph_tokens = self.estimate_tokens(paragraph)
            
            # If adding this paragraph would exceed chunk size
            if current_size + paragraph_tokens > self.chunk_size and current_chunk:
                # Create chunk
                chunk_text = "\n\n".join(current_chunk)
                chunks.append(TextChunk(
                    chunk_id=f"{metadata.get('file_path', 'doc')}_{chunk_id}",
                    text=chunk_text,
                    metadata=metadata.copy(),
                    start_char=0,
                    end_char=len(chunk_text),
                    token_count=current_size
                ))
                chunk_id += 1
                
                # Start new chunk with overlap
                overlap_paragraphs = []
                overlap_size = 0
                
                # Add paragraphs from the end for overlap
                for j in range(len(current_chunk) - 1, -1, -1):
                    if overlap_size >= self.overlap:
                        break
                    overlap_paragraphs.insert(0, current_chunk[j])
                    overlap_size += self.estimate_tokens(current_chunk[j])
                
                current_chunk = overlap_paragraphs
                current_size = overlap_size
            
            current_chunk.append(paragraph)
            current_size += paragraph_tokens
        
        # Add the last chunk
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            chunks.append(TextChunk(
                chunk_id=f"{metadata.get('file_path', 'doc')}_{chunk_id}",
                text=chunk_text,
                metadata=metadata.copy(),
                start_char=0,
                end_char=len(chunk_text),
                token_count=current_size
            ))
        
        return chunks
    
    def chunk_document(self, text: str, metadata: Dict) -> List[TextChunk]:
        """Main method to chunk a document"""
        # First try paragraph-based chunking
        paragraphs = self.split_into_paragraphs(text)
        
        if len(paragraphs) > 1:
            chunks = self.create_chunks_from_paragraphs(paragraphs, metadata)
        else:
            # Fall back to sentence-based chunking
            sentences = self.split_into_sentences(text)
            chunks = self.create_chunks_from_sentences(sentences, metadata)
        
        # Update character positions
        total_chars = 0
        for chunk in chunks:
            chunk.start_char = total_chars
            chunk.end_char = total_chars + len(chunk.text)
            total_chars += len(chunk.text) + 1  # +1 for separator
        
        return chunks
    
    def chunk_sections(self, sections: Dict[str, str], metadata: Dict) -> List[TextChunk]:
        """Chunk individual sections of a document"""
        all_chunks = []
        
        for section_name, section_text in sections.items():
            if not section_text.strip():
                continue
            
            # Add section info to metadata
            section_metadata = metadata.copy()
            section_metadata['section'] = section_name
            
            # Chunk the section
            section_chunks = self.chunk_document(section_text, section_metadata)
            all_chunks.extend(section_chunks)
        
        return all_chunks
    
    def save_chunks(self, chunks: List[TextChunk], output_file: str):
        """Save chunks to JSON file"""
        output_path = Path(output_file)
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump([chunk.model_dump() for chunk in chunks], f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(chunks)} chunks to {output_file}")


if __name__ == "__main__":
    import sys
    
    chunker = DocumentChunker()
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'text' in data:
            # Single document
            metadata = data.get('metadata', {})
            text = data['text']
            chunks = chunker.chunk_document(text, metadata)
            
            output_file = input_file.replace('_parsed.json', '_chunks.json')
            chunker.save_chunks(chunks, output_file)
        else:
            # Directory of parsed documents
            for doc_file in Path(input_file).glob("*_parsed.json"):
                with open(doc_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                metadata = data.get('metadata', {})
                text = data['text']
                chunks = chunker.chunk_document(text, metadata)
                
                output_file = str(doc_file).replace('_parsed.json', '_chunks.json')
                chunker.save_chunks(chunks, output_file)
    else:
        print("Usage: python chunker.py <parsed_document.json_or_directory>") 