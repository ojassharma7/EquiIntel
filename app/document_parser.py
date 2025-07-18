"""
Document Parser for EquIntel
Handles PDF parsing, text extraction, and metadata extraction
"""

import os
import re
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import fitz  # PyMuPDF
import pdfplumber
from unstructured.partition.auto import partition
from pydantic import BaseModel


class DocumentMetadata(BaseModel):
    """Metadata extracted from financial documents"""
    ticker: Optional[str] = None
    doc_type: Optional[str] = None
    quarter: Optional[str] = None
    year: Optional[int] = None
    date: Optional[str] = None
    source: Optional[str] = None
    file_path: str
    file_size: int
    page_count: int


class ParsedDocument(BaseModel):
    """Complete parsed document with text and metadata"""
    metadata: DocumentMetadata
    text: str
    sections: Dict[str, str] = {}
    tables: List[Dict] = []


class DocumentParser:
    """Main document parser for financial PDFs"""
    
    def __init__(self, output_dir: str = "parsed_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Common patterns for financial documents
        self.ticker_pattern = r'\b([A-Z]{1,5})\b'
        self.date_patterns = [
            r'(\d{1,2})/(\d{1,2})/(\d{4})',
            r'(\d{4})-(\d{1,2})-(\d{1,2})',
            r'(\w+)\s+(\d{1,2}),?\s+(\d{4})'
        ]
        self.doc_type_patterns = {
            '10-K': r'\b10-?K\b',
            '10-Q': r'\b10-?Q\b',
            '8-K': r'\b8-?K\b',
            'earnings_call': r'earnings\s+call|quarterly\s+call',
            'annual_report': r'annual\s+report|year\s+end',
            'quarterly_report': r'quarterly\s+report|quarter\s+end'
        }
    
    def extract_metadata_from_filename(self, filename: str) -> DocumentMetadata:
        """Extract metadata from filename patterns"""
        metadata = DocumentMetadata(file_path=filename)
        
        # Extract ticker from filename
        ticker_match = re.search(self.ticker_pattern, filename.upper())
        if ticker_match:
            metadata.ticker = ticker_match.group(1)
        
        # Extract document type
        for doc_type, pattern in self.doc_type_patterns.items():
            if re.search(pattern, filename, re.IGNORECASE):
                metadata.doc_type = doc_type
                break
        
        # Extract year
        year_match = re.search(r'20\d{2}', filename)
        if year_match:
            metadata.year = int(year_match.group())
        
        # Extract quarter
        quarter_match = re.search(r'Q[1-4]', filename.upper())
        if quarter_match:
            metadata.quarter = quarter_match.group()
        
        return metadata
    
    def extract_metadata_from_content(self, text: str, filename: str) -> DocumentMetadata:
        """Extract metadata from document content"""
        metadata = self.extract_metadata_from_filename(filename)
        
        # Look for ticker in content if not found in filename
        if not metadata.ticker:
            ticker_match = re.search(self.ticker_pattern, text[:1000])
            if ticker_match:
                metadata.ticker = ticker_match.group(1)
        
        # Look for document type in content
        if not metadata.doc_type:
            for doc_type, pattern in self.doc_type_patterns.items():
                if re.search(pattern, text[:2000], re.IGNORECASE):
                    metadata.doc_type = doc_type
                    break
        
        # Look for dates in content
        for pattern in self.date_patterns:
            date_match = re.search(pattern, text[:2000])
            if date_match:
                try:
                    if len(date_match.groups()) == 3:
                        if len(date_match.group(1)) == 4:  # YYYY-MM-DD
                            metadata.date = f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}"
                        else:  # MM/DD/YYYY
                            metadata.date = f"{date_match.group(3)}-{date_match.group(1)}-{date_match.group(2)}"
                    break
                except:
                    continue
        
        return metadata
    
    def extract_tables(self, pdf_path: str) -> List[Dict]:
        """Extract tables from PDF using pdfplumber"""
        tables = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_tables = page.extract_tables()
                    for table_num, table in enumerate(page_tables):
                        if table and len(table) > 1:  # At least header + one row
                            tables.append({
                                'page': page_num + 1,
                                'table_num': table_num + 1,
                                'data': table,
                                'rows': len(table),
                                'cols': len(table[0]) if table else 0
                            })
        except Exception as e:
            print(f"Error extracting tables from {pdf_path}: {e}")
        
        return tables
    
    def extract_sections(self, text: str) -> Dict[str, str]:
        """Extract document sections based on common financial document structure"""
        sections = {}
        
        # Common section headers in financial documents
        section_patterns = {
            'risk_factors': r'risk\s+factors?|risk\s+and\s+uncertainties?',
            'business_overview': r'business\s+overview|company\s+overview|overview',
            'financial_statements': r'financial\s+statements?|consolidated\s+financial\s+statements?',
            'management_discussion': r'management\s+discussion|md&a|management\s+analysis',
            'executive_summary': r'executive\s+summary|summary',
            'forward_looking': r'forward\s+looking|cautionary\s+statements?'
        }
        
        lines = text.split('\n')
        current_section = 'general'
        current_content = []
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Check if this line is a section header
            section_found = False
            for section_name, pattern in section_patterns.items():
                if re.search(pattern, line_lower):
                    # Save previous section
                    if current_content:
                        sections[current_section] = '\n'.join(current_content).strip()
                    
                    # Start new section
                    current_section = section_name
                    current_content = [line]
                    section_found = True
                    break
            
            if not section_found:
                current_content.append(line)
        
        # Save the last section
        if current_content:
            sections[current_section] = '\n'.join(current_content).strip()
        
        return sections
    
    def parse_pdf(self, pdf_path: str) -> ParsedDocument:
        """Parse a PDF file and extract text, metadata, and tables"""
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Get file info
        file_size = pdf_path.stat().st_size
        
        # Extract text using PyMuPDF
        text = ""
        page_count = 0
        try:
            doc = fitz.open(pdf_path)
            page_count = len(doc)
            
            for page in doc:
                text += page.get_text()
            
            doc.close()
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            # Fallback to unstructured
            try:
                elements = partition(str(pdf_path))
                text = "\n".join([str(element) for element in elements])
            except Exception as e2:
                print(f"Fallback extraction also failed: {e2}")
                text = ""
        
        # Extract metadata
        metadata = self.extract_metadata_from_content(text, pdf_path.name)
        metadata.file_size = file_size
        metadata.page_count = page_count
        metadata.source = str(pdf_path)
        
        # Extract tables
        tables = self.extract_tables(str(pdf_path))
        
        # Extract sections
        sections = self.extract_sections(text)
        
        return ParsedDocument(
            metadata=metadata,
            text=text,
            sections=sections,
            tables=tables
        )
    
    def parse_and_save(self, pdf_path: str) -> str:
        """Parse PDF and save to JSON file"""
        parsed_doc = self.parse_pdf(pdf_path)
        
        # Create output filename
        pdf_name = Path(pdf_path).stem
        output_file = self.output_dir / f"{pdf_name}_parsed.json"
        
        # Save to JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(parsed_doc.model_dump(), f, indent=2, ensure_ascii=False)
        
        print(f"Parsed {pdf_path} -> {output_file}")
        return str(output_file)
    
    def parse_directory(self, pdf_dir: str) -> List[str]:
        """Parse all PDFs in a directory"""
        pdf_dir = Path(pdf_dir)
        output_files = []
        
        for pdf_file in pdf_dir.glob("*.pdf"):
            try:
                output_file = self.parse_and_save(str(pdf_file))
                output_files.append(output_file)
            except Exception as e:
                print(f"Error parsing {pdf_file}: {e}")
        
        return output_files


if __name__ == "__main__":
    import sys
    
    parser = DocumentParser()
    
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        if os.path.isdir(pdf_path):
            parser.parse_directory(pdf_path)
        else:
            parser.parse_and_save(pdf_path)
    else:
        print("Usage: python document_parser.py <pdf_file_or_directory>") 