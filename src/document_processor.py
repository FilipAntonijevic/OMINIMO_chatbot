import os
import re
from typing import List, Dict, Tuple
from dataclasses import dataclass

import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class DocumentChunk:
    text: str
    source: str
    page: int
    chunk_id: int
    section: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "source": self.source,
            "page": self.page,
            "chunk_id": self.chunk_id,
            "section": self.section
        }


class DocumentProcessor:
    
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 200):
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[Tuple[int, str]]:
        
        pages_text = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text()
                    if text:
                        text = self._clean_text(text)
                        pages_text.append((i, text))
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
            raise
        
        return pages_text
    
    def _clean_text(self, text: str) -> str:
        
        text = re.sub(r' +', ' ', text)
        
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        
        return text.strip()
    
    def _detect_section(self, text: str) -> str:
        
        lines = text.split('\n')
        for line in lines[:3]: 
            line = line.strip()
            
            if len(line) > 0 and line.isupper() and len(line) < 100:
                return line
            
            if re.match(r'^(\d+\.|Article \d+|Section \d+|Chapter \d+)', line, re.IGNORECASE):
                return line
        
        return "General"
    
    def process_document(self, pdf_path: str, document_name: str) -> List[DocumentChunk]:
        
        print(f"Processing {document_name}...")
        
        pages_text = self.extract_text_from_pdf(pdf_path)
        
        chunks = []
        chunk_id = 0
        
        for page_num, page_text in pages_text:
            text_chunks = self.text_splitter.split_text(page_text)
            
            for chunk_text in text_chunks:
                section = self._detect_section(chunk_text)
                
                chunk = DocumentChunk(
                    text=chunk_text,
                    source=document_name,
                    page=page_num,
                    chunk_id=chunk_id,
                    section=section
                )
                chunks.append(chunk)
                chunk_id += 1
        
        print(f"Created {len(chunks)} chunks from {len(pages_text)} pages")
        return chunks
    
    def extract_text_from_txt(self, txt_path: str) -> List[Tuple[int, str]]:
        
        pages_text = []
        
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            page_splits = re.split(r'--- Page (\d+) ---', content)
            
            for i in range(1, len(page_splits), 2):
                if i + 1 < len(page_splits):
                    page_num = int(page_splits[i])
                    page_text = page_splits[i + 1].strip()
                    if page_text:
                        pages_text.append((page_num, page_text))
            
            if not pages_text and content.strip():
                pages_text.append((1, content.strip()))
                
        except Exception as e:
            print(f"Error processing {txt_path}: {e}")
            raise
        
        return pages_text

    def process_all_documents(self, data_dir: str) -> List[DocumentChunk]:
        doc_mapping = {
            "MTPL_Product_Info_EN.txt": "MTPL Product Information",
            "User_Regulations_EN.txt": "User Regulations",
            "Terms_and_Conditions_EN.txt": "Terms and Conditions"
        }
        
        all_chunks = []
        
        for filename, doc_name in doc_mapping.items():
            file_path = os.path.join(data_dir, filename)
            
            if os.path.exists(file_path):
                if filename.endswith('.txt'):
                    print(f"Processing {doc_name} from TXT...")
                    pages_text = self.extract_text_from_txt(file_path)
                    
                    chunks = []
                    chunk_id = 0
                    
                    for page_num, page_text in pages_text:
                        text_chunks = self.text_splitter.split_text(page_text)
                        
                        for chunk_text in text_chunks:
                            section = self._detect_section(chunk_text)
                            
                            chunk = DocumentChunk(
                                text=chunk_text,
                                source=doc_name,
                                page=page_num,
                                chunk_id=chunk_id,
                                section=section
                            )
                            chunks.append(chunk)
                            chunk_id += 1
                    
                    print(f"  → Created {len(chunks)} chunks from {len(pages_text)} pages")
                    all_chunks.extend(chunks)
                else:
                    chunks = self.process_document(file_path, doc_name)
                    all_chunks.extend(chunks)
            else:
                print(f"Warning: {filename} not found in {data_dir}")
        
        print(f"\nTotal chunks created: {len(all_chunks)}")
        return all_chunks


if __name__ == "__main__":
    processor = DocumentProcessor(chunk_size=800, chunk_overlap=200)
    
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(script_dir), "data")
    
    print(f"Looking for documents in: {data_dir}")
    chunks = processor.process_all_documents(data_dir)
    
    print("\n" + "="*60)
    print("SAMPLE CHUNKS:")
    print("="*60)
    for chunk in chunks[:3]:
        print(f"\nSource: {chunk.source} | Page: {chunk.page} | Section: {chunk.section}")
        print(f"Text preview: {chunk.text[:200]}...")
        print("-" * 60)
