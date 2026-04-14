"""
Rebuild vector database with English documents
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from document_processor import DocumentProcessor
from vector_store import VectorStore

def main():
    print("="*60)
    print("REBUILDING VECTOR DATABASE WITH ENGLISH DOCUMENTS")
    print("="*60)
    
    # Process documents
    processor = DocumentProcessor(chunk_size=800, chunk_overlap=200)
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    chunks = processor.process_all_documents(data_dir)
    
    if len(chunks) == 0:
        print("❌ No chunks created! Check document files.")
        return
    
    # Create vector store
    print('\n' + "="*60)
    print('BUILDING VECTOR DATABASE...')
    print("="*60)
    
    vector_store = VectorStore(persist_directory='vector_db')
    
    # Add chunks (VectorStore expects DocumentChunk objects)
    vector_store.add_documents(chunks)
    
    print('\n' + "="*60)
    print(f'✅ SUCCESS! Vector database created!')
    print(f'   Total chunks: {len(chunks)}')
    print(f'   Location: vector_db/')
    print("="*60)

if __name__ == "__main__":
    main()
