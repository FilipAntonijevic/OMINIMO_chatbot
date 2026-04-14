import os
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from document_processor import DocumentChunk


class VectorStore:
    def __init__(self, 
                 collection_name: str = "insurance_docs",
                 persist_directory: str = "../vector_db",
                 embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        
        load_dotenv()
        
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        
        print(f"Loading embedding model: {embedding_model}...")
        self.embedding_model = SentenceTransformer(embedding_model)
        print("✓ Multilingual embedding model loaded (supports Hungarian + English)")
        
        self.chroma_client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Insurance documents for RAG chatbot"}
        )
        
        print(f"Vector store initialized. Collection: {collection_name}")
        print(f"Current documents in collection: {self.collection.count()}")
    
    def _get_embedding(self, text: str) -> List[float]:
        embedding = self.embedding_model.encode(text, convert_to_tensor=False)
        return embedding.tolist()
    
    def _get_embeddings_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
    
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = self.embedding_model.encode(batch, convert_to_tensor=False, show_progress_bar=False)
            all_embeddings.extend([emb.tolist() for emb in embeddings])
            
            print(f"  Embedded {min(i + batch_size, len(texts))}/{len(texts)} chunks")
        
        return all_embeddings
    
    def add_documents(self, chunks: List[DocumentChunk], batch_size: int = 100):
        if not chunks:
            print("No chunks to add")
            return
        
        print(f"\nAdding {len(chunks)} chunks to vector store...")
        
        texts = [chunk.text for chunk in chunks]
        ids = [f"{chunk.source}_{chunk.chunk_id}" for chunk in chunks]
        
        metadatas = [
            {
                "source": chunk.source,
                "page": chunk.page,
                "chunk_id": chunk.chunk_id,
                "section": chunk.section
            }
            for chunk in chunks
        ]
        
        print("Generating embeddings...")
        embeddings = self._get_embeddings_batch(texts, batch_size)
        
        print("Adding to vector database...")
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"Successfully added {len(chunks)} chunks")
        print(f"Total documents in collection: {self.collection.count()}")
    
    def search(self, 
               query: str, 
               top_k: int = 5,
               filter_dict: Optional[Dict] = None) -> List[Dict]:
        
        query_embedding = self._get_embedding(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_dict if filter_dict else None
        )
        
        formatted_results = []
        if results['documents'] and len(results['documents'][0]) > 0:
            for i in range(len(results['documents'][0])):
                result = {
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None,
                    'id': results['ids'][0][i]
                }
                formatted_results.append(result)
        
        return formatted_results
    
    def reset_collection(self):
        self.chroma_client.delete_collection(name=self.collection.name)
        self.collection = self.chroma_client.create_collection(
            name=self.collection.name,
            metadata={"description": "Insurance documents for RAG chatbot"}
        )
        print("Collection reset successfully")
    
    def get_stats(self) -> Dict:
        count = self.collection.count()
        
        if count > 0:
            sample = self.collection.get(limit=count)
            sources = {}
            for metadata in sample['metadatas']:
                source = metadata.get('source', 'Unknown')
                sources[source] = sources.get(source, 0) + 1
            
            return {
                'total_chunks': count,
                'sources': sources
            }
        
        return {'total_chunks': 0, 'sources': {}}


def build_vector_store(data_dir: str = "../data", 
                       persist_dir: str = "../vector_db",
                       force_rebuild: bool = False):
    
    from document_processor import DocumentProcessor
    
    vector_store = VectorStore(persist_directory=persist_dir)
    
    if vector_store.collection.count() > 0 and not force_rebuild:
        print("\nVector store already populated!")
        stats = vector_store.get_stats()
        print(f"Total chunks: {stats['total_chunks']}")
        print("Sources:")
        for source, count in stats['sources'].items():
            print(f"  - {source}: {count} chunks")
        
        rebuild = input("\nRebuild vector store? (y/n): ").lower()
        if rebuild != 'y':
            return vector_store
        
        vector_store.reset_collection()
    
    processor = DocumentProcessor(
        chunk_size=int(os.getenv("CHUNK_SIZE", 800)),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 200))
    )
    
    chunks = processor.process_all_documents(data_dir)
    
    if chunks:
        vector_store.add_documents(chunks)
    else:
        print("\nNo documents found to process!")
        print(f"Please add PDF files to {data_dir}:")
        print("  - MTPL_Product_Info.pdf")
        print("  - User_Regulations.pdf")
        print("  - Terms_and_Conditions.pdf")
    
    return vector_store


if __name__ == "__main__":
    vector_store = build_vector_store(
        data_dir="../data",
        persist_dir="../vector_db"
    )
    
    if vector_store.collection.count() > 0:
        print("\n" + "="*60)
        print("TESTING SEARCH:")
        print("="*60)
        
        test_query = "What is covered under MTPL insurance?"
        print(f"\nQuery: {test_query}")
        
        results = vector_store.search(test_query, top_k=3)
        
        for i, result in enumerate(results, 1):
            print(f"\n--- Result {i} ---")
            print(f"Source: {result['metadata']['source']} (Page {result['metadata']['page']})")
            print(f"Section: {result['metadata']['section']}")
            print(f"Distance: {result['distance']:.4f}")
            print(f"Text: {result['text'][:200]}...")
