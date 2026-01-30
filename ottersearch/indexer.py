"""
Indexing module for OtterSearch
"""
import time
import gc
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from .config import config
from .models import Document
from .extractors import extract_pdf, extract_image
from .storage import VectorStore, MetadataStore
from .ml_models import ModelManager


# Global indexing status
indexing_status = {
    "running": False, 
    "progress": "", 
    "error": None, 
    "count": 0,
    "total": 0,      
    "processed": 0  
}


class Indexer:
    def __init__(self):
        self.model_manager = ModelManager()
        self.vector_store_pdf = VectorStore(store_type="pdf", dim=config.text_vector_dim)
        self.vector_store_image = VectorStore(store_type="image", dim=config.image_vector_dim)

    
    def index_directory(self, directory: Path, recursive: bool = True, update_mode: bool = True):
        """Index all PDF and image files in a directory.
        
        Args:
            directory: Path to directory to index
            recursive: If True, recursively search subdirectories
            update_mode: If True (default), only index new files not already in metadata store.
                        If False, reindex all files regardless of existing entries.
        
        Updates global indexing_status dict with progress information:
            - total: Total number of files to index
            - processed: Number of files completed
            - progress: Current batch status string
            
        Error Handling:
            - Files larger than 50MB are silently skipped
            - Individual file processing errors do not stop the batch
            - Failed embeddings are skipped but indexing continues
            - No rollback on partial failure - successfully processed files remain indexed
        """
        global indexing_status
        
        start = time.time()
        self.vector_store_pdf.initialize()
        self.vector_store_image.initialize()

        with MetadataStore() as store:
            files = self._discover_files(directory, recursive)
            
            if update_mode:
                docs_to_index = [f for f in files if not store.document_exists(f)]
            else:
                docs_to_index = files
            
            indexing_status["total"] = len(docs_to_index)

            if not docs_to_index:
                indexing_status["total"] = 0
                indexing_status["processed"] = 0
                return

            chunk_size = 100
            max_workers = 5

            for i in range(0, len(docs_to_index), chunk_size):
                chunk = docs_to_index[i:i + chunk_size]
                batch_num = i//chunk_size + 1
                total_batches = (len(docs_to_index)-1)//chunk_size + 1

                indexing_status["progress"] = f"Batch {batch_num}/{total_batches}"
                indexing_status["processed"] = i
                
                documents = []

                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(self._process_file, file_path): file_path
                        for file_path in chunk
                    }

                    for future in as_completed(futures):
                        try:
                            result = future.result()
                            if result is not None:
                                documents.append(result)
                        except Exception as e:
                            pass
        
                all_chunks = []
                for doc_or_chunks in documents:
                    if isinstance(doc_or_chunks, list):
                        all_chunks.extend(doc_or_chunks)
                    else:
                        all_chunks.append(doc_or_chunks)

                if all_chunks:
                    with MetadataStore() as batch_store:
                        self._index_batch(all_chunks, batch_store)
                indexing_status["processed"] = min(i + chunk_size, len(docs_to_index))

                gc.collect()
                
        indexing_status["processed"] = len(docs_to_index)
    
    def index_file(self, path: Path):
        """Index a single PDF or image file.
        
        Args:
            path: Path to the file to index
            
        Error Handling:
            - Returns silently if file processing fails
            - No error is raised to caller
            - Metadata store connection errors will propagate
        """
        self.vector_store_pdf.initialize()
        self.vector_store_image.initialize()
        
        doc_or_chunks = self._process_file(path)
        if doc_or_chunks is None:
            return
        
        chunks = doc_or_chunks if isinstance(doc_or_chunks, list) else [doc_or_chunks]
        
        with MetadataStore() as store:
            self._index_batch(chunks, store)
    
    def _discover_files(self, directory: Path, recursive: bool) -> list[Path]:
        """Discover PDF and image files in directory.
        
        Args:
            directory: Path to search for files
            recursive: If True, search subdirectories recursively
            
        Returns:
            List of Path objects for all discovered files
        """
        patterns = ['*.pdf', '*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
        files = []
        
        for pattern in patterns:
            if recursive:
                files.extend(directory.rglob(pattern))
            else:
                files.extend(directory.glob(pattern))
        
        return files
    
    def _process_file(self, path: Path) -> Document | list[Document] | None:
        """Process a single file (PDF or image) and extract content.
        
        Args:
            path: Path to the file to process
            
        Returns:
            - For images: Single Document object
            - For small PDFs: Single Document object
            - For large PDFs: List of Document objects (one per chunk)
            - None if processing fails or file is too large
            
        Error Handling:
            - Files > 50MB are skipped (returns None)
            - PDF extraction errors return None
            - Image loading errors return None
            - All exceptions are caught and return None (no propagation)
        """
        if path.stat().st_size > 50 * 1024 * 1024:
            return None
        
        try:
            suffix = path.suffix.lower()
            
            if suffix == ".pdf":
                content, page_count = extract_pdf(path)
            
                chunks = self.model_manager.chunk_text_token_safe(
                    content,
                    max_tokens=config.chunk_size,
                    overlap=50,
                )
                
                if len(chunks) == 1:
                    doc = Document.from_path(path, content, "pdf")
                    doc.page_count = page_count
                    return doc
                
                chunk_docs = []
                for i, chunk in enumerate(chunks):
                    doc = Document.from_path(path, chunk, "pdf")
                    doc.page_count = page_count
                    doc.chunk_index = i
                    doc.total_chunks = len(chunks)
                    chunk_docs.append(doc)
                
                return chunk_docs
                
            elif suffix in {".png", ".jpg", ".jpeg"}:
                content = extract_image(path)
                return Document.from_path(path, content, "image")
            
            return None
        except Exception as e:
            return None
    
    def _index_batch(self, documents: list[Document], store: MetadataStore):
        """Index a batch of documents by encoding and storing embeddings.
        
        Separates documents by type (PDF/image), encodes using appropriate models,
        and stores embeddings in type-specific vector stores.
        
        Args:
            documents: List of Document objects to index
            store: MetadataStore instance for storing document metadata (must be connected)
            
        Error Handling:
            - Failed image encodings are skipped (only successful ones indexed)
            - Database operations wrapped in transaction (BEGIN/COMMIT)
            - No rollback on vector store failures
            - Embedding failures are silent - only successful embeddings are stored
            
        Side Effects:
            - Modifies vector stores (PDF and image)
            - Commits transaction to metadata store
            - Saves vector store indexes to disk
        """
        pdf_items = []
        pdf_docs = []
        image_items = []
        image_docs = []

        store.conn.execute("BEGIN")
        
        for doc in documents:
            if doc.doc_type == "image":
                image_items.append(doc.content)
                image_docs.append(doc)
            else:
                filename_prefix = f"Filename: {doc.path.name}\n\n"
                content = (filename_prefix + doc.content)[:2000]
                pdf_items.append(content)
                pdf_docs.append(doc)
            
            store.upsert_document(doc)

        store.conn.commit()
        
        if pdf_items:
            pdf_embeddings = self.model_manager.encode_text(pdf_items)
            pdf_doc_ids = [doc.id for doc in pdf_docs]
            
            self.vector_store_pdf.add_vectors(pdf_doc_ids, pdf_embeddings)
            self.vector_store_pdf.save()
            del pdf_embeddings, pdf_doc_ids
        
        if image_items:
            image_embeddings, success_idx = self.model_manager.encode_images(image_items)
            
            if len(image_embeddings) > 0:
                image_doc_ids = [image_docs[i].id for i in success_idx]
                self.vector_store_image.add_vectors(image_doc_ids, image_embeddings)
                self.vector_store_image.save()
                del image_embeddings, image_doc_ids
        
        del pdf_items, pdf_docs, image_items, image_docs
