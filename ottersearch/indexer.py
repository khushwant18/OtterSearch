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

    
    def index_directory(self, directory: Path, recursive: bool = True, update_mode: bool = False):
        global indexing_status
        
        start = time.time()
        self.vector_store_pdf.initialize()
        self.vector_store_image.initialize()

        with MetadataStore() as store:
            files = self._discover_files(directory, recursive)
            
            # Only filter for new files if update_mode is True
            if update_mode:
                docs_to_index = [f for f in files if not store.document_exists(f)]
            else:
                docs_to_index = files  # Index everything
            
            indexing_status["total"] = len(docs_to_index)

            if not docs_to_index:
                print("No new documents to index")
                indexing_status["total"] = 0
                indexing_status["processed"] = 0
                return

            print(f"Processing {len(docs_to_index)} files...")

            chunk_size = 100
            max_workers = 5

            for i in range(0, len(docs_to_index), chunk_size):
                chunk = docs_to_index[i:i + chunk_size]
                batch_num = i//chunk_size + 1
                total_batches = (len(docs_to_index)-1)//chunk_size + 1

                batch_start = time.time() 
                
                # Update progress
                indexing_status["progress"] = f"Batch {batch_num}/{total_batches}"
                indexing_status["processed"] = i
                
                print(f"\nProcessing batch {batch_num}/{total_batches}")

                documents = []

                #  Parallel file processing
                processing_start = time.time()
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
                            print(f"Error processing {futures[future]}: {e}")
                
                print(f"⏱️  File processing: {time.time() - processing_start:.2f}s")  # ADD THIS
        
                # Flatten chunks
                flatten_start = time.time() 
                all_chunks = []
                for doc_or_chunks in documents:
                    if isinstance(doc_or_chunks, list):
                        all_chunks.extend(doc_or_chunks)
                    else:
                        all_chunks.append(doc_or_chunks)
                print(f"⏱️  Flattening: {time.time() - flatten_start:.2f}s ({len(all_chunks)} chunks)")  # ADD THIS

                if all_chunks:
                    indexing_start = time.time() 
                    with MetadataStore() as batch_store:
                        self._index_batch(all_chunks, batch_store)
                    print(f"⏱️  Indexing batch: {time.time() - indexing_start:.2f}s")  # ADD THIS

                print(f"⏱️  TOTAL BATCH TIME: {time.time() - batch_start:.2f}s")  # ADD THIS
                print(f"{'='*60}\n")

                # Update processed count after each batch
                indexing_status["processed"] = min(i + chunk_size, len(docs_to_index))

                gc.collect()
                
        # Final update
        indexing_status["processed"] = len(docs_to_index)
        end = time.time()
        print(f"\nIndexing completed in {end - start:.2f} seconds.")
    
    def index_file(self, path: Path):
        self.vector_store_pdf.initialize()  # Change this
        self.vector_store_image.initialize()  # Add this
        
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
            Single Document for images, list of Documents for chunked PDFs, or None if failed
        """
        file_start = time.time() 

        if path.stat().st_size > 50 * 1024 * 1024:
            print(f"Skipping large file: {path}")
            return None
        
        try:
            suffix = path.suffix.lower()
            
            if suffix == ".pdf":
                extract_start = time.time() 
                content, page_count = extract_pdf(path)
                print(f"    PDF extract {path.name}: {time.time() - extract_start:.2f}s")  # ADD THIS
            
                chunk_start = time.time()   
                chunks = self.model_manager.chunk_text_token_safe(
                                content,
                                max_tokens=config.chunk_size,
                                overlap=50,
                            )
                print(f"    PDF chunk {path.name}: {time.time() - chunk_start:.2f}s ({len(chunks)} chunks)")  # ADD THIS

                
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
            print(f"Failed to process {path}: {e}")
            return None
    
    def _index_batch(self, documents: list[Document], store: MetadataStore):
        """Index a batch of documents by encoding and storing embeddings.
        
        Args:
            documents: List of Document objects to index
            store: MetadataStore instance for storing document metadata
        """
        pdf_items = []
        pdf_docs = []
        image_items = []
        image_docs = []

        db_start = time.time()
        store.conn.execute("BEGIN")
        
        for doc in documents:
            if doc.doc_type == "image":
                image_items.append(doc.content)
                image_docs.append(doc)
            else:
                filename_prefix = f"Filename: {doc.path.name}\n\n"
                # Simple truncation instead of token-safe chunking
                content = (filename_prefix + doc.content)[:2000]
                pdf_items.append(content)
                pdf_docs.append(doc)
            
            store.upsert_document(doc)

        store.conn.commit()
        print(f"  ⏱️  DB operations: {time.time() - db_start:.2f}s")
        
        # Index PDFs with text encoder
        if pdf_items:
            pdf_encode_start = time.time() 
            pdf_embeddings = self.model_manager.encode_text(pdf_items)
            pdf_doc_ids = [doc.id for doc in pdf_docs]
            print(f"  ⏱️  PDF encoding ({len(pdf_items)} items): {time.time() - pdf_encode_start:.2f}s")
            
            pdf_add_start = time.time()
            self.vector_store_pdf.add_vectors(pdf_doc_ids, pdf_embeddings)
            self.vector_store_pdf.save()
            print(f"  ⏱️  PDF vector add+save: {time.time() - pdf_add_start:.2f}s")
            del pdf_embeddings, pdf_doc_ids
        
        # Index images with CLIP
        if image_items:
            img_encode_start = time.time() 
            image_embeddings, success_idx = self.model_manager.encode_images(image_items)
            
            if len(image_embeddings) > 0:
                print(f"  ⏱️  Image encoding ({len(image_embeddings)}/{len(image_items)} items): {time.time() - img_encode_start:.2f}s")
                image_doc_ids = [image_docs[i].id for i in success_idx]
                self.vector_store_image.add_vectors(image_doc_ids, image_embeddings)
                self.vector_store_image.save()
                del image_embeddings, image_doc_ids
        
        del pdf_items, pdf_docs, image_items, image_docs
