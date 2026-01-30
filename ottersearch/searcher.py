"""
Search module for OtterSearch
"""
import torch
import numpy as np
from .config import config
from .models import SearchResult
from .storage import VectorStore, MetadataStore
from .ml_models import ModelManager


class HybridSearcher:
    """Hybrid semantic search across PDF and image documents.
    
    Combines query expansion using SLM with type-specific vector search.
    Uses separate vector stores for PDFs (text embeddings) and images (CLIP embeddings).
    """
    def __init__(self):
        """Initialize searcher with model manager and vector stores.
        
        Loads both PDF and image vector stores from disk.
        """
        self.model_manager = ModelManager()
        self.vector_store_pdf = VectorStore(store_type="pdf", dim=config.text_vector_dim)
        self.vector_store_pdf.initialize()
        self.vector_store_image = VectorStore(store_type="image", dim=config.image_vector_dim)
        self.vector_store_image.initialize()

    def _detect_query_type(self, query: str) -> str:
        """Detect whether query is for images, PDFs, or both.
        
        Args:
            query: User search query string
            
        Returns:
            'image', 'pdf', or 'both' based on detected keywords
        """
        query_lower = query.lower()
        
        image_keywords = ['photo', 'image', 'picture', 'pic', 'screenshot', 'png', 'jpg']
        pdf_keywords = ['pdf', 'doc', 'document', 'paper', 'report']
        
        has_image = any(kw in query_lower for kw in image_keywords)
        has_pdf = any(kw in query_lower for kw in pdf_keywords)
        
        if has_image and not has_pdf:
            return 'image'
        elif has_pdf and not has_image:
            return 'pdf'
        else:
            return 'both'
    

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        """Search for documents matching the query.
        
        Process:
        1. Generate query variations using SLM
        2. Detect query type (image/pdf/both) from variations
        3. Encode queries with appropriate models (CLIP for images, text encoder for PDFs)
        4. Search relevant vector stores
        5. Deduplicate results by document path, keeping highest score
        
        Args:
            query: User search query string
            top_k: Maximum results per query variation per store
            
        Returns:
            List of SearchResult objects, deduplicated and sorted by score descending.
            Each unique document path appears only once with its highest similarity score.
        """
        variations = self.model_manager.generate_query_variations(query, n=3)
        query_type = [self._detect_query_type(variation) for variation in variations]
        if 'image' in query_type:
            query_type = 'image'
        elif 'pdf' in query_type:
            query_type = 'pdf'  
        else:
            query_type = 'both'
        
        stores_to_search = []
        if query_type in ['pdf', 'both']:
            stores_to_search.append(('pdf', self.vector_store_pdf))
        if query_type in ['image', 'both']:
            stores_to_search.append(('image', self.vector_store_image))
        
        all_results: dict[str, float] = {}

        for store_type, vector_store in stores_to_search:
            for variation in variations:
                if store_type == 'image':
                    inputs = self.model_manager.clip_processor(text=[variation], return_tensors="pt", padding=True)
                    with torch.no_grad():
                        query_embedding = self.model_manager.clip_model.get_text_features(**inputs)
                        query_embedding = query_embedding / query_embedding.norm(dim=-1, keepdim=True)
                        query_embedding = query_embedding.cpu().numpy()[0]
                else:  # pdf
                    query_embedding = self.model_manager.encode_text([variation])[0]
                
                vec_results = vector_store.search(query_embedding.reshape(1, -1), k=top_k)
                for doc_id, score in vec_results:
                    all_results[doc_id] = score
        
        sorted_ids = sorted(all_results.items(), key=lambda x: x[1], reverse=True)
        
        seen_paths = {}
        unique_docs = {}
        
        with MetadataStore() as store:
            for doc_id, score in sorted_ids:
                doc = store.get_document(doc_id)
                if doc:
                    path_str = str(doc.path)
                    
                    if path_str not in seen_paths:
                        seen_paths[path_str] = score
                        unique_docs[path_str] = doc
        
        results = []
        for path_str, score in seen_paths.items():
            doc = unique_docs[path_str]
            snippet = f"{doc.path.name}" if doc.doc_type == "image" else doc.content[:150]
            
            results.append(SearchResult(
                document=doc,
                score=score,
                source="hybrid",
                snippet=snippet
            ))
        
        return results
