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
    def __init__(self):
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
        variations = self.model_manager.generate_query_variations(query, n=3)
        print(f"Query variations: {variations}")
        query_type = [self._detect_query_type(variation) for variation in variations]
        if 'image' in query_type:
            query_type = 'image'
        elif 'pdf' in query_type:
            query_type = 'pdf'  
        else:
            query_type = 'both'
        print(f"Query type: {query_type}")
        
        # Choose which stores to search
        stores_to_search = []
        if query_type in ['pdf', 'both']:
            stores_to_search.append(('pdf', self.vector_store_pdf))
        if query_type in ['image', 'both']:
            stores_to_search.append(('image', self.vector_store_image))
        
        # Search selected stores
        all_results: dict[str, float] = {}
        
        # for store_type, vector_store in stores_to_search:
        #     print(f"Searching {store_type} index...") 
        #     for variation in variations:
        #         query_embedding = self.model_manager.encode_text([variation])[0]
        #         vec_results = vector_store.search(query_embedding.reshape(1, -1), k=top_k)
        #         for doc_id, score in vec_results:
        #             all_results[doc_id] = score

        for store_type, vector_store in stores_to_search:
            print(f"Searching {store_type} index...") 
            for variation in variations:
                # Use appropriate encoder based on store type
                if store_type == 'image':
                    # Encode text query with CLIP text encoder
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
        
        # Deduplicate by path - keep highest score per document
        seen_paths = {}
        unique_docs = {}
        
        with MetadataStore() as store:
            for doc_id, score in sorted_ids:
                doc = store.get_document(doc_id)
                if doc:
                    path_str = str(doc.path)
                    
                    # Only keep first occurrence (highest score due to sorting)
                    if path_str not in seen_paths:
                        seen_paths[path_str] = score
                        unique_docs[path_str] = doc
        
        # Build results from unique documents
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
