"""
Machine Learning model management for OtterSearch
"""
import torch
import numpy as np
import gc
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
from .config import config


class ModelManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        # Set device: MPS for Apple Silicon, CPU otherwise
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using Apple Silicon (MPS) device")
        else:
            self.device = torch.device("cpu")
            print("Using CPU device")
        
        self.slm: AutoModelForCausalLM | None = None
        self.tokenizer: AutoTokenizer | None = None
        self.bge_tokenizer = AutoTokenizer.from_pretrained(config.bge_model)
        self.text_encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
        self.clip_model.eval()

        self._initialized = True
        self.slm, self.slm_tokenizer = self.get_slm()
    
    def encode_text(self, texts: list[str]) -> np.ndarray:
        """Encode text strings into embeddings using SentenceTransformer.
        
        Args:
            texts: List of text strings to encode
            
        Returns:
            numpy array of shape (len(texts), 384) containing text embeddings
            
        Error Handling:
            - Empty text strings are encoded as-is (model handles them)
            - Model errors propagate to caller
            - No validation of input text length
        """
        return self.text_encoder.encode(
            texts, 
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True
        )
    
    
    def encode_images(self, images: list[object]) -> tuple[np.ndarray, list[int]]:
        """Encode image files into CLIP embeddings.
        
        Args:
            images: List of Path objects pointing to image files
            
        Returns:
            Tuple of (embeddings, successful_indices):
                - embeddings: numpy array of shape (N, 512) where N <= len(images)
                - successful_indices: List of original indices that were successfully encoded
                
        Error Handling:
            - Images smaller than 10x10 pixels are skipped
            - Image loading errors (corrupt files, unsupported formats) are caught and skipped
            - Batch encoding errors skip entire batch but continue with next batch
            - Returns empty array and empty list if all images fail
            - No error propagation - all exceptions silently handled
        """
        batch_size = 16
        all_embeddings = []
        successful_indices = []
        
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch_paths = images[i:i + batch_size]
                
                pil_images = []
                batch_indices = []
                for j, p in enumerate(batch_paths):
                    try:
                        img = Image.open(p).convert("RGB")
                        if img.size[0] < 10 or img.size[1] < 10:
                            continue
                        pil_images.append(img)
                        batch_indices.append(i + j)
                    except Exception as e:
                        continue
                
                if pil_images:
                    try:
                        inputs = self.clip_processor(images=pil_images, return_tensors="pt")
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        embeddings = self.clip_model.get_image_features(**inputs)
                        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                        all_embeddings.append(embeddings.cpu().numpy())
                        successful_indices.extend(batch_indices)
                    except Exception as e:
                        continue
        
        if not all_embeddings:
            return np.array([]).reshape(0, config.image_vector_dim), []
        
        return np.vstack(all_embeddings), successful_indices
    
    def get_slm(self) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Get or load the Small Language Model and its tokenizer.
        
        Lazily loads the SLM on first call. Subsequent calls return cached instances.
        
        Returns:
            Tuple of (model, tokenizer)
            
        Error Handling:
            - Model loading errors propagate to caller
            - Requires ~300MB disk space for model download
            - May fail if insufficient GPU/CPU memory
            - Network errors during download are not handled
        """
        if self.slm is None:
            self.slm = AutoModelForCausalLM.from_pretrained(
                config.slm_model,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                cache_dir=str(config.data_dir / "models")
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.slm_model,
                cache_dir=str(config.data_dir / "models")
            )
        return self.slm, self.tokenizer
    
    def generate_query_variations(self, query: str, n: int = None) -> list[str]:
        """Generate synonym variations of a search query using SLM.
        
        Args:
            query: Original search query string
            n: Number of variations to generate (default: config.num_query_variations)
            
        Returns:
            List containing original query plus up to n variations.
            First element is always the original query.
            
        Error Handling:
            - SLM loading errors propagate to caller
            - If SLM generates fewer than n variations, returns what's available
            - Lines starting with 'Generate', '#', or '-' are filtered out
            - Empty/whitespace-only lines are skipped
            - Model inference errors propagate to caller
        """
        if self.slm is None:
            self.slm, self.tokenizer = self.get_slm()
            
        n = n or config.num_query_variations
        
        
        prompt = f"""Generate {n} synonym variations that mean EXACTLY the same thing as: '{query}'

Rules:
- Use only synonyms with identical meaning
- Keep the same domain
- One variation per line
- No numbering, no explanations
- Output ONLY the synonym phrases"""

        system = "Generate search query synonyms. Preserve exact meaning and domain. No creative reinterpretation."

        input_ids = self.slm_tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
            tokenize=True,
        ).to(self.slm.device)

        output = self.slm.generate(
            input_ids,
            do_sample=True,
            temperature=0.7,
            min_p=0.15,
            repetition_penalty=1.05,
            max_new_tokens=200,
        )
        
        response = self.slm_tokenizer.decode(output[0][len(input_ids[0]):], skip_special_tokens=True)
        
        variations = [
            line.strip() 
            for line in response.strip().split("\n")
            if line.strip() and not line.startswith(("Generate", "#", "-"))
        ][:n]
        
        return [query] + variations

    def chunk_text_token_safe(
        self,
        text: str,
        max_tokens: int = 500,
        overlap: int = 50,
    ) -> list[str]:
        """Chunk text into token-safe segments for model processing.
        
        Uses BGE tokenizer to ensure chunks never exceed model token limits.
        
        Args:
            text: Text string to chunk
            max_tokens: Maximum tokens per chunk
            overlap: Number of overlapping tokens between consecutive chunks
            
        Returns:
            List of text chunks, each guaranteed to be <= max_tokens
            
        Error Handling:
            - Empty text returns empty list after one iteration
            - Tokenization errors propagate to caller
            - Very short text may return single chunk
        """
        input_ids = self.bge_tokenizer(
            text,
            add_special_tokens=False
        )["input_ids"]

        chunks = []
        start = 0

        while start < len(input_ids):
            end = start + max_tokens
            chunk_ids = input_ids[start:end]

            chunk_text = self.bge_tokenizer.decode(
                chunk_ids,
                skip_special_tokens=True
            )
            chunks.append(chunk_text)

            start = end - overlap
            if start < 0:
                start = 0

        return chunks
    
    def unload_slm(self):
        """Free SLM memory and trigger garbage collection.
        
        Deletes model and tokenizer instances, clears CUDA cache if available,
        and forces Python garbage collection to reclaim memory.
        
        Error Handling:
            - Safe to call multiple times (checks if slm is None)
            - CUDA cache clearing fails silently on non-CUDA systems
        """
        if self.slm is not None:
            del self.slm
            del self.tokenizer
            self.slm = None
            self.tokenizer = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
            import gc
            gc.collect()
