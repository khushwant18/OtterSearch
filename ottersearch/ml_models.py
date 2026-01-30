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
        
        self.slm: AutoModelForCausalLM | None = None
        self.tokenizer: AutoTokenizer | None = None
        self.bge_tokenizer = AutoTokenizer.from_pretrained(config.bge_model)
        self.text_encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
        self.clip_model.eval()

        self._initialized = True
        self.slm, self.slm_tokenizer = self.get_slm()
    
    def encode_text(self, texts: list[str]) -> np.ndarray:
        return self.text_encoder.encode(
            texts, 
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True
        )
    
    
    def encode_images(self, images: list[object]) -> tuple[np.ndarray, list[int]]:
        """Returns (embeddings, successful_indices)"""
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
                        # Validate image
                        if img.size[0] < 10 or img.size[1] < 10:
                            print(f"⚠️  Skipping tiny image {p.name}")
                            continue
                        pil_images.append(img)
                        batch_indices.append(i + j)
                    except Exception as e:
                        print(f"⚠️  Skipping {p.name}: {e}")
                        continue
                
                if pil_images:
                    try:
                        inputs = self.clip_processor(images=pil_images, return_tensors="pt")
                        embeddings = self.clip_model.get_image_features(**inputs)
                        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                        all_embeddings.append(embeddings.cpu().numpy())
                        successful_indices.extend(batch_indices)
                    except Exception as e:
                        print(f"⚠️  Batch encoding failed: {e}")
                        continue
        
        if not all_embeddings:
            return np.array([]).reshape(0, config.image_vector_dim), []
        
        return np.vstack(all_embeddings), successful_indices
    
    def get_slm(self) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
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
    
    # def encode_text_batch(self, texts: list[str]) -> np.ndarray:
    #     # embeddings = model.encode(text=texts)
    #     text_inputs = self.bge_tokenizer(texts, return_tensors="pt", padding=True).to(self.model.device)
    #     embeddings = self.model.encode_text(text_inputs)

    #     return embeddings.cpu().numpy()


    def chunk_text_token_safe(
        self,
        text: str,
        max_tokens: int = 500,
        overlap: int = 50,
    ) -> list[str]:
        """
        Chunk text using tokenizer tokens so it NEVER exceeds model limits
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

            # overlap
            start = end - overlap
            if start < 0:
                start = 0

        return chunks
    
    def unload_slm(self):
        """Free SLM memory"""
        if self.slm is not None:
            del self.slm
            del self.tokenizer
            self.slm = None
            self.tokenizer = None
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            import gc
            gc.collect()
