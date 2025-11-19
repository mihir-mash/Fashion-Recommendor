"""
Train and save outfit matching model based on CLIP embeddings and category matching.

This script:
1. Loads fashion dataset
2. Extracts CLIP embeddings
3. Builds FAISS index
4. Saves model as a dictionary (not a class instance) for easy loading
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    print("Warning: faiss not available. Will use numpy-based similarity search.")
    FAISS_AVAILABLE = False

try:
    from transformers import CLIPProcessor, CLIPModel
    CLIP_AVAILABLE = True
except ImportError:
    print("Error: transformers not installed. Install with: pip install transformers torch")
    sys.exit(1)

from config import Settings
from embeddings import load_meta

settings = Settings()


# Category mapping (from your Colab code)
CATEGORY_MAPPING = {
    'Shirts': 'tops', 'Tshirts': 'tops', 'Tops': 'tops', 'Sweatshirts': 'tops',
    'Sweaters': 'tops', 'Jackets': 'tops', 'Blazers': 'tops',
    'Jeans': 'bottoms', 'Trousers': 'bottoms', 'Shorts': 'bottoms', 'Skirts': 'bottoms',
    'Track Pants': 'bottoms', 'Leggings': 'bottoms',
    'Dresses': 'dresses', 'Kurta Sets': 'dresses', 'Gowns': 'dresses',
    'Watches': 'accessories', 'Bags': 'accessories', 'Belts': 'accessories',
    'Sunglasses': 'accessories', 'Jewellery': 'accessories', 'Scarves': 'accessories',
    'Shoes': 'shoes', 'Heels': 'shoes', 'Flats': 'shoes', 'Sandals': 'shoes',
    'Flip Flops': 'shoes', 'Casual Shoes': 'shoes'
}

# Complementary categories (from your Colab code)
COMPLEMENTARY_CATEGORIES = {
    "tops": ["bottoms", "shoes", "accessories"],
    "bottoms": ["tops", "shoes", "accessories"],
    "shoes": ["tops", "dresses"],
    "accessories": ["tops", "bottoms", "dresses"],
    "dresses": ["shoes", "accessories"]
}


class OutfitModel:
    """Outfit matching model using CLIP embeddings and category matching."""
    
    def __init__(self, embeddings, categories, item_ids, faiss_index=None):
        self.embeddings = embeddings  # numpy array (N, 768)
        self.categories = categories  # list of category strings
        self.item_ids = item_ids  # list of item IDs
        self.faiss_index = faiss_index
        self.clip_model = None
        self.clip_processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def load_clip(self):
        """Lazy load CLIP model (only when needed)."""
        if self.clip_model is None:
            print("Loading CLIP model...")
            model_name = "openai/clip-vit-large-patch14"
            self.clip_model = CLIPModel.from_pretrained(model_name).to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained(model_name)
            self.clip_model.eval()
    
    def get_embedding(self, image_path_or_pil):
        """Get CLIP embedding for an image."""
        self.load_clip()
        
        try:
            if isinstance(image_path_or_pil, str):
                img = Image.open(image_path_or_pil).convert("RGB")
            else:
                img = image_path_or_pil.convert("RGB")
            
            inputs = self.clip_processor(images=img, return_tensors="pt").to(self.device)
            with torch.no_grad():
                emb = self.clip_model.get_image_features(**inputs)
            emb = emb / emb.norm(p=2)
            return emb.cpu().numpy().flatten()
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return np.zeros(768)
    
    def predict_category(self, image_path_or_pil, top_k=10):
        """Predict category of an image by finding similar items."""
        query_emb = self.get_embedding(image_path_or_pil).reshape(1, -1)
        
        # Normalize
        query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-10)
        
        # Search for similar items
        if self.faiss_index is not None and FAISS_AVAILABLE:
            # Use FAISS
            query_emb_faiss = query_emb.astype('float32')
            faiss.normalize_L2(query_emb_faiss)
            similarities, indices = self.faiss_index.search(query_emb_faiss, top_k)
            similarities = similarities[0]
            indices = indices[0]
        else:
            # Use numpy cosine similarity
            similarities = (self.embeddings @ query_emb.T).flatten()
            indices = np.argsort(-similarities)[:top_k]
            similarities = similarities[indices]
        
        # Get most common category from top matches
        top_categories = [self.categories[i] for i in indices]
        from collections import Counter
        most_common = Counter(top_categories).most_common(1)
        return most_common[0][0] if most_common else "tops"
    
    def predict_matches(self, embedding, target_type=None, k=6):
        """
        Predict matching items for an outfit.
        
        Args:
            embedding: numpy array of shape (768,) - image embedding
            target_type: optional target category to filter by
            k: number of matches to return
        
        Returns:
            list of tuples: [(item_id, score), ...]
        """
        # Normalize embedding
        embedding = embedding.reshape(1, -1)
        embedding = embedding / (np.linalg.norm(embedding) + 1e-10)
        embedding = embedding.astype('float32')
        
        # Search for similar items
        if self.faiss_index is not None and FAISS_AVAILABLE:
            faiss.normalize_L2(embedding)
            similarities, indices = self.faiss_index.search(embedding, min(k * 10, len(self.embeddings)))
            similarities = similarities[0]
            indices = indices[0]
        else:
            # Use numpy cosine similarity
            similarities = (self.embeddings @ embedding.T).flatten()
            indices = np.argsort(-similarities)[:min(k * 10, len(self.embeddings))]
            similarities = similarities[indices]
        
        # Filter by target_type if provided
        results = []
        for sim, idx in zip(similarities, indices):
            item_id = str(self.item_ids[idx])
            category = self.categories[idx]
            
            # If target_type specified, only include matching categories
            if target_type:
                complementary = COMPLEMENTARY_CATEGORIES.get(target_type.lower(), [])
                if category.lower() not in [c.lower() for c in complementary]:
                    continue
            
            results.append((item_id, float(sim)))
            if len(results) >= k:
                break
        
        return results


def train_outfit_model(meta_path=None, embeddings_path=None, output_path=None, max_items=800):
    """
    Train outfit matching model from metadata and embeddings.
    
    Args:
        meta_path: path to metadata CSV
        embeddings_path: path to embeddings.npy file
        output_path: where to save the model
        max_items: maximum number of items to use (for faster training)
    """
    print("=" * 60)
    print("Training Outfit Matching Model")
    print("=" * 60)
    
    # Load metadata
    meta_path = meta_path or settings.META_PATH
    print(f"Loading metadata from {meta_path}...")
    df = load_meta(meta_path)
    
    if df.empty:
        raise ValueError("Metadata is empty!")
    
    print(f"Loaded {len(df)} items")
    
    # Load embeddings
    embeddings_path = embeddings_path or settings.EMBEDDINGS_PATH
    print(f"Loading embeddings from {embeddings_path}...")
    embeddings = np.load(embeddings_path)
    print(f"Loaded embeddings shape: {embeddings.shape}")
    
    # Ensure embeddings match metadata length
    min_len = min(len(df), len(embeddings))
    df = df.iloc[:min_len].reset_index(drop=True)
    embeddings = embeddings[:min_len]
    
    # Map categories
    print("Mapping categories...")
    if 'articleType' in df.columns:
        df['category'] = df['articleType'].map(CATEGORY_MAPPING)
    elif 'subCategory' in df.columns:
        df['category'] = df['subCategory'].map(CATEGORY_MAPPING)
    else:
        print("Warning: No articleType or subCategory column found. Using default category.")
        df['category'] = 'tops'
    
    # Filter out items without category mapping
    df = df[df['category'].notna()].reset_index(drop=True)
    embeddings = embeddings[:len(df)]
    
    # Sample if needed
    if len(df) > max_items:
        print(f"Sampling {max_items} items from {len(df)}...")
        sample_idx = np.random.choice(len(df), max_items, replace=False)
        df = df.iloc[sample_idx].reset_index(drop=True)
        embeddings = embeddings[sample_idx]
    
    print(f"Final dataset size: {len(df)}")
    print(f"Category distribution:\n{df['category'].value_counts()}")
    
    # Normalize embeddings
    print("Normalizing embeddings...")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-10)
    
    # Build FAISS index
    faiss_index = None
    if FAISS_AVAILABLE:
        print("Building FAISS index...")
        faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
        embeddings_faiss = embeddings.astype('float32')
        faiss_index.add(embeddings_faiss)
        print("FAISS index ready!")
    else:
        print("Using numpy-based similarity search (FAISS not available)")
    
    # Get item IDs and categories
    item_ids = df['id'].astype(str).tolist() if 'id' in df.columns else [str(i) for i in range(len(df))]
    categories = df['category'].tolist()
    
    # Create model
    model = OutfitModel(
        embeddings=embeddings,
        categories=categories,
        item_ids=item_ids,
        faiss_index=faiss_index
    )
    
    # Save model as dictionary (not class instance) for easy loading
    print(f"Saving model to {output_path}...")
    output_path = output_path or settings.OUTFIT_MODEL_PATH
    
    # Resolve path
    output_path = Path(output_path)
    if not output_path.is_absolute():
        output_path = (Path(__file__).parent.parent / output_path).resolve()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as dictionary with all necessary data
    model_dict = {
        'embeddings': embeddings,
        'categories': categories,
        'item_ids': item_ids,
        'complementary_categories': COMPLEMENTARY_CATEGORIES,
        'category_mapping': CATEGORY_MAPPING,
        'has_faiss': faiss_index is not None,
    }
    
    # Save FAISS index separately if available
    if faiss_index is not None:
        faiss_index_path = output_path.parent / 'outfit_faiss.index'
        faiss.write_index(faiss_index, str(faiss_index_path))
        model_dict['faiss_index_path'] = str(faiss_index_path)
    
    joblib.dump(model_dict, output_path)
    print(f"✅ Model saved to {output_path}")
    print(f"   - Embeddings: {embeddings.shape}")
    print(f"   - Items: {len(item_ids)}")
    print(f"   - Categories: {len(set(categories))}")
    
    return model


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train outfit matching model')
    parser.add_argument('--meta', default=None, help='Path to metadata CSV')
    parser.add_argument('--embeddings', default=None, help='Path to embeddings.npy')
    parser.add_argument('--output', default=None, help='Output path for model')
    parser.add_argument('--max-items', type=int, default=800, help='Maximum items to use')
    args = parser.parse_args()
    
    train_outfit_model(
        meta_path=args.meta,
        embeddings_path=args.embeddings,
        output_path=args.output,
        max_items=args.max_items
    )

