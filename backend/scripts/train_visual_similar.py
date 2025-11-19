"""
Train and save visual similarity model based on ResNet50 embeddings.

This script:
1. Loads fashion dataset images
2. Extracts ResNet50 embeddings
3. Builds FAISS index for fast similarity search
4. Saves model as a dictionary for easy loading
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from numpy.linalg import norm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    print("Warning: faiss not available. Will use numpy-based similarity search.")
    FAISS_AVAILABLE = False

from config import Settings
from embeddings import load_meta

settings = Settings()


class VisualSimilarityModel:
    """Visual similarity model using ResNet50 embeddings."""
    
    def __init__(self, embeddings, item_ids, image_paths, faiss_index=None):
        """
        Initialize visual similarity model.
        
        Args:
            embeddings: numpy array (N, 2048) - ResNet50 feature vectors
            item_ids: list of item IDs
            image_paths: list of image file paths
            faiss_index: optional FAISS index for fast search
        """
        self.embeddings = embeddings
        self.item_ids = item_ids
        self.image_paths = image_paths
        self.faiss_index = faiss_index
        self.model = None
    
    def load_model(self):
        """Lazy load ResNet50 model (only when needed for new predictions)."""
        if self.model is None:
            print("Loading ResNet50 model...")
            base_model = ResNet50(weights='imagenet', include_top=False, 
                                 input_shape=(224, 224, 3))
            base_model.trainable = False
            self.model = tf.keras.Sequential([
                base_model,
                GlobalMaxPooling2D()
            ])
            print("ResNet50 model loaded successfully.")
    
    def extract_features(self, img_path_or_pil):
        """
        Extract normalized features from an image.
        
        Args:
            img_path_or_pil: path to image file or PIL Image object
        
        Returns:
            numpy array of shape (2048,) - normalized feature vector
        """
        self.load_model()
        
        try:
            # Load and preprocess image
            if isinstance(img_path_or_pil, str):
                img = keras_image.load_img(img_path_or_pil, target_size=(224, 224))
            else:
                img = img_path_or_pil.resize((224, 224))
            
            img_array = keras_image.img_to_array(img)
            expanded_img_array = np.expand_dims(img_array, axis=0)
            preprocessed_img = preprocess_input(expanded_img_array)
            
            # Extract features
            result = self.model.predict(preprocessed_img, verbose=0).flatten()
            
            # Normalize for better distance calculation
            normalized_result = result / (norm(result) + 1e-10)
            return normalized_result
        
        except Exception as e:
            print(f"Error extracting features: {e}")
            return np.zeros(2048)
    
    def find_similar(self, image_path_or_embedding, k=6, return_distances=False):
        """
        Find visually similar images.
        
        Args:
            image_path_or_embedding: path to query image, PIL Image, or embedding vector
            k: number of similar images to return
            return_distances: whether to return similarity scores
        
        Returns:
            list of tuples: [(item_id, image_path, [distance]), ...]
        """
        # Get query embedding
        if isinstance(image_path_or_embedding, np.ndarray):
            query_emb = image_path_or_embedding
        else:
            query_emb = self.extract_features(image_path_or_embedding)
        
        # Normalize
        query_emb = query_emb.reshape(1, -1)
        query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-10)
        query_emb = query_emb.astype('float32')
        
        # Search for similar items
        if self.faiss_index is not None and FAISS_AVAILABLE:
            faiss.normalize_L2(query_emb)
            distances, indices = self.faiss_index.search(query_emb, k + 1)
            distances = distances[0]
            indices = indices[0]
        else:
            # Use numpy cosine similarity
            similarities = (self.embeddings @ query_emb.T).flatten()
            indices = np.argsort(-similarities)[:k + 1]
            distances = 1 - similarities[indices]  # Convert similarity to distance
        
        # Filter out the query image itself (distance ~0)
        results = []
        for dist, idx in zip(distances, indices):
            if dist < 0.001:  # Skip the exact match (query itself)
                continue
            
            item_id = str(self.item_ids[idx])
            image_path = self.image_paths[idx]
            
            if return_distances:
                # Convert distance to similarity score (0-1)
                similarity = 1 - float(dist)
                results.append((item_id, image_path, similarity))
            else:
                results.append((item_id, image_path))
            
            if len(results) >= k:
                break
        
        return results
    
    def predict_similar(self, embedding, k=6):
        """
        Predict similar items (for API compatibility).
        
        Args:
            embedding: numpy array of shape (2048,) - image embedding
            k: number of matches to return
        
        Returns:
            list of tuples: [(item_id, similarity_score), ...]
        """
        results = self.find_similar(embedding, k=k, return_distances=True)
        return [(item_id, score) for item_id, _, score in results]


def extract_features_batch(image_paths, model, batch_size=32):
    """
    Extract features from multiple images efficiently.
    
    Args:
        image_paths: list of image file paths
        model: ResNet50 feature extraction model
        batch_size: number of images to process at once
    
    Returns:
        numpy array of shape (N, 2048)
    """
    features = []
    
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting Features"):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = []
        
        for img_path in batch_paths:
            try:
                img = keras_image.load_img(img_path, target_size=(224, 224))
                img_array = keras_image.img_to_array(img)
                batch_images.append(img_array)
            except Exception as e:
                # Use zero vector for corrupted images
                batch_images.append(np.zeros((224, 224, 3)))
        
        # Process batch
        batch_array = np.array(batch_images)
        preprocessed_batch = preprocess_input(batch_array)
        batch_features = model.predict(preprocessed_batch, verbose=0)
        
        # Normalize each feature vector
        for feature in batch_features:
            normalized = feature / (norm(feature) + 1e-10)
            features.append(normalized)
    
    return np.array(features)


def train_visual_similarity_model(
    images_dir=None, 
    meta_path=None, 
    output_path=None, 
    max_images=5000,
    batch_size=32
):
    """
    Train visual similarity model from image directory.
    
    Args:
        images_dir: directory containing images
        meta_path: optional path to metadata CSV
        output_path: where to save the model
        max_images: maximum number of images to process
        batch_size: batch size for feature extraction
    """
    print("=" * 60)
    print("Training Visual Similarity Model")
    print("=" * 60)
    
    # Set default paths
    images_dir = images_dir or settings.META_PATH
    output_path = output_path or settings.IMAGE_MODEL_PATH
    
    # Resolve paths
    images_dir = Path(images_dir)
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    
    # Find all image files
    print(f"Scanning {images_dir} for images...")
    image_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(list(images_dir.glob(f'*{ext}')))
    
    print(f"Found {len(image_paths)} images")
    
    # Limit number of images
    if len(image_paths) > max_images:
        print(f"Sampling {max_images} images from {len(image_paths)}...")
        np.random.seed(42)
        sample_idx = np.random.choice(len(image_paths), max_images, replace=False)
        image_paths = [image_paths[i] for i in sorted(sample_idx)]
    
    # Convert to strings
    image_paths = [str(p) for p in image_paths]
    
    # Load metadata if available
    item_ids = None
    if meta_path:
        try:
            print(f"Loading metadata from {meta_path}...")
            df = load_meta(meta_path)
            if not df.empty and 'id' in df.columns:
                # Match image paths to item IDs
                item_ids = []
                for img_path in image_paths:
                    img_name = Path(img_path).stem
                    # Try to find matching ID in metadata
                    match = df[df['id'].astype(str) == img_name]
                    if not match.empty:
                        item_ids.append(str(match.iloc[0]['id']))
                    else:
                        item_ids.append(img_name)
        except Exception as e:
            print(f"Warning: Could not load metadata: {e}")
    
    # Use image names as IDs if no metadata
    if item_ids is None:
        item_ids = [Path(p).stem for p in image_paths]
    
    print(f"Processing {len(image_paths)} images...")
    
    # Load ResNet50 model
    print("Loading ResNet50 model...")
    base_model = ResNet50(weights='imagenet', include_top=False, 
                         input_shape=(224, 224, 3))
    base_model.trainable = False
    feature_model = tf.keras.Sequential([
        base_model,
        GlobalMaxPooling2D()
    ])
    print("Model loaded successfully.")
    
    # Extract features
    embeddings = extract_features_batch(image_paths, feature_model, batch_size=batch_size)
    print(f"Extracted features shape: {embeddings.shape}")
    
    # Build FAISS index
    faiss_index = None
    if FAISS_AVAILABLE:
        print("Building FAISS index...")
        faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
        embeddings_faiss = embeddings.astype('float32')
        faiss.normalize_L2(embeddings_faiss)
        faiss_index.add(embeddings_faiss)
        print("FAISS index ready!")
    else:
        print("Using numpy-based similarity search (FAISS not available)")
    
    # Create model
    model = VisualSimilarityModel(
        embeddings=embeddings,
        item_ids=item_ids,
        image_paths=image_paths,
        faiss_index=faiss_index
    )
    
    # Save model
    print(f"Saving model to {output_path}...")
    output_path = Path(output_path)
    if not output_path.is_absolute():
        output_path = (Path(__file__).parent.parent / output_path).resolve()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as dictionary
    model_dict = {
        'embeddings': embeddings,
        'item_ids': item_ids,
        'image_paths': image_paths,
        'has_faiss': faiss_index is not None,
        'embedding_dim': embeddings.shape[1]
    }
    
    # Save FAISS index separately if available
    if faiss_index is not None:
        faiss_index_path = output_path.parent / 'visual_faiss.index'
        faiss.write_index(faiss_index, str(faiss_index_path))
        model_dict['faiss_index_path'] = str(faiss_index_path)
    
    joblib.dump(model_dict, output_path)
    print(f"✅ Model saved to {output_path}")
    print(f"   - Embeddings: {embeddings.shape}")
    print(f"   - Images: {len(image_paths)}")
    print(f"   - Feature dimension: {embeddings.shape[1]}")
    
    return model


def load_visual_similarity_model(model_path=None):
    """
    Load a trained visual similarity model.
    
    Args:
        model_path: path to saved model file
    
    Returns:
        VisualSimilarityModel instance
    """
    model_path = model_path or settings.VISUAL_MODEL_PATH
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    print(f"Loading visual similarity model from {model_path}...")
    model_dict = joblib.load(model_path)
    
    # Load FAISS index if available
    faiss_index = None
    if model_dict.get('has_faiss') and FAISS_AVAILABLE:
        faiss_index_path = model_dict.get('faiss_index_path')
        if faiss_index_path and Path(faiss_index_path).exists():
            faiss_index = faiss.read_index(faiss_index_path)
            print("FAISS index loaded.")
    
    # Create model instance
    model = VisualSimilarityModel(
        embeddings=model_dict['embeddings'],
        item_ids=model_dict['item_ids'],
        image_paths=model_dict['image_paths'],
        faiss_index=faiss_index
    )
    
    print(f"✅ Model loaded: {len(model.item_ids)} items")
    return model


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train visual similarity model')
    parser.add_argument('--images-dir', default=None, help='Directory containing images')
    parser.add_argument('--meta', default=None, help='Path to metadata CSV (optional)')
    parser.add_argument('--output', default=None, help='Output path for model')
    parser.add_argument('--max-images', type=int, default=5000, help='Maximum images to process')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for extraction')
    args = parser.parse_args()
    
    train_visual_similarity_model(
        images_dir=args.images_dir,
        meta_path=args.meta,
        output_path=args.output,
        max_images=args.max_images,
        batch_size=args.batch_size
    )