import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import time

load_dotenv()

# Load environment variables
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
INDEX_NAME = "recipe-index"

# Initialize embedding model
print("Loading embedding model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Embedding model loaded successfully!")


class RecipeVectorStore:
    """Handles all vector database operations for recipes"""
    
    def __init__(self, index_name: str = INDEX_NAME):
        """Initialize Pinecone connection and index"""
        self.index_name = index_name
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index = None
        
    def create_index(self, dimension: int = 384, metric: str = "cosine"):
        """
        Create a new Pinecone index if it doesn't exist
        
        Args:
            dimension: Embedding dimension (384 for all-MiniLM-L6-v2)
            metric: Distance metric (cosine, euclidean, or dotproduct)
        """
        try:
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            if self.index_name in existing_indexes:
                print(f"Index '{self.index_name}' already exists. Skipping creation.")
            else:
                print(f"Creating new index '{self.index_name}'...")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=dimension,
                    metric=metric,
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"  
                    )
                )
                print(f"Index '{self.index_name}' created successfully!")
                
                # Wait for index to be ready
                print("Waiting for index to be ready...")
                time.sleep(10)
        except Exception as e:
            if "ALREADY_EXISTS" in str(e):
                print(f"Index '{self.index_name}' already exists (confirmed via error). Continuing...")
            else:
                raise e
    
    def connect_to_index(self):
        """Connect to existing Pinecone index"""
        try:
            self.index = self.pc.Index(self.index_name)
            stats = self.index.describe_index_stats()
            print(f"Connected to index '{self.index_name}'")
            print(f"Index stats: {stats['total_vector_count']} vectors")
            return True
        except Exception as e:
            print(f"Error connecting to index: {e}")
            return False
    
    def create_searchable_text(self, recipe: Dict) -> str:
        """
        Create optimized searchable text from recipe for embedding
        
        Args:
            recipe: Recipe chunk dictionary
            
        Returns:
            Combined searchable text
        """
        parts = []
        
        # Recipe name 
        recipe_name = recipe['metadata'].get('recipe_name', recipe['recipe_header'])
        parts.append(f"{recipe_name} {recipe_name}")
        
        # Recipe type
        recipe_type = recipe['metadata'].get('recipe_type', '')
        if recipe_type:
            parts.append(recipe_type)
        
        # Ingredients 
        ingredients = recipe.get('ingredients', '')
        if ingredients:
            # Clean ingredients text
            ingredients_clean = ingredients.replace('\n', ' ').strip()
            parts.append(ingredients_clean)
        
        # Method
        method = recipe.get('method', '')
        if method:
            method_preview = method[:500].replace('\n', ' ').strip()
            parts.append(method_preview)
        
        # Chef tips if available
        chef_tip = recipe.get('chef_tip', '')
        if chef_tip:
            parts.append(chef_tip)
        
        return ' '.join(parts)
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector for given text
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        embedding = embedding_model.encode(text)
        return embedding.tolist()
    
    def upload_recipes(self, recipe_chunks: List[Dict], batch_size: int = 10):
        """
        Upload recipe chunks to Pinecone vector database
        
        Args:
            recipe_chunks: List of recipe chunk dictionaries
            batch_size: Number of vectors to upload per batch
        """
        if not self.index:
            print("Error: Not connected to index. Call connect_to_index() first.")
            return
        
        print(f"\nStarting upload of {len(recipe_chunks)} recipes...")
        
        vectors_to_upsert = []
        successful_uploads = 0
        
        for i, recipe in enumerate(recipe_chunks):
            try:
                # Create searchable text
                searchable_text = self.create_searchable_text(recipe)
                
                # Generate embedding
                embedding = self.generate_embedding(searchable_text)
                
                # Prepare metadata (Pinecone has metadata size limits)
                metadata = {
                    "recipe_name": recipe['metadata'].get('recipe_name', '')[:200],
                    "recipe_type": recipe['metadata'].get('recipe_type', '')[:100],
                    "serving_suggestion": recipe.get('serving_suggestion', '')[:100],
                    "ingredients": recipe.get('ingredients', '')[:2000],  # Truncate if too long
                    "method": recipe.get('method', '')[:2000],
                    "chef_tip": recipe.get('chef_tip', '')[:500],
                    "page": int(recipe['metadata'].get('page', 0)),
                    "page_label": recipe['metadata'].get('page_label', ''),
                    "source": recipe['metadata'].get('source', '')
                }
                
                # Create vector ID
                vector_id = f"recipe_page_{recipe['metadata'].get('page', i)}"
                
                # Add to batch
                vectors_to_upsert.append({
                    "id": vector_id,
                    "values": embedding,
                    "metadata": metadata
                })
                
                # Upload batch when it reaches batch_size
                if len(vectors_to_upsert) >= batch_size:
                    self.index.upsert(vectors=vectors_to_upsert)
                    successful_uploads += len(vectors_to_upsert)
                    print(f"Uploaded batch: {successful_uploads}/{len(recipe_chunks)} recipes")
                    vectors_to_upsert = []
                
            except Exception as e:
                print(f"Error processing recipe {i}: {e}")
                continue
        
        # Upload remaining vectors
        if vectors_to_upsert:
            self.index.upsert(vectors=vectors_to_upsert)
            successful_uploads += len(vectors_to_upsert)
        
        print(f"\n✅ Upload complete! Successfully uploaded {successful_uploads}/{len(recipe_chunks)} recipes")
        
        # Wait for indexing to complete
        print("\n⏳ Waiting for Pinecone to index vectors (this may take 10-30 seconds)...")
        time.sleep(15)
        
        # Verify upload
        stats = self.index.describe_index_stats()
        print(f"✅ Verification: {stats['total_vector_count']} vectors now in database")
        
        return successful_uploads
    
    def search_recipes(self, query: str, top_k: int = 5, filter_dict: Dict = None):
        """
        Search for recipes based on query text
        
        Args:
            query: Search query (e.g., "chicken and rice recipes")
            top_k: Number of results to return
            filter_dict: Optional metadata filters
            
        Returns:
            Search results with scores
        """
        if not self.index:
            print("Error: Not connected to index. Call connect_to_index() first.")
            return None
        
        # Generate query embedding
        query_embedding = self.generate_embedding(query)
        
        # Search
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict
        )
        
        return results
    
    def delete_all_vectors(self):
        """Delete all vectors from the index (use with caution!)"""
        if not self.index:
            print("Error: Not connected to index.")
            return
        
        confirm = input(f"Are you sure you want to delete all vectors from '{self.index_name}'? (yes/no): ")
        if confirm.lower() == 'yes':
            self.index.delete(delete_all=True)
            print("All vectors deleted!")
        else:
            print("Deletion cancelled.")
    
    def get_index_stats(self):
        """Get statistics about the index"""
        if not self.index:
            print("Error: Not connected to index.")
            return None
        
        stats = self.index.describe_index_stats()
        return stats


def initialize_vector_store():
    """
    Initialize and set up the vector store
    
    Returns:
        RecipeVectorStore instance
    """
    vector_store = RecipeVectorStore()
    vector_store.create_index()
    vector_store.connect_to_index()
    return vector_store


if __name__ == "__main__":
    # Test the vector store setup
    print("Testing Vector Store Setup...")
    print("=" * 50)
    
    # Initialize
    vector_store = initialize_vector_store()
    
    # Get stats
    stats = vector_store.get_index_stats()
    print(f"\nCurrent index stats:")
    print(f"Total vectors: {stats['total_vector_count']}")
    print(f"Dimension: {stats.get('dimension', 'N/A')}")
    
    # Test embedding generation
    test_text = "chicken rice garlic onion"
    test_embedding = vector_store.generate_embedding(test_text)
    print(f"\nTest embedding generated successfully!")
    print(f"Embedding dimension: {len(test_embedding)}")
    print(f"First 5 values: {test_embedding[:5]}")
    
    print("\n✅ Vector store is ready to use!")
    print("\nNext step: Run the upload script to add your recipes to the vector database.")
