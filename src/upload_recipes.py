"""
Upload recipes to Pinecone vector database
This script loads recipes from PDF and uploads them to Pinecone
"""

import os
from dotenv import load_dotenv
from pdf_loader import load_pdf_documents, split_into_chunks
from vector_store import initialize_vector_store, RecipeVectorStore

load_dotenv()

DATASET_PATH = os.getenv('DATASET_PATH')


def main():
    """Main pipeline: Load PDF -> Extract recipes -> Upload to Pinecone"""
    
    print("=" * 60)
    print("RECIPE RAG SYSTEM - UPLOAD PIPELINE")
    print("=" * 60)
    
    # Step 1: Load PDF documents
    print("\n[1/4] Loading PDF documents...")
    documents = load_pdf_documents(DATASET_PATH)
    print(f"✅ Loaded {len(documents)} pages from {DATASET_PATH}")
    
    # Step 2: Extract recipe chunks
    print("\n[2/4] Extracting recipe chunks...")
    recipe_chunks = split_into_chunks(documents)
    print(f"✅ Extracted {len(recipe_chunks)} recipe chunks")
    
    # Display sample recipe
    if recipe_chunks:
        sample = recipe_chunks[0]
        print(f"\nSample Recipe:")
        print(f"  Name: {sample['metadata']['recipe_name']}")
        print(f"  Type: {sample['metadata']['recipe_type']}")
        print(f"  Serving: {sample['serving_suggestion']}")
        print(f"  Ingredients (first 100 chars): {sample['ingredients'][:100]}...")
    
    # Step 3: Initialize vector store
    print("\n[3/4] Initializing Pinecone vector store...")
    try:
        vector_store = initialize_vector_store()
    except Exception as e:
        if "ALREADY_EXISTS" in str(e):
            print("Index already exists, connecting to existing index...")
            vector_store = RecipeVectorStore()
            vector_store.connect_to_index()
        else:
            raise e
    
    # Check current stats
    stats = vector_store.get_index_stats()
    current_count = stats['total_vector_count']
    print(f"Current vectors in database: {current_count}")
    
    # Step 4: Upload recipes
    print("\n[4/4] Uploading recipes to Pinecone...")
    
    # Ask for confirmation if database already has vectors
    if current_count > 0:
        print(f"\n⚠️  Warning: Database already contains {current_count} vectors")
        response = input("Do you want to proceed? This will add more vectors. (yes/no): ")
        if response.lower() != 'yes':
            print("Upload cancelled.")
            return
    
    vector_store.upload_recipes(recipe_chunks, batch_size=100)
    
    # Wait a bit more to ensure indexing is complete
    print("\n⏳ Waiting additional time for indexing to fully complete...")
    import time
    time.sleep(10)
    
    # Final stats
    print("\n" + "=" * 60)
    print("UPLOAD COMPLETE!")
    print("=" * 60)
    final_stats = vector_store.get_index_stats()
    print(f"Total recipes in database: {final_stats['total_vector_count']}")
    print(f"Dimension: {final_stats.get('dimension', 384)}")
    
    # Test search
    print("\n" + "=" * 60)
    print("TESTING SEARCH FUNCTIONALITY")
    print("=" * 60)
    
    test_queries = [
        "chicken recipes",
        "dessert with chocolate",
        "salad recipes"
    ]
    
    for query in test_queries:
        print(f"\nTest Query: '{query}'")
        results = vector_store.search_recipes(query, top_k=3)
        
        if results and results['matches']:
            print(f"Found {len(results['matches'])} matches:")
            for i, match in enumerate(results['matches'], 1):
                recipe_name = match['metadata'].get('recipe_name', 'Unknown')
                score = match['score']
                print(f"  {i}. {recipe_name} (score: {score:.4f})")
        else:
            print("  No matches found")
    
    print("\n✅ All done! Your RAG system is ready to use.")
    print("Next step: Create the main RAG interface to query recipes.")


if __name__ == "__main__":
    main()