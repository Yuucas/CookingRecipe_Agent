"""
Test script for Recipe RAG System
"""

from rag_system import RecipeRAG

def test_basic_query():
    """Test basic ingredient query"""
    print("\n" + "="*80)
    print("TEST 1: Basic Ingredient Query")
    print("="*80)
    
    rag = RecipeRAG()
    
    # Test query
    ingredients = "chicken, rice, garlic"
    print(f"\nQuery: What can I make with {ingredients}?")
    
    response = rag.query(ingredients=ingredients, top_k=3)
    
    print("\n" + "-"*80)
    print("RESPONSE:")
    print("-"*80)
    print(response)
    print("="*80)


def test_filtered_query():
    """Test query with recipe type filter"""
    print("\n" + "="*80)
    print("TEST 2: Filtered Query (Desserts)")
    print("="*80)
    
    rag = RecipeRAG()
    
    # Test query with filter
    ingredients = "chocolate, butter, flour"
    recipe_type = "DESSERT & BAKING"
    
    print(f"\nQuery: Show me desserts I can make with {ingredients}")
    
    response = rag.query(
        ingredients=ingredients,
        recipe_type=recipe_type,
        top_k=3
    )
    
    print("\n" + "-"*80)
    print("RESPONSE:")
    print("-"*80)
    print(response)
    print("="*80)


def test_additional_requirements():
    """Test query with additional requirements"""
    print("\n" + "="*80)
    print("TEST 3: Query with Additional Requirements")
    print("="*80)
    
    rag = RecipeRAG()
    
    # Test query with additional request
    ingredients = "beef, onion, tomato"
    additional = "I want something that takes less than 30 minutes"
    
    print(f"\nQuery: Recipes with {ingredients}")
    print(f"Requirement: {additional}")
    
    response = rag.query(
        ingredients=ingredients,
        additional_request=additional,
        top_k=3
    )
    
    print("\n" + "-"*80)
    print("RESPONSE:")
    print("-"*80)
    print(response)
    print("="*80)


if __name__ == "__main__":
    print("\n🧪 TESTING RECIPE RAG SYSTEM")
    print("="*80)
    
    # Run tests
    try:
        test_basic_query()
        input("\nPress Enter to continue to next test...")
        
        test_filtered_query()
        input("\nPress Enter to continue to next test...")
        
        test_additional_requirements()
        
        print("\n✅ All tests completed!")
        print("\nYou can now run: python rag_system.py")
        print("for interactive mode!\n")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")