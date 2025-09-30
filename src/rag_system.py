"""
Recipe RAG System - Main Interface
Combines vector search with Claude AI for intelligent recipe recommendations
"""

import os
from dotenv import load_dotenv
from vector_store import RecipeVectorStore
from anthropic import Anthropic

load_dotenv()

# Initialize clients
ANTHROPIC_API_KEY = os.getenv('CLAUDE_API_KEY')
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)


class RecipeRAG:
    """Main RAG system for recipe recommendations"""
    
    def __init__(self):
        """Initialize the RAG system"""
        print("Initializing Recipe RAG System...")
        
        # Connect to vector store
        self.vector_store = RecipeVectorStore()
        self.vector_store.connect_to_index()
        
        # Verify connection
        stats = self.vector_store.get_index_stats()
        print(f"✅ Connected to vector database with {stats['total_vector_count']} recipes")
        
        if stats['total_vector_count'] == 0:
            print("⚠️  Warning: No recipes found in database. Run upload_recipes.py first!")
    
    def search_recipes(self, ingredients: str, top_k: int = 5, recipe_type: str = None):
        """
        Search for recipes based on ingredients
        
        Args:
            ingredients: Comma-separated ingredients (e.g., "chicken, rice, onion")
            top_k: Number of recipes to retrieve
            recipe_type: Optional filter by recipe type
            
        Returns:
            Search results with metadata
        """
        # Build filter if recipe_type specified
        filter_dict = None
        if recipe_type:
            filter_dict = {"recipe_type": {"$eq": recipe_type}}
        
        # Search vector database
        results = self.vector_store.search_recipes(
            query=ingredients,
            top_k=top_k,
            filter_dict=filter_dict
        )
        
        return results
    
    def format_recipes_for_context(self, search_results):
        """
        Format retrieved recipes for Claude's context
        
        Args:
            search_results: Results from vector search
            
        Returns:
            Formatted string with recipe details
        """
        if not search_results or not search_results['matches']:
            return "No matching recipes found."
        
        formatted_recipes = []
        
        for i, match in enumerate(search_results['matches'], 1):
            metadata = match['metadata']
            score = match['score']
            
            recipe_text = f"""
Recipe {i}: {metadata['recipe_name']}
Type: {metadata['recipe_type']}
Serving: {metadata['serving_suggestion']}
Relevance Score: {score:.2f}

INGREDIENTS:
{metadata['ingredients']}

METHOD:
{metadata['method'][:500]}{'...' if len(metadata['method']) > 500 else ''}
"""
            
            if metadata.get('chef_tip'):
                recipe_text += f"\nCHEF'S TIP: {metadata['chef_tip']}\n"
            
            recipe_text += "\n" + "="*80 + "\n"
            formatted_recipes.append(recipe_text)
        
        return "\n".join(formatted_recipes)
    
    def create_prompt(self, user_ingredients: str, retrieved_recipes: str, user_query: str = None):
        """
        Create the prompt for Claude
        
        Args:
            user_ingredients: Ingredients the user has
            retrieved_recipes: Formatted recipes from vector search
            user_query: Optional additional user request
            
        Returns:
            Formatted prompt string
        """
        base_prompt = f"""You are a helpful cooking assistant with access to a recipe database. 

USER'S AVAILABLE INGREDIENTS:
{user_ingredients}

RELEVANT RECIPES FROM DATABASE:
{retrieved_recipes}

Based on the user's available ingredients and the retrieved recipes above, provide helpful recipe recommendations. 

For each recommended recipe:
1. Explain why it's a good match for their ingredients
2. List which ingredients they already have
3. List any missing ingredients (if any)
4. Provide cooking tips or substitution suggestions if relevant

Be conversational and friendly. If none of the recipes are a perfect match, suggest the closest options and explain what additional ingredients they would need.
"""
        
        if user_query:
            base_prompt += f"\n\nADDITIONAL USER REQUEST:\n{user_query}\n"
        
        return base_prompt
    
    def generate_response(self, prompt: str, max_tokens: int = 2000):
        """
        Generate response using Claude
        
        Args:
            prompt: The complete prompt with context
            max_tokens: Maximum tokens for response
            
        Returns:
            Claude's response text
        """
        try:
            message = anthropic_client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=max_tokens,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            return message.content[0].text
        
        except Exception as e:
            return f"Error generating response: {e}"
    
    def query(self, ingredients: str, additional_request: str = None, 
              top_k: int = 5, recipe_type: str = None):
        """
        Main query interface - combines search and generation
        
        Args:
            ingredients: User's available ingredients
            additional_request: Optional additional requirements (e.g., "for 2 people", "quick recipes")
            top_k: Number of recipes to retrieve
            recipe_type: Optional filter by type
            
        Returns:
            AI-generated response with recipe recommendations
        """
        print(f"\n🔍 Searching for recipes with: {ingredients}")
        if recipe_type:
            print(f"   Filtering by type: {recipe_type}")
        
        # Step 1: Retrieve relevant recipes
        search_results = self.search_recipes(ingredients, top_k, recipe_type)
        
        if not search_results or not search_results['matches']:
            return "Sorry, I couldn't find any recipes matching your ingredients. Try different ingredients or add more options."
        
        print(f"✅ Found {len(search_results['matches'])} relevant recipes")
        
        # Step 2: Format recipes for context
        formatted_recipes = self.format_recipes_for_context(search_results)
        
        # Step 3: Create prompt
        prompt = self.create_prompt(ingredients, formatted_recipes, additional_request)
        
        # Step 4: Generate response with Claude
        print("🤖 Generating recommendations with Claude...")
        response = self.generate_response(prompt)
        
        return response
    
    def interactive_mode(self):
        """Run interactive CLI mode"""
        print("\n" + "="*80)
        print("🍳 RECIPE RAG SYSTEM - INTERACTIVE MODE")
        print("="*80)
        print("\nHow to use:")
        print("  - Enter ingredients separated by commas (e.g., 'chicken, rice, onion')")
        print("  - Optionally add requirements after '|' (e.g., 'chicken | quick and easy')")
        print("  - Type 'filter:TYPE' to filter by recipe type")
        print("  - Type 'quit' to exit\n")
        print("Available recipe types:")
        print("  - STARTERS & SALADS")
        print("  - LIGHT MEALS")
        print("  - MAIN MEALS")
        print("  - GOURMET DOGS")
        print("  - BURGERS")
        print("  - DESSERT & BAKING")
        print("  - COLD SAUCES\n")
        print("="*80 + "\n")
        
        while True:
            try:
                # Get user input
                user_input = input("🍽️  Enter your ingredients (or 'quit'): ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Thanks for using Recipe RAG! Happy cooking!")
                    break
                
                if not user_input:
                    continue
                
                # Parse input
                recipe_type = None
                additional_request = None
                
                # Check for recipe type filter
                if 'filter:' in user_input.lower():
                    parts = user_input.split('filter:', 1)
                    user_input = parts[0].strip()
                    recipe_type = parts[1].strip().upper()
                
                # Check for additional request
                if '|' in user_input:
                    ingredients, additional_request = user_input.split('|', 1)
                    ingredients = ingredients.strip()
                    additional_request = additional_request.strip()
                else:
                    ingredients = user_input
                
                # Query the system
                response = self.query(
                    ingredients=ingredients,
                    additional_request=additional_request,
                    top_k=5,
                    recipe_type=recipe_type
                )
                
                # Display response
                print("\n" + "="*80)
                print("🤖 RECIPE RECOMMENDATIONS:")
                print("="*80)
                print(response)
                print("\n" + "="*80 + "\n")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")


def main():
    """Main entry point"""
    # Initialize RAG system
    rag = RecipeRAG()
    
    # Run in interactive mode
    rag.interactive_mode()


if __name__ == "__main__":
    main()