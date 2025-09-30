# Recipe RAG System 🍳

An intelligent recipe recommendation system built using Retrieval-Augmented Generation (RAG) that suggests recipes based on available ingredients. The system combines semantic search with Claude AI to provide personalized, conversational recipe recommendations.

## 🌟 Features

- **Intelligent Ingredient Matching**: Uses semantic embeddings to find recipes that match your available ingredients
- **Conversational AI**: Powered by Claude Sonnet 3.5 Haiku for natural, helpful cooking advice
- **Smart Filtering**: Filter recipes by type (starters, mains, desserts, etc.)
- **Missing Ingredient Detection**: Identifies what you're missing and suggests substitutions
- **Interactive CLI**: Easy-to-use command-line interface for quick recipe searches
- **Vector Search**: Fast and accurate recipe retrieval using Pinecone vector database

## 🏗️ Architecture

```
User Query (ingredients)
         ↓
    Vector Search (Pinecone)
         ↓
   Top K Relevant Recipes
         ↓
    Context + User Query
         ↓
   Claude AI (LLM)
         ↓
  Personalized Recommendations
```

## 🛠️ Technologies

- **Python 3.11+**
- **LangChain**: Document loading and processing
- **Pinecone**: Vector database for semantic search
- **Sentence Transformers**: Text embeddings (`all-MiniLM-L6-v2`)
- **Anthropic Claude**: Large language model for response generation
- **PyPDF**: PDF document parsing

## 📋 Prerequisites

- Python 3.11 or higher
- Pinecone account (free tier available)
- Anthropic API key
- Recipe PDF file (or use your own)

## 🚀 Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/recipe-rag-system.git
cd recipe-rag-system
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**

Create a `.env` file in the project root:
```env
API_KEY=your_anthropic_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
DATASET_PATH=data/Recipe-Book-1-2.pdf
```

To get API keys:
- **Anthropic API**: Sign up at [console.anthropic.com](https://console.anthropic.com)
- **Pinecone API**: Sign up at [app.pinecone.io](https://app.pinecone.io)

## 📁 Project Structure

```
recipe-rag-system/
│
├── data/
│   └── Recipe-Book-1-2.pdf      # Your recipe PDF
│
├── pdf_loader.py                 # PDF parsing and recipe extraction
├── vector_store.py               # Pinecone vector database operations
├── upload_recipes.py             # Upload pipeline script
├── rag_system.py                 # Main RAG interface
├── test_rag.py                   # Testing script
│
├── .env                          # Environment variables (not in repo)
├── .gitignore                    # Git ignore file
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 💻 Usage

### 1. Upload Recipes to Vector Database

First, process your PDF and upload recipes to Pinecone:

```bash
python upload_recipes.py
```

This will:
- Parse the recipe PDF
- Extract structured recipe data
- Generate embeddings
- Upload to Pinecone vector database

### 2. Test the System

Run tests to verify everything works:

```bash
python test_rag.py
```

### 3. Interactive Mode

Start the interactive recipe assistant:

```bash
python rag_system.py
```

## 🎯 Example Queries

### Basic Query
```
🍽️  Enter your ingredients: chicken, rice, garlic

🤖 RECIPE RECOMMENDATIONS:
Based on your available ingredients (chicken, rice, and garlic), here are my top recommendations:

1. **Chicken Fried Rice** - Perfect match! You have all the main ingredients...
```

### With Recipe Type Filter
```
🍽️  Enter your ingredients: chocolate, flour, butter filter:DESSERT & BAKING

🔍 Searching for recipes with: chocolate, flour, butter
   Filtering by type: DESSERT & BAKING
✅ Found 5 relevant recipes
```

### With Additional Requirements
```
🍽️  Enter your ingredients: beef, onion, tomato | quick and easy under 30 minutes

🤖 RECIPE RECOMMENDATIONS:
I found some great quick recipes that match your ingredients and time constraint...
```

## 🎨 Available Recipe Types

- `STARTERS & SALADS`
- `LIGHT MEALS`
- `MAIN MEALS`
- `GOURMET DOGS`
- `BURGERS`
- `DESSERT & BAKING`
- `COLD SAUCES`

## 🧪 Testing

Run the test suite:

```bash
python test_rag.py
```

Tests include:
- Basic ingredient queries
- Filtered queries by recipe type
- Queries with additional requirements

## 📊 How It Works

### 1. PDF Processing
```python
# pdf_loader.py extracts structured data from PDF
{
    'recipe_header': 'SMOKED SNOEK MOUSSE',
    'ingredients': '200g smoked snoek, skinned...',
    'method': 'In a pan, heat the olive oil...',
    'chef_tip': 'Substitute the salmon with...',
    'metadata': {...}
}
```

### 2. Vector Embeddings
Each recipe is converted to a 384-dimensional vector using Sentence Transformers:
```python
searchable_text = f"{recipe_name} {ingredients} {method}"
embedding = model.encode(searchable_text)
```

### 3. Semantic Search
User queries are embedded and compared to recipe vectors using cosine similarity:
```python
query_embedding = model.encode("chicken rice garlic")
results = index.query(vector=query_embedding, top_k=5)
```

### 4. LLM Generation
Retrieved recipes are passed to Claude along with user requirements:
```python
prompt = f"""
User's available ingredients: {ingredients}
Relevant recipes: {retrieved_recipes}
Provide helpful recommendations...
"""
response = claude.generate(prompt)
```

## 📝 Requirements

```
langchain==0.3.13
langchain-community==0.3.13
pinecone-client==5.0.1
sentence-transformers==3.3.1
anthropic==0.39.0
python-dotenv==1.0.1
pypdf==5.1.0
numpy==2.2.0
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🐛 Troubleshooting

### "No recipes found in database"
- Make sure you ran `upload_recipes.py` first
- Check if your Pinecone index has vectors: run `python vector_store.py`


## 🙏 Acknowledgments

- Recipe PDF content from [source name](https://rclfoods.com/wp-content/uploads/2020/12/Recipe-Book-1-2.pdf)
- Built with [LangChain](https://github.com/langchain-ai/langchain)
- Powered by [Anthropic Claude](https://www.anthropic.com)
- Vector search by [Pinecone](https://www.pinecone.io)


## 🚀 Future Enhancements

- [ ] Web interface using Streamlit or Gradio
- [ ] Support for multiple recipe PDFs
- [ ] Nutritional information integration
- [ ] Shopping list generation
- [ ] Recipe image display
- [ ] User ratings and favorites
- [ ] Dietary restriction filters (vegetarian, gluten-free, etc.)
- [ ] Cooking time estimation
- [ ] Multi-language support
