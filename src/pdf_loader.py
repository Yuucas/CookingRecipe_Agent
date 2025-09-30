import os
import numpy as np
from dotenv import load_dotenv
import re

from langchain_community.document_loaders import DirectoryLoader, PDFMinerLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

API_KEY = os.getenv('API_KEY')
DATASET_PATH = os.getenv('DATASET_PATH')


def load_pdf_documents(file_path):
    loader = PyPDFLoader(file_path,
                        mode='page',
                        extract_images=True,
                        extraction_mode='layout'
                        )
    documents = loader.load()
    return documents


def split_into_chunks(documents):
    recipe_chunks = []
    
    for i, doc in enumerate(documents):
        # IMPORTANT: Reset variables for EACH recipe
        recipe_header = ""
        serving_suggestion = ""
        ingredients = ""
        method = ""
        chef_tip = ""
        
        text = doc.page_content
        
        # Skip pages with too little content
        if len(text) < 100:
            continue
            
        # Check if page contains both INGREDIENTS and METHOD sections
        s1 = re.escape("INGREDIENTS")
        s2 = re.escape("METHOD")
        pattern_string = rf"(?=.*{s1})(?=.*{s2})"
        pattern = re.compile(pattern_string)
        if not pattern.search(text): 
            continue
        
        # Parse the page content
        for line_idx, line in enumerate(text.split('\n')):
            line_parts = re.split(r'\s{3,}', line)
            
            # Extract header info from first 5 lines
            if line_idx < 5 and line_parts != ['']:
                recipe_header += line_parts[0] + " "
                if len(line_parts) > 1:
                    serving_suggestion += line_parts[1] + " "
            
            # Extract ingredients and method from remaining lines
            elif line_idx >= 5 and line_idx < 45 and line_parts != ['']:
                ingredients += line_parts[0] + "\n"
                if len(line_parts) > 1:
                    if re.search(":", line_parts[1]):
                        method += "\n" + line_parts[1] + "\n"
                    else:
                        method += line_parts[1] + " "

        # Clean up extracted content
        ingredients = ingredients.replace("INGREDIENTS", "").strip()
        
        # Extract chef's tip if present
        method, separator, chef_tip = method.partition("CHEF'S TIP:")
        method = method.replace("METHOD", "").strip()
        if separator:
            chef_tip = chef_tip.replace("\n", " ").strip()

        # Add recipe-specific metadata
        doc.metadata["recipe_name"] = recipe_header.strip()
        doc.metadata["serving_suggestion"] = serving_suggestion.strip()

        # Categorize recipe type by page number
        page_label = int(doc.metadata["page_label"])
        if page_label < 28:
            doc.metadata["recipe_type"] = "STARTERS & SALADS"
        elif page_label < 51:
            doc.metadata["recipe_type"] = "LIGHT MEALS"
        elif page_label < 92:
            doc.metadata["recipe_type"] = "MAIN MEALS"
        elif page_label < 110:
            doc.metadata["recipe_type"] = "GOURMET DOGS"
        elif page_label < 128:
            doc.metadata["recipe_type"] = "BURGERS"
        elif page_label >= 135:
            doc.metadata["recipe_type"] = "DESSERT & BAKING"
        else:
            doc.metadata["recipe_type"] = "COLD SAUCES"

        # Create recipe chunk
        recipe_chunks.append({
            "recipe_header": recipe_header.strip(),
            "serving_suggestion": serving_suggestion.strip(),
            "ingredients": ingredients,
            "method": method,
            "chef_tip": chef_tip,
            "metadata": doc.metadata
        })

    return recipe_chunks


if __name__ == "__main__":
    documents = load_pdf_documents(DATASET_PATH)
    print(f"Loaded {len(documents)} documents from {DATASET_PATH}")
    print(f"First document content:\n{documents[7].page_content}...\n")
    print(f"First document metadata:\n{documents[7].metadata}...\n")

    recipe_chunks = split_into_chunks(documents)
    print(f"Extracted {len(recipe_chunks)} recipe chunks")
    print(f"\nFirst recipe chunk:")
    print(f"Recipe name: {recipe_chunks[0]['metadata']['recipe_name']}")
    print(f"Ingredients (first 100): {recipe_chunks[0]['ingredients'][:100]}...")
    
    # Test multiple recipes to verify they're different
    if len(recipe_chunks) > 2:
        print(f"\nSecond recipe chunk:")
        print(f"Recipe name: {recipe_chunks[1]['metadata']['recipe_name']}")
        print(f"\nThird recipe chunk:")
        print(f"Recipe name: {recipe_chunks[2]['metadata']['recipe_name']}")