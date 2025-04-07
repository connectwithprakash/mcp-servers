import os
import re
import tiktoken
import traceback
import time
import requests
from bs4 import BeautifulSoup
from typing import List, Set
import hashlib
import json
from tqdm import tqdm

from langchain_community.document_loaders import RecursiveUrlLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import SKLearnVectorStore


class CustomOllamaEmbeddings(Embeddings):
    """
    Custom implementation of Ollama embeddings to avoid dependency conflicts.
    """

    def __init__(
        self,
        model: str = "gemma",
        base_url: str = "http://localhost:11434",
    ):
        """Initialize the Ollama embeddings client."""
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.client = requests.Session()

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed texts using Ollama API."""
        results = []
        for text in tqdm(texts, desc="Generating embeddings", unit="text"):
            try:
                response = self.client.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.model, "prompt": text},
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()
                data = response.json()
                embedding = data.get("embedding")
                if embedding:
                    results.append(embedding)
                else:
                    raise ValueError(f"No embedding returned for text: {text[:50]}...")
            except Exception as e:
                print(f"Error generating embedding: {str(e)}")
                # Return a zero vector as fallback (not ideal but prevents complete failure)
                if results and len(results) > 0:
                    # Use the same dimension as previous embeddings
                    results.append([0.0] * len(results[0]))
                else:
                    # If no previous embeddings, use a standard dimension
                    results.append([0.0] * 1536)
        return results

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents using Ollama."""
        return self._embed_texts(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed a query using Ollama."""
        return self._embed_texts([text])[0]


def count_tokens(text, model="cl100k_base"):
    """
    Count the number of tokens in the text using tiktoken.

    Args:
        text (str): The text to count tokens for
        model (str): The tokenizer model to use (default: cl100k_base for GPT-4)

    Returns:
        int: Number of tokens in the text
    """
    try:
        encoder = tiktoken.get_encoding(model)
        return len(encoder.encode(text))
    except Exception:
        # Fallback token counting (approximate)
        return len(text) // 4  # Rough approximation


def generate_url_hash(url: str) -> str:
    """Generate a hash for a URL to use as a unique identifier."""
    return hashlib.md5(url.encode()).hexdigest()


def get_great_learning_dir():
    """Get the great_learning directory path."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return current_dir


def load_processed_urls() -> Set[str]:
    """Load previously processed URLs from metadata file."""
    processed_urls = set()
    metadata_path = os.path.join(get_great_learning_dir(), "great_learning_metadata.json")

    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                processed_urls = set(metadata.get("processed_urls", []))
                print(f"Loaded {len(processed_urls)} previously processed URLs.")
        except Exception as e:
            print(f"Error loading processed URLs: {str(e)}")

    return processed_urls


def save_processed_urls(processed_urls: Set[str]):
    """Save processed URLs to metadata file."""
    metadata_path = os.path.join(get_great_learning_dir(), "great_learning_metadata.json")

    try:
        metadata = {"processed_urls": list(processed_urls)}
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)
        print(f"Saved {len(processed_urls)} processed URLs to metadata.")
    except Exception as e:
        print(f"Error saving processed URLs: {str(e)}")


def bs4_extractor(html: str) -> str:
    """
    Extract meaningful content from HTML using BeautifulSoup.
    Specifically tailored for Great Learning website structure.

    Args:
        html (str): Raw HTML content

    Returns:
        str: Extracted text content
    """
    soup = BeautifulSoup(html, "lxml")

    # Remove unwanted elements
    for element in soup.select("script, style, iframe, noscript, header, footer, nav, .cookie-banner, .popup, .modal"):
        element.extract()

    # Try to find main content sections specific to Great Learning
    main_content = None

    # Look for program details
    program_content = soup.select(
        ".program-details, .course-details, .program-info, .course-info, .program-content, article"
    )
    if program_content:
        main_content = " ".join([pc.get_text() for pc in program_content])

    # If no program content found, try to find content sections
    if not main_content:
        content_sections = soup.select("main, .main-content, .content, section")
        if content_sections:
            main_content = " ".join([cs.get_text() for cs in content_sections])

    # If still no content found, use the whole body but try to clean it up
    if not main_content:
        body = soup.find("body")
        if body:
            main_content = body.get_text()
        else:
            main_content = soup.get_text()

    # Clean up whitespace
    lines = (line.strip() for line in main_content.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = "\n".join(chunk for chunk in chunks if chunk)

    # Clean up with regex
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n\s*\n", "\n\n", text)

    return text.strip()


def load_existing_documents():
    """Load existing documents from file."""
    documents = []
    document_path = os.path.join(get_great_learning_dir(), "great_learning_full.txt")

    if os.path.exists(document_path):
        from langchain_community.document_loaders import TextLoader

        try:
            loader = TextLoader(document_path)
            documents = loader.load()
            print(f"Loaded {len(documents)} existing documents.")
        except Exception as e:
            print(f"Error loading existing documents: {str(e)}")

    return documents


def load_great_learning_docs():
    """
    Load Great Learning documentation from the official website.
    Only scrapes URLs that haven't been processed before.

    This function:
    1. Uses RecursiveUrlLoader to fetch pages from the Great Learning website
    2. Counts the total documents and tokens loaded

    Returns:
        list: A list of Document objects containing the loaded content
        list: A list of tokens per document
    """
    print("Loading Great Learning content...")

    # Load previously processed URLs
    processed_urls = load_processed_urls()

    # Base URL for Great Learning
    base_url = "https://www.mygreatlearning.com"

    # Main sections to start scraping from
    urls = [
        base_url,
        f"{base_url}/artificial-intelligence/courses",
        f"{base_url}/data-science/courses",
        f"{base_url}/business-analytics/courses",
        f"{base_url}/cloud-computing/courses",
        f"{base_url}/cyber-security/courses",
        f"{base_url}/software-development/courses",
        f"{base_url}/digital-marketing/courses",
        f"{base_url}/pg-program-data-science-business-analytics-course",
        f"{base_url}/pg-program-online-artificial-intelligence-machine-learning",
        f"{base_url}/gen-ai/courses",
        f"{base_url}/management/courses",
    ]

    # Filter out already processed URLs
    new_urls = [url for url in urls if url not in processed_urls]
    if not new_urls:
        print("No new URLs to process. Using existing documents.")
        return load_existing_documents(), []

    print(f"Found {len(new_urls)} new URLs to process.")

    docs = []
    new_processed_urls = set(processed_urls)  # Create a copy to track newly processed URLs

    for url in tqdm(new_urls, desc="Processing URLs", unit="url"):
        try:
            print(f"\nScraping: {url}")

            loader = RecursiveUrlLoader(
                url,
                max_depth=2,  # Reduced depth to prevent excessive scraping
                extractor=bs4_extractor,
                prevent_outside=True,  # Stay on the Great Learning domain
                exclude_dirs=[
                    f"{base_url}/blog",  # Exclude blog to prevent too many documents
                    f"{base_url}/login",
                    f"{base_url}/register",
                    f"{base_url}/cart",
                    "logout",
                    "account",
                    "password",
                    "feedback",
                    "?",  # Exclude URLs with query parameters
                    "#",  # Exclude URLs with fragments
                ],
                use_async=True,  # Use async for faster loading
                timeout=30,  # Timeout for requests
            )

            # Load documents using lazy loading (memory efficient)
            docs_lazy = loader.lazy_load()

            # Load documents and track URLs
            url_count = 0
            doc_progress = tqdm(desc=f"Documents from {url.split('/')[-1]}", unit="doc")

            for d in docs_lazy:
                # Sleep between document retrievals to avoid rate limiting
                time.sleep(1)
                doc_progress.update(1)

                if url_count > 50:  # Limit to 50 documents per starting URL to avoid overwhelming
                    print(f"Reached document limit for {url}, moving to next URL")
                    break

                # Only include documents with a minimum amount of content
                if len(d.page_content) > 500:  # Skip pages with very little content
                    # Check if this specific document URL has been processed
                    doc_url = d.metadata.get("source", "Unknown URL")
                    if doc_url not in processed_urls:
                        docs.append(d)
                        new_processed_urls.add(doc_url)
                        print(f"Loaded new content: {doc_url}")
                        url_count += 1
                    else:
                        print(f"Skipping already processed URL: {doc_url}")
                else:
                    print(f"Skipping document with insufficient content: {d.metadata.get('source', 'Unknown URL')}")

            doc_progress.close()

            # Mark the base URL as processed
            new_processed_urls.add(url)

        except Exception as e:
            print(f"Error scraping {url}: {str(e)}")
            continue

    # Load existing documents if any new documents were found
    if docs:
        # Get existing documents
        existing_docs = load_existing_documents()

        # Combine with new documents
        combined_docs = existing_docs + docs
        print(f"Combined {len(existing_docs)} existing documents with {len(docs)} new documents.")

        # Save the updated list of processed URLs
        save_processed_urls(new_processed_urls)

        # Count total tokens in documents
        total_tokens = 0
        tokens_per_doc = []

        # Add progress bar for token counting
        for doc in tqdm(combined_docs, desc="Counting tokens", unit="doc"):
            try:
                doc_tokens = count_tokens(doc.page_content)
                total_tokens += doc_tokens
                tokens_per_doc.append(doc_tokens)
            except Exception as e:
                print(f"Error counting tokens for document {doc.metadata.get('source', 'Unknown URL')}: {str(e)}")
                tokens_per_doc.append(0)

        print(f"Total tokens in loaded documents: {total_tokens}")
        return combined_docs, tokens_per_doc
    else:
        print("No new documents found. Using existing documents.")
        existing_docs = load_existing_documents()
        return existing_docs, []


def save_documents(documents):
    """Save the documents to a file, overwriting any existing file."""
    if not documents:
        print("No documents to save.")
        return

    # Open the output file
    output_filename = os.path.join(get_great_learning_dir(), "great_learning_full.txt")

    with open(output_filename, "w") as f:
        # Write each document with progress bar
        for i, doc in tqdm(enumerate(documents), desc="Saving documents", unit="doc", total=len(documents)):
            # Get the source (URL) from metadata
            source = doc.metadata.get("source", "Unknown URL")

            # Write the document with proper formatting
            f.write(f"DOCUMENT {i+1}\n")
            f.write(f"SOURCE: {source}\n")
            f.write("CONTENT:\n")
            f.write(doc.page_content)
            f.write("\n\n" + "=" * 80 + "\n\n")

    print(f"Documents concatenated into {output_filename}")


def split_documents(documents):
    """
    Split documents into smaller chunks for improved retrieval.

    This function:
    1. Uses RecursiveCharacterTextSplitter with tiktoken to create semantically meaningful chunks
    2. Ensures chunks are appropriately sized for embedding and retrieval
    3. Counts the resulting chunks and their total tokens

    Args:
        documents (list): List of Document objects to split

    Returns:
        list: A list of split Document objects
    """
    if not documents:
        print("No documents to split.")
        return []

    print("Splitting documents...")

    # Initialize text splitter using tiktoken for accurate token counting
    # chunk_size=8,000 creates relatively large chunks for comprehensive context
    # chunk_overlap=500 ensures continuity between chunks
    try:
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=8000, chunk_overlap=500)
    except Exception:
        # Fallback to character-based splitting if tiktoken is not available
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=8000, chunk_overlap=500)

    # Split documents into chunks with progress bar
    print("Splitting documents into chunks...")
    split_docs = []
    for doc in tqdm(documents, desc="Splitting documents", unit="doc"):
        split_docs.extend(text_splitter.split_documents([doc]))

    print(f"Created {len(split_docs)} chunks from documents.")

    # Count total tokens in split documents
    total_tokens = 0
    for doc in tqdm(split_docs, desc="Counting tokens in chunks", unit="chunk"):
        total_tokens += count_tokens(doc.page_content)

    print(f"Total tokens in split documents: {total_tokens}")

    return split_docs


def create_vectorstore(splits):
    """
    Create a vector store from document chunks using SKLearnVectorStore with Ollama.

    This function:
    1. Initializes an embedding model to convert text into vector representations using Ollama
    2. Creates a vector store from the document chunks

    Args:
        splits (list): List of split Document objects to embed

    Returns:
        SKLearnVectorStore: A vector store containing the embedded documents
    """
    if not splits:
        print("No document splits to create vector store.")
        return None

    print("Creating SKLearnVectorStore with custom Ollama embeddings...")

    # Initialize custom Ollama embeddings with Gemma model
    embeddings = CustomOllamaEmbeddings(
        model="gemma",  # Using Gemma model
        base_url="http://localhost:11434",  # Default Ollama server URL
    )

    # Check if vector store already exists
    persist_path = os.path.join(get_great_learning_dir(), "great_learning_vectorstore.parquet")

    if os.path.exists(persist_path):
        print(f"Existing vector store found at {persist_path}")
        print("Updating with new documents...")

        # Create a new vector store with all documents
        vectorstore = SKLearnVectorStore.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_path=persist_path,
            serializer="parquet",
        )
    else:
        # Create vector store from documents using SKLearn
        print("Creating new vector store...")
        vectorstore = SKLearnVectorStore.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_path=persist_path,
            serializer="parquet",
        )

    print("SKLearnVectorStore created successfully.")

    vectorstore.persist()
    print("SKLearnVectorStore was persisted to", persist_path)

    return vectorstore


def main():
    """Main function to run the Great Learning scraping and vector store creation"""
    try:
        # Load documents from Great Learning website
        docs, tokens_per_doc = load_great_learning_docs()

        if not docs:
            print("No documents were loaded. Exiting.")
            return None

        # Save documents to file
        save_documents(docs)

        # Split documents into chunks
        splits = split_documents(docs)

        if not splits:
            print("No document splits were created. Check document content.")
            return None

        # Create vector store
        vectorstore = create_vectorstore(splits)

        print("Process completed successfully!")
        return vectorstore

    except Exception as e:
        print(f"An error occurred during execution: {str(e)}")
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
