# Great Learning Web Scraper and Vector Store Creator

This script scrapes content from the Great Learning website (https://www.mygreatlearning.com) and creates a vector store for efficient retrieval using semantic search.

## Features

- Scrapes multiple sections of the Great Learning website
- Extracts and cleans the content using BeautifulSoup
- Splits documents into manageable chunks for better retrieval
- Creates a vector store using SKLearnVectorStore and Ollama embeddings with Gemma model
- Saves the raw scraped content to a text file

## Requirements

- Python 3.12+
- Poetry for dependency management
- Ollama with Gemma model installed

## Installation

1. Install Poetry if you haven't already:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. Install dependencies:
   ```bash
   poetry install
   ```

3. Install Ollama:
   ```bash
   # On macOS
   brew install ollama
   
   # On Linux
   curl -fsSL https://ollama.com/install.sh | sh
   ```

4. Pull the Gemma model:
   ```bash
   ollama pull gemma
   ```

## Usage

Run the script to scrape the Great Learning website and create a vector store:

```bash
poetry run python run.py
```

The script will:
1. Scrape the Great Learning website
2. Save the raw content to `great_learning_full.txt`
3. Split the documents into chunks
4. Create a vector store saved as `great_learning_vectorstore.parquet`

## Customization

You can modify the script to:
- Adjust scraping depth by changing the `max_depth` parameter
- Include or exclude additional URL paths
- Modify chunk size for document splitting
- Change the embedding model

## Notes

- The script respects website boundaries by staying within the Great Learning domain
- Carefully consider rate limiting when scraping to avoid overloading the website
- The vector store uses Ollama for generating embeddings with the Gemma model

Add the config files in the following directories as required by the MCP server:
**Cursor** 
`~/.cursor/mcp.json` 

**Windsurf**
`~/.codeium/windsurf/mcp_config.json`

 **Claude Desktop**
`~/Library/Application\ Support/Claude/claude_desktop_config.json`

**Claude Code**
`~/.claude.json`