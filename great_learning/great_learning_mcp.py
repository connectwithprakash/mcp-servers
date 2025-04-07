#!/usr/bin/env python

import os
from mcp.server.fastmcp import FastMCP
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import SKLearnVectorStore


# Define path to the great_learning directory
def get_great_learning_dir():
    """Get the great_learning directory path."""
    return os.path.dirname(os.path.abspath(__file__))


# Create an MCP server
mcp = FastMCP("Great-Learning-MCP-Server")


# Add a tool to query the Great Learning documentation
@mcp.tool()
def great_learning_query_tool(query: str):
    """
    Query the Great Learning documentation using a retriever.

    Args:
        query (str): The query to search the documentation with

    Returns:
        str: A str of the retrieved documents
    """
    # Create Ollama embeddings
    embeddings = OllamaEmbeddings(
        model="gemma",  # Using Gemma model
        base_url="http://localhost:11434",  # Default Ollama server URL
    )

    # Get path to the vector store
    persist_path = os.path.join(get_great_learning_dir(), "great_learning_vectorstore.parquet")

    # Create retriever from vector store
    retriever = SKLearnVectorStore(embedding=embeddings, persist_path=persist_path, serializer="parquet").as_retriever(
        search_kwargs={"k": 3}
    )

    # Get relevant documents
    relevant_docs = retriever.invoke(query)
    print(f"Retrieved {len(relevant_docs)} relevant documents")

    # Format the retrieved documents
    formatted_context = "\n\n".join([f"==DOCUMENT {i+1}==\n{doc.page_content}" for i, doc in enumerate(relevant_docs)])
    return formatted_context


# The @mcp.resource() decorator maps a URI pattern to a function that provides the resource content
@mcp.resource("docs://greatlearning/full")
def get_all_great_learning_docs() -> str:
    """
    Get all the Great Learning documentation. Returns the contents of the file great_learning_full.txt,
    which contains the full set of Great Learning documentation. This is useful
    for a comprehensive response to questions about Great Learning courses and programs.

    Args: None

    Returns:
        str: The contents of the Great Learning documentation
    """
    # Local path to the Great Learning documentation
    doc_path = os.path.join(get_great_learning_dir(), "great_learning_full.txt")
    try:
        with open(doc_path, "r") as file:
            return file.read()
    except Exception as e:
        return f"Error reading Great Learning documentation file: {str(e)}"


# Add a tool to get information about specific Great Learning courses
@mcp.tool()
def get_course_info(course_name: str):
    """
    Get information about a specific Great Learning course.

    Args:
        course_name (str): The name or keywords for the course to search for

    Returns:
        str: Information about the course
    """
    # Use the query tool to find course information
    return great_learning_query_tool(f"information about {course_name} course")


# Add a tool to compare different Great Learning courses
@mcp.tool()
def compare_courses(course1: str, course2: str):
    """
    Compare two Great Learning courses.

    Args:
        course1 (str): The name of the first course
        course2 (str): The name of the second course

    Returns:
        str: Comparison information about the two courses
    """
    # Get information about both courses
    info1 = great_learning_query_tool(f"details about {course1} course")
    info2 = great_learning_query_tool(f"details about {course2} course")

    return f"Information about {course1}:\n\n{info1}\n\n===COMPARISON===\n\nInformation about {course2}:\n\n{info2}"


if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport="stdio")
