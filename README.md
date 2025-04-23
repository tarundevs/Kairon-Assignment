# Deep research AI Agent

A powerful research Agent that leverages Retrieval-Augmented Generation (RAG) and dual agent architecture to answer research questions with comprehensive, sourced responses.

## Overview

This system combines web search, vector database retrieval, and advanced language models to:

1. Generate targeted search queries based on your question
2. Retrieve relevant information from the web
3. Store and index information in a vector database
4. Extract key insights and themes
5. Synthesize a comprehensive answer with citations

## Features

- **Multi-query search**: Generates multiple diversified search queries to explore different aspects of your question
- **Information extraction**: Processes search results into a structured database
- **Vector similarity search**: Finds the most relevant context for your specific question
- **Insight analysis**: Identifies key insights and themes from the research
- **Source tracking**: Maintains attribution to original sources
- **Comprehensive answers**: Generates well-structured responses with proper citations

## Setup

### Prerequisites

- Python 3.8+
- Tavily API key (for web search)
- HuggingFace API token (for language model access)

### Installation

1. Clone this repository
2. Install required packages:

```bash
pip install langchain langchain_tavily langgraph pydantic huggingface-hub chromadb sentence-transformers
```

3. Set up environment variables:

```bash
export TAVILY_API_KEY="your_tavily_api_key"
export HUGGINGFACEHUB_API_TOKEN="your_huggingface_token"
```

## Usage

```python
from rag_research import run_research

# Run a research query
result = run_research("What are the latest developments in nuclear fusion technology?")

# Access the structured results
print(result.answer)
print(result.insights)
print(result.sources)
```

## Architecture

This system uses LangGraph to orchestrate a workflow of specialized components:

1. **Query Generation**: Creates diverse search queries from your question
2. **Web Search**: Retrieves information from the web using Tavily
3. **RAG Database**: Processes search results into chunks and stores them
4. **Context Retrieval**: Finds the most relevant information for your question
5. **Result Analysis**: Identifies key insights and themes from the research
6. **Answer Generation**: Synthesizes a comprehensive response with citations

## Model Information

- **Language Model**: Mixtral-8x7B-Instruct-v0.1 (via HuggingFace Hub)
- **Embedding Model**: sentence-transformers/all-MiniLM-L6-v2
- **Vector Database**: Chroma (with persistent storage)

## Customization

You can modify various parameters to customize the research process:

- Change the LLM by updating the `repo_id` in the `HuggingFaceHub` initialization
- Adjust search depth by modifying `max_results` in `TavilySearch`
- Change text chunking parameters in `RecursiveCharacterTextSplitter`
- Modify vector search parameters in the `similarity_search` method

## Error Handling

The system includes robust error handling:
- Fallback query generation if the LLM response cannot be parsed
- Multiple JSON extraction strategies for handling different LLM output formats
- Exception handling at each pipeline stage
- Graceful degradation when components fail

## Limitations

- Limited by the quality and recency of web search results
- Performance depends on the capabilities of the underlying language model
- Research quality may vary based on the specificity and complexity of the question

This project uses several open-source libraries:
- LangChain and LangGraph for orchestration
- HuggingFace for model access
- Tavily for web search capabilities
- ChromaDB for vector storage
