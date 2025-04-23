# LangGraph Research Agent

A powerful research assistant built using LangGraph, LangChain, and Mistral's Mixtral LLM. This system creates a multi-agent workflow that automates web research for any question, generating comprehensive answers with source citations.

## Overview

This research agent automates the information discovery and synthesis process by:

1. Generating relevant search queries based on a user question
2. Searching the web for recent information via the Tavily API
3. Analyzing search results to extract key insights and themes
4. Drafting a concise, informative answer with proper source citations

## Features

- **Dual-agent architecture**: Separates research and content drafting for more focused capabilities
- **Structured workflow**: Breaks research into clear stages using LangGraph's directed graph
- **Source verification**: Includes citation links to original sources
- **Error handling**: Gracefully manages API failures and parsing issues
- **Pydantic models**: Ensures structured data throughout the workflow

## Requirements

- Python 3.8+
- Tavily API key
- HuggingFace API token

## Installation

```bash
pip install langgraph langchain pydantic langchain_tavily
```

## Environment Setup

Create a `.env` file in your project directory with your API keys:

```
TAVILY_API_KEY=your-tavily-api-key
HUGGINGFACEHUB_API_TOKEN=your-hf-api-token
```

## Usage

```python
from your_module import ResearchState, research_graph

# Create the initial state with your research question
initial_state = ResearchState(question="What are the latest advancements in quantum computing?")

# Run the research workflow
final_state_dict = research_graph.invoke(initial_state)
final_state = ResearchState(**final_state_dict)

# Display the results
print("\n=== Final Answer ===")
print(final_state.answer)
print("\n=== Sources ===")
for i, source in enumerate(final_state.sources, 1):
    print(f"[{i}] {source}")
```

## How It Works

### Components

1. **ResearchState Model**: Pydantic model that maintains the state throughout the workflow
2. **LLM**: Mixtral-8x7B-Instruct from HuggingFace for query generation and answer drafting
3. **Search Tool**: Tavily Search API for accessing up-to-date web content
4. **LangGraph**: Orchestrates the workflow between different agent nodes

### Workflow Nodes

1. **generate_queries**: Creates strategic search queries based on the user question
2. **perform_searches**: Executes web searches and processes the results
3. **analyze_results**: Extracts insights and themes from search results
4. **draft_answer**: Synthesizes a coherent answer with proper citations
