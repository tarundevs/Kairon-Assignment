RAG Research Workflow
Overview
This project implements a Retrieval-Augmented Generation (RAG) research workflow using LangChain and LangGraph. It automates the process of researching a user-provided question by generating search queries, performing web searches, building a vector database, retrieving relevant context, analyzing results, and drafting a comprehensive answer.
Features

Query Generation: Generates diverse search queries to explore different aspects of the research question.
Web Search: Uses the Tavily search tool to gather relevant web results.
Vector Database: Stores processed search results in a Chroma vector database for efficient retrieval.
Context Retrieval: Retrieves the most relevant documents from the vector database using similarity search.
Result Analysis: Extracts key insights, themes, and sources from search results and retrieved context.
Answer Drafting: Produces a well-structured, 300-400 word answer with citations.

Requirements

Python 3.8+
Required libraries (install via pip):pip install langchain langchain-tavily langchain-huggingface pydantic chromadb langgraph


API keys:
HuggingFace Hub API key for the language model.
Tavily API key for web search.


Set environment variables:export HUGGINGFACEHUB_API_TOKEN='your-huggingface-token'
export TAVILY_API_KEY='your-tavily-token'



Usage

Clone the repository:git clone <repository-url>
cd <repository-directory>


Install dependencies:pip install -r requirements.txt


Run the script:python main.py


Enter a research question when prompted, or use the default question ("What are the latest developments in nuclear fusion technology?").

Code Structure

main.py: Contains the core RAG workflow implementation, including:
RAGResearchState: Pydantic model for tracking research state.
Node functions (generate_queries, perform_searches, etc.) for each workflow step.
create_research_graph: Defines the LangGraph workflow.
run_research: Executes the full research pipeline.


Chroma Database: Persists vector embeddings in ./chroma_db.

Example
question = "What are the latest developments in nuclear fusion technology?"
final_state = run_research(question)
print(final_state.answer)

Output
The script outputs:

The research question and generated queries.
A list of unique sources found.
A comprehensive answer with citations, incorporating insights and themes.

Notes

The workflow uses the mistralai/Mixtral-8x7B-Instruct-v0.1 model from HuggingFace for language tasks and sentence-transformers/all-MiniLM-L6-v2 for embeddings.
Error handling ensures robustness, with fallback mechanisms for query generation and answer drafting.
The Chroma vector database persists data automatically, but ensure the ./chroma_db directory is writable.

License
This project is licensed under the MIT License.
