import os
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from langchain_tavily import TavilySearch
from langchain.llms import HuggingFaceHub
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph
import json

# Set up environment variables for API keys
os.environ["TAVILY_API_KEY"] = "api_key"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "api_key"

# Define the enhanced state model with RAG components
class RAGResearchState(BaseModel):
    question: str
    queries: List[str] = Field(default_factory=list)
    results: List[Dict] = Field(default_factory=list)
    rag_documents: List[Dict] = Field(default_factory=list)
    relevant_context: str = ""
    insights: List[str] = Field(default_factory=list)
    themes: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)
    answer: str = ""

# Initialize LLM, embeddings model, and search tool
llm = HuggingFaceHub(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    model_kwargs={"temperature": 0.2, "max_length": 500}
)

# Use a sentence transformer model for embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Initialize search tool
tavily_tool = TavilySearch(max_results=5)

# Initialize text splitter for chunking documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

# Initialize vector database - persist_directory allows for reuse across sessions
db = Chroma(embedding_function=embeddings, persist_directory="./chroma_db")

# Research Agent Nodes
def generate_queries(state: RAGResearchState) -> RAGResearchState:
    """Generate search queries based on the input question"""
    prompt = f"""Generate 4 diverse search queries for the question: {state.question}.
    Include different phrasings and focus areas to ensure comprehensive coverage.
    Return only a JSON object: {{ "queries": ["query1", "query2", "query3", "query4"] }}"""
    
    try:
        response = llm(prompt)
        data = json.loads(response)
        state.queries = data.get("queries", [])
    except Exception as e:
        print(f"Error generating queries: {e}")
        state.queries = [
            state.question, 
            f"{state.question} latest research", 
            f"{state.question} explained", 
            f"{state.question} examples"
        ]
    return state

def perform_searches(state: RAGResearchState) -> RAGResearchState:
    """Execute web searches and process results"""
    for query in state.queries:
        try:
            # Search the web
            raw_results = tavily_tool.invoke(query)
            
            # Process and clean results
            parsed_results = []
            if isinstance(raw_results, list):
                for item in raw_results:
                    if isinstance(item, dict) and all(k in item for k in ["title", "content", "url"]):
                        parsed_results.append(item)
            
            state.results.extend(parsed_results)
        except Exception as e:
            print(f"Search error for query {query}: {e}")
    
    # Remove duplicates based on URLs
    urls_seen = set()
    unique_results = []
    for result in state.results:
        if result["url"] not in urls_seen:
            urls_seen.add(result["url"])
            unique_results.append(result)
    
    state.results = unique_results
    return state

def build_rag_database(state: RAGResearchState) -> RAGResearchState:
    """Process search results into RAG documents and store in vector database"""
    documents = []
    
    # Convert search results to documents
    for result in state.results:
        # Create a document with metadata
        content = f"Title: {result['title']}\nContent: {result['content']}"
        metadata = {
            "url": result["url"],
            "title": result["title"],
            "question": state.question
        }
        
        # Split into chunks
        doc_chunks = text_splitter.create_documents(
            texts=[content],
            metadatas=[metadata]
        )
        documents.extend(doc_chunks)
    
    # Store document information in state
    state.rag_documents = [
        {"content": doc.page_content, "metadata": doc.metadata} 
        for doc in documents
    ]
    
    # Add to vector store
    if documents:
        db.add_documents(documents)
        db.persist()  # Save to disk
    
    return state

def retrieve_relevant_context(state: RAGResearchState) -> RAGResearchState:
    """Retrieve relevant context from the vector database"""
    # Search the vector database
    results = db.similarity_search(
        state.question,
        k=7,  # Number of relevant documents to retrieve
        filter=None  # Could filter by metadata if needed
    )
    
    # Format retrieved documents into context
    context_parts = []
    for i, doc in enumerate(results):
        context_parts.append(f"[Document {i+1}]\nContent: {doc.page_content}\nSource: {doc.metadata.get('url', 'Unknown')}")
    
    state.relevant_context = "\n\n".join(context_parts)
    return state

def analyze_results(state: RAGResearchState) -> RAGResearchState:
    """Analyze search results and RAG context to extract insights and themes"""
    # Use both fresh search results and RAG context for analysis
    combined_content = f"""
    Question: {state.question}
    
    RAG Context:
    {state.relevant_context}
    
    Recent Search Results:
    {', '.join([f"Title: {r['title']}, URL: {r['url']}" for r in state.results[:5]])}
    """
    
    prompt = f"""Analyze the following research materials for the question: {state.question}
    
    {combined_content}
    
    Extract key insights, identify major themes, and list all unique sources.
    Return as JSON: {{"insights": ["insight1", "insight2", ...], "themes": ["theme1", "theme2", ...], "sources": ["url1", "url2", ...]}}
    """
    
    try:
        response = llm(prompt)
        data = json.loads(response)
        state.insights = data.get("insights", [])
        state.themes = data.get("themes", [])
        
        # Collect all unique sources
        unique_sources = set()
        # From search results
        for r in state.results:
            unique_sources.add(r["url"])
        # From RAG context sources
        if "sources" in data:
            for source in data["sources"]:
                unique_sources.add(source)
                
        state.sources = list(unique_sources)
    except Exception as e:
        print(f"Analysis error: {e}")
        state.insights = ["Error extracting insights"]
        state.themes = ["Error identifying themes"]
        state.sources = [r["url"] for r in state.results]
    
    return state

def draft_answer(state: RAGResearchState) -> RAGResearchState:
    """Draft a comprehensive answer using all available information"""
    prompt = f"""
    Write a comprehensive 300-400 word answer for the question: {state.question}
    
    Base your answer on:
    
    Key insights: {state.insights}
    Major themes: {state.themes}
    
    Include specific information from the most relevant sources, and cite all sources used with numbered citations [1], [2], etc.
    
    Ensure the answer is well-structured, informative, and directly addresses the question with accurate information.
    """
    
    try:
        response = llm(prompt)
        state.answer = response
    except Exception as e:
        state.answer = f"Failed to generate answer: {e}"
    
    return state

# Create the LangGraph workflow
graph_builder = StateGraph(RAGResearchState)

# Add all nodes
graph_builder.add_node("generate_queries", generate_queries)
graph_builder.add_node("perform_searches", perform_searches)
graph_builder.add_node("build_rag_database", build_rag_database)
graph_builder.add_node("retrieve_relevant_context", retrieve_relevant_context)
graph_builder.add_node("analyze_results", analyze_results)
graph_builder.add_node("draft_answer", draft_answer)

# Define the workflow
graph_builder.add_edge("generate_queries", "perform_searches")
graph_builder.add_edge("perform_searches", "build_rag_database")
graph_builder.add_edge("build_rag_database", "retrieve_relevant_context")
graph_builder.add_edge("retrieve_relevant_context", "analyze_results")
graph_builder.add_edge("analyze_results", "draft_answer")

# Set entry point
graph_builder.set_entry_point("generate_queries")

# Compile graph
rag_research_graph = graph_builder.compile()

# Example usage
if __name__ == "__main__":
    initial_state = RAGResearchState(question="What are the latest developments in nuclear fusion technology?")
    final_state_dict = rag_research_graph.invoke(initial_state)
    final_state = RAGResearchState(**final_state_dict)
    
    print("\n=== Final Answer ===")
    print(final_state.answer)
    print("\n=== Sources ===")
    for i, source in enumerate(final_state.sources, 1):
        print(f"[{i}] {source}")
