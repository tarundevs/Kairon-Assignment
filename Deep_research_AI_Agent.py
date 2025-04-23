import os
from typing import List, Dict, Optional
import re
import json
from pydantic import BaseModel, Field

# LangChain imports for search, LLM, embeddings, and vector store
from langchain_tavily import TavilySearch
from langchain.llms import HuggingFaceHub
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph

# Define the state model to track research progress
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

# Initialize the language model using HuggingFace's Mixtral
llm = HuggingFaceHub(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    model_kwargs={"temperature": 0.2, "max_length": 500}
)

# Set up embeddings for document vectorization
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Initialize Tavily search tool for web queries
tavily_tool = TavilySearch(max_results=5)

# Set up text splitter to chunk documents for vector store
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

# Initialize Chroma vector database for storing documents
db = Chroma(embedding_function=embeddings, persist_directory="./chroma_db")

# Helper function to extract JSON from LLM responses
def extract_json_from_response(response: str) -> dict:
    """
    Attempts to extract JSON from LLM response using multiple strategies.
    Returns extracted data or an empty dict if all strategies fail.
    """
    
    # Try extracting JSON after "Output JSON:" marker
    try:
        output_section = response.split("Output JSON:")[-1].strip()
        return json.loads(output_section)
    except (IndexError, json.JSONDecodeError) as e:
        print(f"Strategy 1 failed: {e}")
    
    # Look for JSON content between curly braces
    json_pattern = r'\{.*\}'
    match = re.search(json_pattern, response, re.DOTALL)
    if match:
        try:
            json_str = match.group(0)
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"Strategy 2 failed: {e}")
    
    # Extract queries specifically if JSON extraction fails
    queries_pattern = r'"queries"\s*:\s*\[(.*?)\]'
    match = re.search(queries_pattern, response, re.DOTALL)
    if match:
        queries_text = match.group(1)
        queries = [q.strip(' \n\t"\'') for q in re.findall(r'"([^"]*)"', queries_text)]
        if queries:
            return {"queries": queries}
    
    # Extract insights, themes, or sources if available
    insights = []
    insights_pattern = r'"insights"\s*:\s*\[(.*?)\]'
    match = re.search(insights_pattern, response, re.DOTALL)
    if match:
        insights_text = match.group(1)
        insights = [i.strip(' \n\t"\'') for i in re.findall(r'"([^"]*)"', insights_text)]
    
    themes = []
    themes_pattern = r'"themes"\s*:\s*\[(.*?)\]'
    match = re.search(themes_pattern, response, re.DOTALL)
    if match:
        themes_text = match.group(1)
        themes = [t.strip(' \n\t"\'') for t in re.findall(r'"([^"]*)"', themes_text)]
    
    sources = []
    sources_pattern = r'"sources"\s*:\s*\[(.*?)\]'
    match = re.search(sources_pattern, response, re.DOTALL)
    if match:
        sources_text = match.group(1)
        sources = [s.strip(' \n\t"\'') for s in re.findall(r'"([^"]*)"', sources_text)]
    
    if insights or themes or sources:
        return {
            "insights": insights,
            "themes": themes,
            "sources": sources
        }
    print("All JSON extraction strategies failed")
    return {}

# Generate search queries based on the input question
def generate_queries(state: RAGResearchState) -> RAGResearchState:
    """Generate search queries based on the input question"""
    prompt = f"""
    Task: Generate 4 diverse search queries for researching this question.
    Question: "{state.question}"
    
    Instructions:
    - Create 8 different search queries that explore various aspects of the question
    - Use different phrasings and keywords to ensure comprehensive coverage
    - Make each query clear and specific
    
    Format your response as a valid JSON object like this:
    {{
      "queries": [
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        ""
      ]
    }}
    
    Output JSON:
    """
    
    try:
        response = llm.invoke(prompt)
        print(f"Query generation response:\n{response}")
        
        # Extract and validate queries from response
        data = extract_json_from_response(response)
        
        if "queries" in data and isinstance(data["queries"], list) and len(data["queries"]) > 0:
            # Filter out any placeholder text that might have been copied from the prompt
            filtered_queries = [
                q for q in data["queries"] 
            ]
            
            if filtered_queries:
                state.queries = filtered_queries
            else:
                state.queries = data["queries"]
                
            # Print the actual queries that will be used
            print(f"Using queries: {state.queries}")
        else:
            raise ValueError("No valid queries found in response")
            
    except Exception as e:
        print(f"Error generating queries: {e}")
        # Fallback queries based on the original question
        state.queries = [
            state.question, 
            f"{state.question} latest developments", 
            f"{state.question} recent breakthroughs", 
            f"{state.question} current status"
        ]
        print(f"Using fallback queries: {state.queries}")
        
    return state

# Execute web searches for each query
def perform_searches(state: RAGResearchState) -> RAGResearchState:
    """Run web searches and collect results."""
    all_results = []
    
    for query in state.queries:
        try:
            print(f"Searching for: {query}")
            raw_results = tavily_tool.invoke(query)
            
            # Process search results
            if isinstance(raw_results, list):
                for item in raw_results:
                    if isinstance(item, dict) and all(k in item for k in ["title", "content", "url"]):
                        all_results.append(item)
            elif isinstance(raw_results, dict) and "results" in raw_results:
                # Handle case where results might be nested
                for item in raw_results["results"]:
                    if isinstance(item, dict) and all(k in item for k in ["title", "content", "url"]):
                        all_results.append(item)
                        
        except Exception as e:
            print(f"Search error for query '{query}': {e}")
    
    # Remove duplicates based on URLs
    urls_seen = set()
    unique_results = []
    for result in all_results:
        if result["url"] not in urls_seen:
            urls_seen.add(result["url"])
            unique_results.append(result)
    
    state.results = unique_results
    print(f"Found {len(state.results)} unique search results")
    return state

# Convert search results into documents for vector store
def build_rag_database(state: RAGResearchState) -> RAGResearchState:
    """Process search results into documents and store in vector database."""
    documents = []
    
    for result in state.results:
        content = f"Title: {result['title']}\nContent: {result['content']}"
        metadata = {
            "url": result["url"],
            "title": result["title"],
            "question": state.question
        }
        
        doc_chunks = text_splitter.create_documents(
            texts=[content],
            metadatas=[metadata]
        )
        documents.extend(doc_chunks)
    
    state.rag_documents = [
        {"content": doc.page_content, "metadata": doc.metadata} 
        for doc in documents
    ]
    
    if documents:
        print(f"Adding {len(documents)} documents to vector store")
        db.add_documents(documents)
        try:
            db.persist()
        except Exception as e:
            print(f"Persistence note: {e}")
    else:
        print("No documents to add to vector store")
    
    return state

# Retrieve relevant documents from vector store
def retrieve_relevant_context(state: RAGResearchState) -> RAGResearchState:
    """Fetch relevant context for the question from the vector database."""
    try:
        results = db.similarity_search(
            state.question,
            k=7,
            filter=None
        )
        
        context_parts = []
        for i, doc in enumerate(results):
            source = doc.metadata.get('url', 'Unknown')
            context_parts.append(f"[Document {i+1}]\nContent: {doc.page_content}\nSource: {source}")
        
        state.relevant_context = "\n\n".join(context_parts)
        print(f"Retrieved {len(results)} relevant documents from vector store")
    except Exception as e:
        print(f"Error retrieving context: {e}")
        state.relevant_context = "No relevant context could be retrieved from the database."
    
    return state

# Analyze results to extract insights and themes
def analyze_results(state: RAGResearchState) -> RAGResearchState:
    """Extract insights and themes from search results and context."""
    # Build a context from search results if we have them
    search_context = ""
    if state.results:
        search_summaries = []
        for i, result in enumerate(state.results[:5]):  # Limit to top 5 results
            summary = f"Result {i+1}: {result['title']}\nURL: {result['url']}\nSummary: {result['content'][:200]}..."
            search_summaries.append(summary)
        search_context = "\n\n".join(search_summaries)
    
    prompt = f"""
    Task: Analyze research materials on: "{state.question}"
    
    Available materials:
    
    SEARCH RESULTS:
    {search_context}
    
    RETRIEVED CONTEXT:
    {state.relevant_context[:2000] if state.relevant_context else "No additional context available."}
    
    Instructions:
    1. Identify 3-5 key insights related to {state.question}
    2. Identify 2-4 major themes related to {state.question}
    3. List all unique source URLs mentioned
    
    Format your response as a valid JSON object like this:
    {{
      "insights": [
        "First key insight",
        "Second key insight",
        "Third key insight"
      ],
      "themes": [
        "First major theme",
        "Second major theme"
      ],
      "sources": [
        "https://example.com/source1",
        "https://example.com/source2"
      ]
    }}
    
    Output JSON:
    """
    
    try:
        response = llm.invoke(prompt)
        print(f"Analysis response:\n{response}")
        
        data = extract_json_from_response(response)
        
        if "insights" in data and data["insights"]:
            state.insights = data["insights"]
        else:
            state.insights = ["No clear insights could be extracted from the available information."]
        
        if "themes" in data and data["themes"]:
            state.themes = data["themes"]
        else:
            state.themes = ["Insufficient information to identify clear themes."]
        
        unique_sources = set()
        
        for r in state.results:
            unique_sources.add(r["url"])
        
        if "sources" in data and data["sources"]:
            for source in data["sources"]:
                if isinstance(source, str) and source.startswith("http"):
                    unique_sources.add(source)
                    
        state.sources = list(set(state.sources))
        
    except Exception as e:
        print(f"Analysis error: {e}")
        state.insights = ["Error extracting insights from the available information."]
        state.themes = ["Error identifying themes from the available information."]
        
        state.sources = [r["url"] for r in state.results]
    
    return state

# Generate a final answer based on research
def draft_answer(state: RAGResearchState) -> RAGResearchState:
    """Draft a comprehensive answer using insights, themes, and sources."""
    
    insights_text = "\n".join([f"- {insight}" for insight in state.insights])
    themes_text = "\n".join([f"- {theme}" for theme in state.themes])
    
    sources_with_titles = []
    urls_seen = set()
    
    for result in state.results:
        if result["url"] not in urls_seen:
            urls_seen.add(result["url"])
            sources_with_titles.append(f"{result['url']} - {result['title']}")
    
    for source in state.sources:
        if source not in urls_seen and source.startswith("http"):
            urls_seen.add(source)
            sources_with_titles.append(source)
    
    sources_text = ""
    for i, source in enumerate(sources_with_titles, 1):
        sources_text += f"{i}. {source}\n"
    
    prompt = f"""
    Task: Write a comprehensive answer to the question: "{state.question}"
    
    Based on the following research:
    
    KEY INSIGHTS:
    {insights_text}
    
    MAJOR THEMES:
    {themes_text}
    
    Please write a well-structured, informative answer that:
    - Is approximately 300-400 words
    - Directly addresses the question
    - Incorporates the key insights and themes
    - Cites sources using numbered citations [1], [2], etc.
    - Has a clear introduction, body, and conclusion
    
    Available sources to cite:
    {sources_text}
    """
    
    try:
        state.answer = llm.invoke(prompt)
        print("Answer generated successfully")
    except Exception as e:
        print(f"Error generating answer: {e}")
        state.answer = f"""
        Unable to generate a complete answer due to technical issues.
        
        Here are the key insights found during research:
        {insights_text}
        
        Main themes identified:
        {themes_text}
        
        Sources:
        {sources_text}
        """
    
    return state

# Set up the LangGraph workflow
def create_research_graph():
    graph_builder = StateGraph(RAGResearchState)
    
    # Add all nodes
    graph_builder.add_node("generate_queries", generate_queries)
    graph_builder.add_node("perform_searches", perform_searches)
    graph_builder.add_node("build_rag_database", build_rag_database)
    graph_builder.add_node("retrieve_relevant_context", retrieve_relevant_context)
    graph_builder.add_node("analyze_results", analyze_results)
    graph_builder.add_node("draft_answer", draft_answer)
    
    # Define workflow
    graph_builder.add_edge("generate_queries", "perform_searches")
    graph_builder.add_edge("perform_searches", "build_rag_database")
    graph_builder.add_edge("build_rag_database", "retrieve_relevant_context")
    graph_builder.add_edge("retrieve_relevant_context", "analyze_results")
    graph_builder.add_edge("analyze_results", "draft_answer")
    
    # Set entry point
    graph_builder.set_entry_point("generate_queries")
    
    # Compile graph
    return graph_builder.compile()

# Run the full research workflow
def run_research(question: str) -> RAGResearchState:
    """Run a complete research workflow for the given question"""
    print(f"\n{'='*50}")
    print(f"Starting research on: {question}")
    print(f"{'='*50}\n")
    
    initial_state = RAGResearchState(question=question)
    research_graph = create_research_graph()
    
    try:
        final_state_dict = research_graph.invoke(initial_state)
        final_state = RAGResearchState(**final_state_dict)
        
        print("\n=== Research Results ===")
        print(f"Question: {final_state.question}")
        print(f"\nQueries used ({len(final_state.queries)}):")
        for i, query in enumerate(final_state.queries, 1):
            print(f"{i}. {query}")
            
        print(f"\nSources found ({len(final_state.sources)}):")
        for i, source in enumerate(final_state.sources, 1):
            print(f"{i}. {source}")
            
        print("\n=== Final Answer ===")
        print(final_state.answer)
        
        return final_state
    
    except Exception as e:
        print(f"Error during research workflow: {e}")
        return initial_state

if __name__ == "__main__":
    question = input("Enter your research question: ")
    if not question:
        question = "What are the latest developments in nuclear fusion technology?"
    
    final_state = run_research(question)
