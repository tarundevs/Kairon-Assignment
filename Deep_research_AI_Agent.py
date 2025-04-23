import os
from typing import List, Dict
from pydantic import BaseModel
from langchain_tavily import TavilySearch
from langchain.llms import HuggingFaceHub
from langgraph.graph import StateGraph
import json

# Set up environment variables for API keys
# Note: In a production environment, use secure key management
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY", "your-tavily-api-key")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN", "your-hf-api-token")

# Define the state model
class ResearchState(BaseModel):
    question: str
    queries: List[str] = []
    results: List[Dict] = []
    insights: List[str] = []
    themes: List[str] = []
    sources: List[str] = []
    answer: str = ""

# Initialize LLM and Tavily tool
llm = HuggingFaceHub(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    model_kwargs={"temperature": 0.2, "max_length": 500}
)
tavily_tool = TavilySearch(max_results=3)

# Research Agent Nodes
def generate_queries(state: ResearchState) -> ResearchState:
    prompt = f"Generate 3 search queries for the question: {state.question}. Return only a JSON object: {{ \"queries\": [\"query1\", \"query2\", \"query3\"] }}"
    try:
        response = llm(prompt)
        data = json.loads(response)
        state.queries = data.get("queries", [])
    except:
        state.queries = [state.question, f"{state.question} 2025", f"{state.question} recent developments"]
    return state

def perform_searches(state: ResearchState) -> ResearchState:
    for query in state.queries:
        try:
            raw_results = tavily_tool.invoke(query)
            parsed_results = []
            if isinstance(raw_results, list):
                for item in raw_results:
                    if isinstance(item, dict) and all(k in item for k in ["title", "content", "url"]):
                        parsed_results.append(item)
                    elif isinstance(item, str):
                        parsed_results.append({
                            "title": f"Result for {query}",
                            "content": item[:500],
                            "url": "https://example.com"
                        })
            elif isinstance(raw_results, str):
                parsed_results.append({
                    "title": f"Result for {query}",
                    "content": raw_results[:500],
                    "url": "https://example.com"
                })
            state.results.extend(parsed_results)
        except Exception as e:
            print(f"Search error for query {query}: {e}")
    return state

def analyze_results(state: ResearchState) -> ResearchState:
    all_content = "\n".join([f"Title: {r['title']}\nContent: {r['content']}\nURL: {r['url']}" for r in state.results])
    prompt = f"Analyze the following research results for the question: {state.question}\nResults:\n{all_content}\nReturn JSON: {{ \"insights\": [], \"themes\": [], \"sources\": [] }}"
    try:
        response = llm(prompt)
        data = json.loads(response)
        state.insights = data.get("insights", [])
        state.themes = data.get("themes", [])
        state.sources = [r["url"] for r in state.results]
    except:
        state.insights = ["No insights extracted"]
        state.themes = ["No themes identified"]
        state.sources = [r["url"] for r in state.results]
    return state

# Answer Drafter Agent Node
def draft_answer(state: ResearchState) -> ResearchState:
    prompt = (
        f"Write a 200-300 word answer for the question: {state.question}\n"
        f"Using insights: {state.insights}\n"
        f"Themes: {state.themes}\n"
        f"Sources: {state.sources}\n"
        "Cite sources as [1], [2], etc. Ensure the answer is coherent, concise, and informative."
    )
    try:
        response = llm(prompt)
        state.answer = response
    except Exception as e:
        state.answer = f"Failed to generate answer: {e}"
    return state

# Create the LangGraph workflow with dual-agent structure
graph_builder = StateGraph(ResearchState)

# Research Agent nodes
graph_builder.add_node("generate_queries", generate_queries)
graph_builder.add_node("perform_searches", perform_searches)
graph_builder.add_node("analyze_results", analyze_results)

# Answer Drafter Agent node
graph_builder.add_node("draft_answer", draft_answer)

# Define edges for workflow
graph_builder.add_edge("generate_queries", "perform_searches")
graph_builder.add_edge("perform_searches", "analyze_results")
graph_builder.add_edge("analyze_results", "draft_answer")

# Set entry point
graph_builder.set_entry_point("generate_queries")

# Compile graph
research_graph = graph_builder.compile()

# Example usage
if __name__ == "__main__":
    initial_state = ResearchState(question="latest advancements in quantum computing")
    final_state_dict = research_graph.invoke(initial_state)
    final_state = ResearchState(**final_state_dict)
    print("\n=== Final Answer ===")
    print(final_state.answer)
    print("\n=== Sources ===")
    for i, source in enumerate(final_state.sources, 1):
        print(f"[{i}] {source}")
