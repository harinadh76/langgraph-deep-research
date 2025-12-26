"""
VERSION 1: Simple Research
==========================
Goal: Search the web for a query and summarize the results

This teaches:
- How to use Tavily for web search
- How to integrate tools with LangGraph
- Basic research workflow
"""

import os
from dotenv import load_dotenv

# Load environment variables FIRST
load_dotenv()

# Now import the rest
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
import operator
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage

# ============================================
# STEP 1: Understand our tools
# ============================================

print("""
┌─────────────────────────────────────────────────────────────────┐
│                  VERSION 1: SIMPLE RESEARCH                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌───────────┐     ┌───────────┐     ┌───────────┐            │
│   │   USER    │────▶│  SEARCH   │────▶│ SUMMARIZE │            │
│   │   QUERY   │     │    🔍     │     │    📝     │            │
│   └───────────┘     └───────────┘     └───────────┘            │
│                           │                 │                    │
│                           ▼                 ▼                    │
│                      Web Results      Final Summary              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
""")


# ============================================
# STEP 2: Setup our tools
# ============================================

# Initialize the LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",  # Affordable and capable
    temperature=0.3       # Lower = more focused/factual
)

# Initialize the search tool
# Tavily is designed for AI - it returns clean, relevant results
search_tool = TavilySearchResults(
    max_results=5,           # Get 5 search results
    search_depth="basic",    # "basic" or "advanced"
    include_answer=True,     # Include a direct answer if available
    include_raw_content=False  # We don't need raw HTML
)


# ============================================
# STEP 3: Define our State
# ============================================

class ResearchState(TypedDict):
    """
    This holds all the data that flows through our graph
    """
    # The user's research question
    query: str
    
    # Raw search results from Tavily
    search_results: list
    
    # Our final summary
    summary: str
    
    # Keep track of sources
    sources: list


# ============================================
# STEP 4: Define our Nodes (Functions)
# ============================================

def search_node(state: ResearchState) -> dict:
    """
    Node 1: Search the web using Tavily
    
    This node:
    1. Takes the user's query
    2. Searches the web
    3. Returns the results
    """
    print("\n🔍 Searching the web...")
    print(f"   Query: {state['query']}")
    
    # Perform the search
    results = search_tool.invoke(state["query"])
    
    # Extract sources (URLs) from results
    sources = []
    for result in results:
        if isinstance(result, dict) and "url" in result:
            sources.append(result["url"])
    
    print(f"   Found {len(results)} results")
    
    return {
        "search_results": results,
        "sources": sources
    }


def summarize_node(state: ResearchState) -> dict:
    """
    Node 2: Summarize the search results using LLM
    
    This node:
    1. Takes the search results
    2. Asks the LLM to create a summary
    3. Returns the summary
    """
    print("\n📝 Summarizing findings...")
    
    # Format search results for the LLM
    search_content = ""
    for i, result in enumerate(state["search_results"], 1):
        if isinstance(result, dict):
            title = result.get("title", "No title")
            content = result.get("content", "No content")
            url = result.get("url", "No URL")
            search_content += f"""
Result {i}:
Title: {title}
Content: {content}
Source: {url}
---
"""
    
    # Create the prompt for summarization
    messages = [
        SystemMessage(content="""You are a research assistant. 
        Summarize the search results into a clear, informative response.
        Include key facts and cite sources where appropriate.
        Be comprehensive but concise."""),
        HumanMessage(content=f"""
Research Question: {state['query']}

Search Results:
{search_content}

Please provide a well-organized summary that answers the research question.
Include the most important information and cite sources.
""")
    ]
    
    # Get the summary from LLM
    response = llm.invoke(messages)
    
    print("   Summary complete!")
    
    return {"summary": response.content}


# ============================================
# STEP 5: Build the Graph
# ============================================

def build_simple_research_graph():
    """Build and compile the research graph"""
    
    # Create the graph
    graph_builder = StateGraph(ResearchState)
    
    # Add nodes
    graph_builder.add_node("search", search_node)
    graph_builder.add_node("summarize", summarize_node)
    
    # Add edges (define the flow)
    graph_builder.add_edge(START, "search")
    graph_builder.add_edge("search", "summarize")
    graph_builder.add_edge("summarize", END)
    
    # Compile and return
    return graph_builder.compile()


# ============================================
# STEP 6: Run the Research
# ============================================

def do_research(query: str) -> dict:
    """
    Main function to run a research query
    """
    # Build the graph
    graph = build_simple_research_graph()
    
    # Create initial state
    initial_state = {
        "query": query,
        "search_results": [],
        "summary": "",
        "sources": []
    }
    
    # Run the graph
    result = graph.invoke(initial_state)
    
    return result


# ============================================
# STEP 7: Main Execution
# ============================================

if __name__ == "__main__":
    # Example research queries to try
    queries = [
        "What are the latest developments in AI agents in 2024?",
        # "What is quantum computing and how does it work?",
        # "What are the health benefits of intermittent fasting?",
    ]
    
    for query in queries:
        print("\n" + "="*70)
        print(f"📚 RESEARCH QUERY: {query}")
        print("="*70)
        
        # Run the research
        result = do_research(query)
        
        # Display results
        print("\n" + "-"*70)
        print("📋 RESEARCH SUMMARY:")
        print("-"*70)
        print(result["summary"])
        
        print("\n" + "-"*70)
        print("🔗 SOURCES:")
        print("-"*70)
        for i, source in enumerate(result["sources"], 1):
            print(f"   {i}. {source}")
        
        print("\n")