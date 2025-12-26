"""
VERSION 2: Multi-Query Research
================================
Goal: Break down a complex question into sub-questions,
      research each one, then combine the findings.

This teaches:
- How to decompose complex questions
- How to loop through multiple queries
- How to combine multiple research results
"""

import os
from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, List
import operator
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage
import json

# ============================================
# Architecture Diagram
# ============================================

print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│                    VERSION 2: MULTI-QUERY RESEARCH                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────┐                                                           │
│   │    USER     │                                                           │
│   │    QUERY    │                                                           │
│   └──────┬──────┘                                                           │
│          │                                                                   │
│          ▼                                                                   │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                         PLANNER                                      │   │
│   │                           🧠                                         │   │
│   │   "What are the effects of climate change?"                         │   │
│   │                            │                                         │   │
│   │                            ▼                                         │   │
│   │   Sub-questions:                                                     │   │
│   │   1. What causes climate change?                                     │   │
│   │   2. What are environmental effects?                                 │   │
│   │   3. What are economic impacts?                                      │   │
│   │   4. What solutions exist?                                           │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                            │                                                 │
│                            ▼                                                 │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                       RESEARCHER                                     │   │
│   │                          🔍                                          │   │
│   │                                                                      │   │
│   │   For EACH sub-question:                                             │   │
│   │   ┌────────────┐   ┌────────────┐   ┌────────────┐   ┌────────────┐ │   │
│   │   │  Search 1  │   │  Search 2  │   │  Search 3  │   │  Search 4  │ │   │
│   │   │   🔍       │   │   🔍       │   │   🔍       │   │   🔍       │ │   │
│   │   └────────────┘   └────────────┘   └────────────┘   └────────────┘ │   │
│   │        │                │                │                │          │   │
│   │        ▼                ▼                ▼                ▼          │   │
│   │   [Results 1]      [Results 2]      [Results 3]      [Results 4]    │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                            │                                                 │
│                            ▼                                                 │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                        WRITER                                        │   │
│   │                          ✍️                                          │   │
│   │   Combine all findings into comprehensive report                    │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                            │                                                 │
│                            ▼                                                 │
│                    ┌──────────────┐                                         │
│                    │    FINAL     │                                         │
│                    │    REPORT    │                                         │
│                    │      📄      │                                         │
│                    └──────────────┘                                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
""")


# ============================================
# Setup Tools
# ============================================

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
search_tool = TavilySearchResults(max_results=3)


# ============================================
# State Definition
# ============================================

class DeepResearchState(TypedDict):
    # Original query from user
    original_query: str
    
    # List of sub-questions generated by planner
    sub_questions: List[str]
    
    # Current sub-question being researched (index)
    current_question_index: int
    
    # Research findings for each sub-question
    # Format: {"question": "...", "findings": "...", "sources": [...]}
    research_findings: Annotated[List[dict], operator.add]
    
    # All sources collected
    all_sources: Annotated[List[str], operator.add]
    
    # Final report
    final_report: str


# ============================================
# Node 1: Planner
# ============================================

def planner_node(state: DeepResearchState) -> dict:
    """
    Breaks down the main query into sub-questions
    """
    print("\n🧠 PLANNER: Analyzing query and creating sub-questions...")
    
    messages = [
        SystemMessage(content="""You are a research planner. 
        Your job is to break down a complex research question into 3-5 specific sub-questions.
        
        Each sub-question should:
        - Cover a different aspect of the topic
        - Be specific enough to search for
        - Together, fully answer the main question
        
        Return ONLY a JSON array of questions, like:
        ["Question 1?", "Question 2?", "Question 3?"]
        """),
        HumanMessage(content=f"Break down this research question: {state['original_query']}")
    ]
    
    response = llm.invoke(messages)
    
    # Parse the JSON response
    try:
        # Try to extract JSON from the response
        content = response.content
        # Find the JSON array in the response
        start = content.find('[')
        end = content.rfind(']') + 1
        if start != -1 and end != 0:
            sub_questions = json.loads(content[start:end])
        else:
            # Fallback: just use the original query
            sub_questions = [state['original_query']]
    except json.JSONDecodeError:
        sub_questions = [state['original_query']]
    
    print(f"\n   Generated {len(sub_questions)} sub-questions:")
    for i, q in enumerate(sub_questions, 1):
        print(f"   {i}. {q}")
    
    return {
        "sub_questions": sub_questions,
        "current_question_index": 0
    }


# ============================================
# Node 2: Researcher
# ============================================

def researcher_node(state: DeepResearchState) -> dict:
    """
    Researches the current sub-question
    """
    current_index = state["current_question_index"]
    current_question = state["sub_questions"][current_index]
    
    print(f"\n🔍 RESEARCHER: Investigating question {current_index + 1}/{len(state['sub_questions'])}")
    print(f"   Question: {current_question}")
    
    # Search for this question
    search_results = search_tool.invoke(current_question)
    
    # Extract content and sources
    content_parts = []
    sources = []
    
    for result in search_results:
        if isinstance(result, dict):
            content_parts.append(result.get("content", ""))
            if "url" in result:
                sources.append(result["url"])
    
    combined_content = "\n".join(content_parts)
    
    # Summarize the findings for this question
    messages = [
        SystemMessage(content="""Summarize these search results into clear findings.
        Focus on facts and key information. Be concise but comprehensive."""),
        HumanMessage(content=f"""
Question: {current_question}

Search Results:
{combined_content}

Provide a focused summary of the findings:""")
    ]
    
    summary = llm.invoke(messages)
    
    print(f"   ✓ Found {len(sources)} sources")
    
    # Create finding object
    finding = {
        "question": current_question,
        "findings": summary.content,
        "sources": sources
    }
    
    return {
        "research_findings": [finding],
        "all_sources": sources,
        "current_question_index": current_index + 1
    }


# ============================================
# Node 3: Should Continue Researching?
# ============================================

def should_continue_research(state: DeepResearchState) -> str:
    """
    Decides if we should research more questions or move to writing
    """
    current_index = state["current_question_index"]
    total_questions = len(state["sub_questions"])
    
    if current_index < total_questions:
        print(f"\n   📊 Progress: {current_index}/{total_questions} questions researched")
        return "continue_research"
    else:
        print(f"\n   ✅ All {total_questions} questions researched!")
        return "write_report"


# ============================================
# Node 4: Writer
# ============================================

def writer_node(state: DeepResearchState) -> dict:
    """
    Combines all findings into a comprehensive report
    """
    print("\n✍️  WRITER: Composing final report...")
    
    # Format all findings
    findings_text = ""
    for i, finding in enumerate(state["research_findings"], 1):
        findings_text += f"""
## Section {i}: {finding['question']}

{finding['findings']}

Sources: {', '.join(finding['sources'][:2]) if finding['sources'] else 'N/A'}

---
"""
    
    # Generate the final report
    messages = [
        SystemMessage(content="""You are a research report writer.
        Create a comprehensive, well-organized report based on the research findings.
        
        Structure your report with:
        1. Executive Summary (2-3 sentences)
        2. Key Findings (organized by topic)
        3. Conclusion
        
        Make it readable and informative. Use markdown formatting."""),
        HumanMessage(content=f"""
Original Research Question: {state['original_query']}

Research Findings:
{findings_text}

Write a comprehensive research report:""")
    ]
    
    report = llm.invoke(messages)
    
    print("   ✓ Report complete!")
    
    return {"final_report": report.content}


# ============================================
# Build the Graph
# ============================================

def build_multi_query_graph():
    """Build the multi-query research graph"""
    
    graph_builder = StateGraph(DeepResearchState)
    
    # Add nodes
    graph_builder.add_node("planner", planner_node)
    graph_builder.add_node("researcher", researcher_node)
    graph_builder.add_node("writer", writer_node)
    
    # Add edges
    graph_builder.add_edge(START, "planner")
    graph_builder.add_edge("planner", "researcher")
    
    # Conditional edge: continue research or write report
    graph_builder.add_conditional_edges(
        "researcher",
        should_continue_research,
        {
            "continue_research": "researcher",  # Loop back
            "write_report": "writer"            # Move to writer
        }
    )
    
    graph_builder.add_edge("writer", END)
    
    return graph_builder.compile()


# ============================================
# Main Function
# ============================================

def deep_research(query: str) -> dict:
    """Run deep research on a query"""
    
    print("\n" + "="*70)
    print("🚀 STARTING DEEP RESEARCH")
    print("="*70)
    print(f"\n📌 Research Topic: {query}")
    
    # Build graph
    graph = build_multi_query_graph()
    
    # Initial state
    initial_state = {
        "original_query": query,
        "sub_questions": [],
        "current_question_index": 0,
        "research_findings": [],
        "all_sources": [],
        "final_report": ""
    }
    
    # Run the graph
    result = graph.invoke(initial_state)
    
    return result


# ============================================
# Run It!
# ============================================

if __name__ == "__main__":
    # Try different research topics
    query = "What are the potential benefits and risks of artificial general intelligence (AGI)?"
    
    result = deep_research(query)
    
    print("\n" + "="*70)
    print("📄 FINAL RESEARCH REPORT")
    print("="*70)
    print(result["final_report"])
    
    print("\n" + "="*70)
    print("🔗 ALL SOURCES")
    print("="*70)
    for i, source in enumerate(set(result["all_sources"]), 1):
        print(f"  {i}. {source}")