"""
VERSION 3: Full Deep Research System
=====================================
A complete deep research system with:
- Supervisor coordinating the workflow
- Specialized agents for different tasks
- Quality checking
- Iterative refinement
"""

import os
from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, List, Literal
import operator
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import json
from datetime import datetime

# ============================================
# Architecture
# ============================================

print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│                  VERSION 3: FULL DEEP RESEARCH SYSTEM                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                            ┌─────────────┐                                  │
│                            │    USER     │                                  │
│                            │    QUERY    │                                  │
│                            └──────┬──────┘                                  │
│                                   │                                          │
│                                   ▼                                          │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                          SUPERVISOR                                     │ │
│  │                              👔                                         │ │
│  │   Decides: What to do next?                                            │ │
│  │   • Need to plan? → Planner                                            │ │
│  │   • Need research? → Researcher                                        │ │
│  │   • Need to analyze? → Analyst                                         │ │
│  │   • Ready to write? → Writer                                           │ │
│  │   • Done? → Finish                                                     │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│         │              │              │              │                       │
│         ▼              ▼              ▼              ▼                       │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐                   │
│   │ PLANNER  │  │RESEARCHER│  │ ANALYST  │  │  WRITER  │                   │
│   │    🧠    │  │    🔍    │  │    📊    │  │    ✍️    │                   │
│   │          │  │          │  │          │  │          │                   │
│   │ Creates  │  │ Searches │  │ Analyzes │  │ Writes   │                   │
│   │ research │  │ the web  │  │ findings │  │ final    │                   │
│   │ plan     │  │          │  │          │  │ report   │                   │
│   └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘                   │
│        │             │             │             │                          │
│        └─────────────┴──────┬──────┴─────────────┘                          │
│                             │                                                │
│                             ▼                                                │
│                    Back to SUPERVISOR                                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
""")


# ============================================
# Setup
# ============================================

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
search_tool = TavilySearchResults(max_results=5, search_depth="advanced")

# Workers available
WORKERS = ["planner", "researcher", "analyst", "writer"]


# ============================================
# State
# ============================================

class DeepResearchState(TypedDict):
    # Input
    original_query: str
    
    # Research plan
    research_plan: dict  # Contains sub-questions and strategy
    
    # Research data
    sub_questions: List[str]
    current_question_index: int
    research_data: Annotated[List[dict], operator.add]
    
    # Analysis
    analysis: str
    
    # Output
    final_report: str
    
    # Control flow
    next_worker: str
    iteration: int
    messages: Annotated[List[str], operator.add]
    
    # Sources
    sources: Annotated[List[str], operator.add]


# ============================================
# Supervisor
# ============================================

def supervisor_node(state: DeepResearchState) -> dict:
    """
    The Supervisor decides what to do next
    """
    iteration = state.get("iteration", 0)
    
    print(f"\n👔 SUPERVISOR (Iteration {iteration + 1})")
    print("   Analyzing current state...")
    
    # Build context for decision
    has_plan = bool(state.get("research_plan"))
    has_questions = bool(state.get("sub_questions"))
    questions_researched = state.get("current_question_index", 0)
    total_questions = len(state.get("sub_questions", []))
    has_research = bool(state.get("research_data"))
    has_analysis = bool(state.get("analysis"))
    has_report = bool(state.get("final_report"))
    
    # Decision logic
    if not has_plan:
        decision = "planner"
        reason = "Need to create research plan"
    elif has_questions and questions_researched < total_questions:
        decision = "researcher"
        reason = f"Need to research question {questions_researched + 1}/{total_questions}"
    elif has_research and not has_analysis:
        decision = "analyst"
        reason = "Need to analyze research findings"
    elif has_analysis and not has_report:
        decision = "writer"
        reason = "Ready to write final report"
    else:
        decision = "FINISH"
        reason = "Research complete"
    
    print(f"   Decision: {decision}")
    print(f"   Reason: {reason}")
    
    return {
        "next_worker": decision,
        "iteration": iteration + 1,
        "messages": [f"Supervisor: Moving to {decision} - {reason}"]
    }


def route_from_supervisor(state: DeepResearchState) -> str:
    """Route based on supervisor's decision"""
    next_worker = state.get("next_worker", "FINISH")
    
    if next_worker == "planner":
        return "planner"
    elif next_worker == "researcher":
        return "researcher"
    elif next_worker == "analyst":
        return "analyst"
    elif next_worker == "writer":
        return "writer"
    else:
        return "finish"


# ============================================
# Planner Agent
# ============================================

def planner_agent(state: DeepResearchState) -> dict:
    """
    Creates a comprehensive research plan
    """
    print("\n🧠 PLANNER AGENT")
    print("   Creating research plan...")
    
    prompt = f"""You are a research planning expert.
    
Create a comprehensive research plan for this query:
"{state['original_query']}"

Your plan should include:
1. Main research objective (1 sentence)
2. Key aspects to investigate (3-5 areas)
3. Specific sub-questions to research (4-6 questions)
4. Expected deliverable description

Return as JSON:
{{
    "objective": "...",
    "key_aspects": ["aspect1", "aspect2", ...],
    "sub_questions": ["question1?", "question2?", ...],
    "deliverable": "..."
}}
"""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    try:
        # Parse the JSON from response
        content = response.content
        start = content.find('{')
        end = content.rfind('}') + 1
        plan = json.loads(content[start:end])
    except:
        plan = {
            "objective": state['original_query'],
            "key_aspects": ["General research"],
            "sub_questions": [state['original_query']],
            "deliverable": "Research report"
        }
    
    print(f"   ✓ Created plan with {len(plan.get('sub_questions', []))} sub-questions")
    
    for i, q in enumerate(plan.get('sub_questions', []), 1):
        print(f"      {i}. {q}")
    
    return {
        "research_plan": plan,
        "sub_questions": plan.get("sub_questions", []),
        "current_question_index": 0,
        "messages": [f"Planner: Created plan with {len(plan.get('sub_questions', []))} questions"]
    }


# ============================================
# Researcher Agent
# ============================================

def researcher_agent(state: DeepResearchState) -> dict:
    """
    Researches the current sub-question
    """
    current_idx = state.get("current_question_index", 0)
    questions = state.get("sub_questions", [])
    
    if current_idx >= len(questions):
        return {"messages": ["Researcher: All questions already researched"]}
    
    current_question = questions[current_idx]
    
    print(f"\n🔍 RESEARCHER AGENT")
    print(f"   Researching: {current_question}")
    print(f"   Progress: {current_idx + 1}/{len(questions)}")
    
    # Perform web search
    try:
        search_results = search_tool.invoke(current_question)
    except Exception as e:
        print(f"   ⚠️ Search error: {e}")
        search_results = []
    
    # Process results
    sources = []
    content_pieces = []
    
    for result in search_results:
        if isinstance(result, dict):
            content_pieces.append(result.get("content", ""))
            if "url" in result:
                sources.append(result["url"])
    
    combined_content = "\n\n".join(content_pieces)
    
    # Summarize findings
    summary_prompt = f"""Summarize these search results for the question:
"{current_question}"

Search Results:
{combined_content[:4000]}  # Limit content length

Provide a comprehensive but focused summary (200-300 words).
Include specific facts, statistics, and findings.
"""
    
    summary = llm.invoke([HumanMessage(content=summary_prompt)])
    
    research_item = {
        "question": current_question,
        "summary": summary.content,
        "sources": sources,
        "raw_results_count": len(search_results)
    }
    
    print(f"   ✓ Found {len(sources)} sources")
    
    return {
        "research_data": [research_item],
        "sources": sources,
        "current_question_index": current_idx + 1,
        "messages": [f"Researcher: Completed research on '{current_question[:50]}...'"]
    }


# ============================================
# Analyst Agent
# ============================================

def analyst_agent(state: DeepResearchState) -> dict:
    """
    Analyzes all research findings
    """
    print("\n📊 ANALYST AGENT")
    print("   Analyzing all research findings...")
    
    research_data = state.get("research_data", [])
    
    # Compile all research
    research_summary = ""
    for i, item in enumerate(research_data, 1):
        research_summary += f"""
### Research {i}: {item['question']}
{item['summary']}
Sources: {len(item.get('sources', []))}

"""
    
    analysis_prompt = f"""You are a research analyst.

Analyze these research findings for the query:
"{state['original_query']}"

Research Findings:
{research_summary}

Provide a thorough analysis including:
1. Key themes and patterns across all findings
2. Important insights and conclusions
3. Any contradictions or gaps in the research
4. Recommendations based on the findings

Be analytical and synthesize information across all sources.
"""
    
    analysis = llm.invoke([HumanMessage(content=analysis_prompt)])
    
    print("   ✓ Analysis complete")
    
    return {
        "analysis": analysis.content,
        "messages": ["Analyst: Completed analysis of all findings"]
    }


# ============================================
# Writer Agent
# ============================================

def writer_agent(state: DeepResearchState) -> dict:
    """
    Writes the final research report
    """
    print("\n✍️  WRITER AGENT")
    print("   Composing final report...")
    
    research_data = state.get("research_data", [])
    analysis = state.get("analysis", "")
    plan = state.get("research_plan", {})
    
    # Compile research sections
    research_sections = ""
    for i, item in enumerate(research_data, 1):
        research_sections += f"""
## {item['question']}

{item['summary']}

"""
    
    # Compile sources
    all_sources = list(set(state.get("sources", [])))
    sources_text = "\n".join([f"- {src}" for src in all_sources[:15]])
    
    report_prompt = f"""You are an expert research report writer.

Write a comprehensive research report based on this information:

**Original Question:** {state['original_query']}

**Research Objective:** {plan.get('objective', 'N/A')}

**Research Findings:**
{research_sections}

**Analysis:**
{analysis}

**Sources Available:**
{sources_text}

Create a well-structured research report with:
1. **Executive Summary** - Brief overview of findings (3-4 sentences)
2. **Introduction** - Context and importance of the topic
3. **Key Findings** - Main discoveries organized by theme
4. **Analysis** - Deeper insights and patterns
5. **Conclusion** - Summary and implications
6. **Sources** - List of references

Use markdown formatting. Be comprehensive but readable.
Target length: 800-1200 words.
"""
    
    report = llm.invoke([HumanMessage(content=report_prompt)])
    
    # Add header with metadata
    header = f"""# Research Report
**Query:** {state['original_query']}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Sources Consulted:** {len(all_sources)}

---

"""
    
    final_report = header + report.content
    
    print("   ✓ Report complete!")
    
    return {
        "final_report": final_report,
        "messages": ["Writer: Completed final research report"]
    }


# ============================================
# Finish Node
# ============================================

def finish_node(state: DeepResearchState) -> dict:
    """Final node - just returns the state"""
    print("\n✅ RESEARCH COMPLETE")
    return {}


# ============================================
# Build the Graph
# ============================================

def build_deep_research_graph():
    """Build the complete deep research graph"""
    
    graph_builder = StateGraph(DeepResearchState)
    
    # Add all nodes
    graph_builder.add_node("supervisor", supervisor_node)
    graph_builder.add_node("planner", planner_agent)
    graph_builder.add_node("researcher", researcher_agent)
    graph_builder.add_node("analyst", analyst_agent)
    graph_builder.add_node("writer", writer_agent)
    graph_builder.add_node("finish", finish_node)
    
    # Start with supervisor
    graph_builder.add_edge(START, "supervisor")
    
    # Supervisor routes to workers
    graph_builder.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "planner": "planner",
            "researcher": "researcher",
            "analyst": "analyst",
            "writer": "writer",
            "finish": "finish"
        }
    )
    
    # All workers return to supervisor
    graph_builder.add_edge("planner", "supervisor")
    graph_builder.add_edge("researcher", "supervisor")
    graph_builder.add_edge("analyst", "supervisor")
    graph_builder.add_edge("writer", "supervisor")
    
    # Finish goes to END
    graph_builder.add_edge("finish", END)
    
    return graph_builder.compile()


# ============================================
# Main Function
# ============================================

def deep_research(query: str, verbose: bool = True) -> dict:
    """
    Run deep research on a query
    
    Args:
        query: The research question
        verbose: Whether to print progress
    
    Returns:
        The final state with the research report
    """
    if verbose:
        print("\n" + "="*80)
        print("🚀 DEEP RESEARCH SYSTEM")
        print("="*80)
        print(f"\n📌 Research Query: {query}\n")
    
    # Build graph
    graph = build_deep_research_graph()
    
    # Initial state
    initial_state: DeepResearchState = {
        "original_query": query,
        "research_plan": {},
        "sub_questions": [],
        "current_question_index": 0,
        "research_data": [],
        "analysis": "",
        "final_report": "",
        "next_worker": "",
        "iteration": 0,
        "messages": [],
        "sources": []
    }
    
    # Run the graph
    result = graph.invoke(initial_state)
    
    return result


# ============================================
# Run It!
# ============================================

if __name__ == "__main__":
    # Research query
    query = """
    What are the most promising renewable energy technologies for 2024-2025, 
    and what challenges do they face for widespread adoption?
    """
    
    # Run deep research
    result = deep_research(query.strip())
    
    # Print the final report
    print("\n" + "="*80)
    print("📄 FINAL RESEARCH REPORT")
    print("="*80)
    print(result["final_report"])
    
    # Print summary stats
    print("\n" + "="*80)
    print("📊 RESEARCH STATISTICS")
    print("="*80)
    print(f"   Sub-questions researched: {len(result.get('research_data', []))}")
    print(f"   Total sources: {len(set(result.get('sources', [])))}")
    print(f"   Iterations: {result.get('iteration', 0)}")
    
    # Save to file
    with open("research_report.md", "w") as f:
        f.write(result["final_report"])
    print("\n   📁 Report saved to: research_report.md")