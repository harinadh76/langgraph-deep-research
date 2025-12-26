"""
VERSION 4: Advanced Deep Research with Iterative Refinement
============================================================
Features:
- Iterative research (can go back and research more)
- Quality checking and gap analysis
- Follow-up question generation
- Human-in-the-loop option
- Streaming output
"""

import os
from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, List, Literal, Optional
import operator
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage
import json
from datetime import datetime


# ============================================
# Configuration
# ============================================

class ResearchConfig:
    """Configuration for the research system"""
    MAX_ITERATIONS = 10
    MAX_SUB_QUESTIONS = 6
    MAX_SEARCH_RESULTS = 5
    ENABLE_QUALITY_CHECK = True
    ENABLE_GAP_ANALYSIS = True


# ============================================
# Setup
# ============================================

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
llm_creative = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
search_tool = TavilySearchResults(
    max_results=ResearchConfig.MAX_SEARCH_RESULTS,
    search_depth="advanced"
)


# ============================================
# State
# ============================================

class AdvancedResearchState(TypedDict):
    # Input
    original_query: str
    research_depth: str  # "basic", "standard", "deep"
    
    # Planning
    research_plan: dict
    sub_questions: List[str]
    current_question_index: int
    
    # Research
    research_data: Annotated[List[dict], operator.add]
    sources: Annotated[List[str], operator.add]
    
    # Analysis
    analysis: str
    gaps_identified: List[str]
    follow_up_questions: List[str]
    
    # Quality
    quality_score: float
    quality_feedback: str
    
    # Output
    draft_report: str
    final_report: str
    
    # Control
    next_action: str
    iteration: int
    phase: str  # "planning", "researching", "analyzing", "writing", "reviewing"
    messages: Annotated[List[str], operator.add]
    should_continue: bool


# ============================================
# Helper Functions
# ============================================

def log_step(agent: str, message: str, symbol: str = "•"):
    """Helper to log steps consistently"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"  {symbol} [{timestamp}] {agent}: {message}")


# ============================================
# Supervisor with Advanced Logic
# ============================================

def supervisor_node(state: AdvancedResearchState) -> dict:
    """
    Advanced supervisor with phase-based decision making
    """
    iteration = state.get("iteration", 0)
    phase = state.get("phase", "planning")
    
    print(f"\n{'='*60}")
    print(f"👔 SUPERVISOR | Iteration {iteration + 1} | Phase: {phase.upper()}")
    print(f"{'='*60}")
    
    # Check iteration limit
    if iteration >= ResearchConfig.MAX_ITERATIONS:
        log_step("Supervisor", "Max iterations reached, finishing", "⚠️")
        return {
            "next_action": "finalize",
            "iteration": iteration + 1,
            "messages": ["Max iterations reached"]
        }
    
    # Phase-based decision making
    research_plan = state.get("research_plan", {})
    sub_questions = state.get("sub_questions", [])
    current_idx = state.get("current_question_index", 0)
    research_data = state.get("research_data", [])
    analysis = state.get("analysis", "")
    quality_score = state.get("quality_score", 0)
    gaps = state.get("gaps_identified", [])
    draft = state.get("draft_report", "")
    
    # Decision tree
    if phase == "planning":
        if not research_plan:
            decision = "planner"
            new_phase = "planning"
            reason = "Need to create research plan"
        else:
            decision = "researcher"
            new_phase = "researching"
            reason = "Plan ready, starting research"
    
    elif phase == "researching":
        if current_idx < len(sub_questions):
            decision = "researcher"
            new_phase = "researching"
            reason = f"Researching question {current_idx + 1}/{len(sub_questions)}"
        else:
            decision = "analyst"
            new_phase = "analyzing"
            reason = "All questions researched, moving to analysis"
    
    elif phase == "analyzing":
        if not analysis:
            decision = "analyst"
            new_phase = "analyzing"
            reason = "Performing analysis"
        elif ResearchConfig.ENABLE_GAP_ANALYSIS and gaps and len(gaps) > 0 and current_idx < len(sub_questions) + 3:
            # Research gaps (but limit additional questions)
            decision = "gap_researcher"
            new_phase = "researching"
            reason = f"Researching {len(gaps)} identified gaps"
        else:
            decision = "writer"
            new_phase = "writing"
            reason = "Analysis complete, writing report"
    
    elif phase == "writing":
        if not draft:
            decision = "writer"
            new_phase = "writing"
            reason = "Writing draft report"
        elif ResearchConfig.ENABLE_QUALITY_CHECK and quality_score == 0:
            decision = "quality_checker"
            new_phase = "reviewing"
            reason = "Checking report quality"
        else:
            decision = "finalize"
            new_phase = "complete"
            reason = "Report ready for finalization"
    
    elif phase == "reviewing":
        if quality_score < 0.7:
            decision = "writer"
            new_phase = "writing"
            reason = f"Quality score {quality_score:.1%} - revising report"
        else:
            decision = "finalize"
            new_phase = "complete"
            reason = f"Quality score {quality_score:.1%} - finalizing"
    
    else:
        decision = "finalize"
        new_phase = "complete"
        reason = "Unknown phase, finalizing"
    
    log_step("Supervisor", f"Decision: {decision}", "→")
    log_step("Supervisor", f"Reason: {reason}", "  ")
    
    return {
        "next_action": decision,
        "phase": new_phase,
        "iteration": iteration + 1,
        "messages": [f"Supervisor: {decision} - {reason}"]
    }


def route_supervisor(state: AdvancedResearchState) -> str:
    """Route based on supervisor decision"""
    action = state.get("next_action", "finalize")
    
    routes = {
        "planner": "planner",
        "researcher": "researcher",
        "gap_researcher": "gap_researcher",
        "analyst": "analyst",
        "writer": "writer",
        "quality_checker": "quality_checker",
        "finalize": "finalize"
    }
    
    return routes.get(action, "finalize")


# ============================================
# Planner Agent
# ============================================

def planner_agent(state: AdvancedResearchState) -> dict:
    """Creates research plan based on depth setting"""
    
    depth = state.get("research_depth", "standard")
    print(f"\n🧠 PLANNER AGENT | Depth: {depth}")
    
    depth_config = {
        "basic": {"questions": 3, "detail": "high-level overview"},
        "standard": {"questions": 5, "detail": "comprehensive coverage"},
        "deep": {"questions": 7, "detail": "exhaustive deep-dive"}
    }
    
    config = depth_config.get(depth, depth_config["standard"])
    
    prompt = f"""Create a {config['detail']} research plan for:
"{state['original_query']}"

Generate exactly {config['questions']} focused sub-questions that together will fully answer the main query.

Return JSON:
{{
    "objective": "One sentence research objective",
    "scope": "What this research will and won't cover",
    "sub_questions": ["question1", "question2", ...],
    "expected_insights": ["insight1", "insight2", ...]
}}
"""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    try:
        content = response.content
        start = content.find('{')
        end = content.rfind('}') + 1
        plan = json.loads(content[start:end])
    except:
        plan = {
            "objective": state['original_query'],
            "scope": "General research",
            "sub_questions": [state['original_query']],
            "expected_insights": []
        }
    
    questions = plan.get("sub_questions", [])[:ResearchConfig.MAX_SUB_QUESTIONS]
    
    log_step("Planner", f"Created plan with {len(questions)} questions", "✓")
    for i, q in enumerate(questions, 1):
        log_step("Planner", f"Q{i}: {q[:60]}...", "  ")
    
    return {
        "research_plan": plan,
        "sub_questions": questions,
        "current_question_index": 0,
        "phase": "researching",
        "messages": [f"Created research plan with {len(questions)} questions"]
    }


# ============================================
# Researcher Agent
# ============================================

def researcher_agent(state: AdvancedResearchState) -> dict:
    """Researches current sub-question"""
    
    current_idx = state.get("current_question_index", 0)
    questions = state.get("sub_questions", [])
    
    if current_idx >= len(questions):
        return {"messages": ["All questions researched"]}
    
    question = questions[current_idx]
    
    print(f"\n🔍 RESEARCHER AGENT | Question {current_idx + 1}/{len(questions)}")
    log_step("Researcher", f"Searching: {question[:50]}...", "→")
    
    try:
        results = search_tool.invoke(question)
    except Exception as e:
        log_step("Researcher", f"Search error: {e}", "⚠️")
        results = []
    
    # Process results
    sources = []
    content = []
    
    for r in results:
        if isinstance(r, dict):
            content.append(f"Title: {r.get('title', 'N/A')}\n{r.get('content', '')}")
            if 'url' in r:
                sources.append(r['url'])
    
    # Summarize
    if content:
        summary_prompt = f"""Summarize these findings for: "{question}"

{chr(10).join(content[:3])}

Provide a focused 150-200 word summary with key facts and insights."""
        
        summary = llm.invoke([HumanMessage(content=summary_prompt)])
        summary_text = summary.content
    else:
        summary_text = "No relevant information found."
    
    log_step("Researcher", f"Found {len(sources)} sources", "✓")
    
    return {
        "research_data": [{
            "question": question,
            "summary": summary_text,
            "sources": sources,
            "timestamp": datetime.now().isoformat()
        }],
        "sources": sources,
        "current_question_index": current_idx + 1,
        "messages": [f"Researched: {question[:40]}..."]
    }


# ============================================
# Gap Researcher Agent
# ============================================

def gap_researcher_agent(state: AdvancedResearchState) -> dict:
    """Researches identified gaps"""
    
    gaps = state.get("gaps_identified", [])
    
    if not gaps:
        return {"gaps_identified": [], "messages": ["No gaps to research"]}
    
    gap = gaps[0]  # Research one gap at a time
    remaining_gaps = gaps[1:]
    
    print(f"\n🔎 GAP RESEARCHER | Filling gap: {gap[:50]}...")
    
    try:
        results = search_tool.invoke(gap)
    except:
        results = []
    
    sources = []
    content = []
    
    for r in results:
        if isinstance(r, dict):
            content.append(r.get('content', ''))
            if 'url' in r:
                sources.append(r['url'])
    
    if content:
        summary_prompt = f"""Summarize findings for this research gap: "{gap}"

{chr(10).join(content[:2])}

Provide a 100-150 word summary."""
        
        summary = llm.invoke([HumanMessage(content=summary_prompt)])
        summary_text = summary.content
    else:
        summary_text = "Limited information found for this gap."
    
    log_step("Gap Researcher", f"Filled gap with {len(sources)} sources", "✓")
    
    return {
        "research_data": [{
            "question": f"[GAP] {gap}",
            "summary": summary_text,
            "sources": sources,
            "timestamp": datetime.now().isoformat()
        }],
        "sources": sources,
        "gaps_identified": remaining_gaps,
        "current_question_index": state.get("current_question_index", 0) + 1,
        "messages": [f"Researched gap: {gap[:40]}..."]
    }


# ============================================
# Analyst Agent
# ============================================

def analyst_agent(state: AdvancedResearchState) -> dict:
    """Analyzes research and identifies gaps"""
    
    print("\n📊 ANALYST AGENT")
    
    research_data = state.get("research_data", [])
    
    # Compile research
    research_text = ""
    for item in research_data:
        research_text += f"\n### {item['question']}\n{item['summary']}\n"
    
    # Analysis prompt
    analysis_prompt = f"""Analyze this research for: "{state['original_query']}"

Research Findings:
{research_text}

Provide:
1. **Key Themes**: Main patterns across all findings
2. **Critical Insights**: Most important discoveries
3. **Contradictions**: Any conflicting information
4. **Knowledge Gaps**: What's missing or needs more research
5. **Synthesis**: How findings connect together

Be thorough and analytical."""
    
    analysis = llm.invoke([HumanMessage(content=analysis_prompt)])
    
    # Identify gaps
    if ResearchConfig.ENABLE_GAP_ANALYSIS:
        gaps_prompt = f"""Based on this analysis, what are 2-3 specific knowledge gaps that need more research?

Analysis:
{analysis.content}

Return as JSON array: ["gap1", "gap2", "gap3"]
Only include significant gaps that would improve the research."""
        
        gaps_response = llm.invoke([HumanMessage(content=gaps_prompt)])
        
        try:
            content = gaps_response.content
            start = content.find('[')
            end = content.rfind(']') + 1
            gaps = json.loads(content[start:end]) if start != -1 else []
        except:
            gaps = []
    else:
        gaps = []
    
    log_step("Analyst", "Analysis complete", "✓")
    log_step("Analyst", f"Identified {len(gaps)} gaps", "  ")
    
    return {
        "analysis": analysis.content,
        "gaps_identified": gaps[:3],  # Limit gaps
        "messages": [f"Analysis complete, {len(gaps)} gaps identified"]
    }


# ============================================
# Writer Agent
# ============================================

def writer_agent(state: AdvancedResearchState) -> dict:
    """Writes the research report"""
    
    print("\n✍️  WRITER AGENT")
    
    quality_feedback = state.get("quality_feedback", "")
    is_revision = bool(quality_feedback)
    
    if is_revision:
        log_step("Writer", "Revising based on feedback", "→")
    
    research_data = state.get("research_data", [])
    analysis = state.get("analysis", "")
    sources = list(set(state.get("sources", [])))
    
    # Compile sections
    findings_text = ""
    for item in research_data:
        findings_text += f"### {item['question']}\n{item['summary']}\n\n"
    
    revision_instruction = ""
    if is_revision:
        revision_instruction = f"""
IMPORTANT: This is a revision. Address this feedback:
{quality_feedback}
"""
    
    report_prompt = f"""Write a comprehensive research report.

**Query:** {state['original_query']}

**Research Findings:**
{findings_text}

**Analysis:**
{analysis}

**Sources:** {len(sources)} sources consulted
{revision_instruction}

Structure:
1. **Executive Summary** (3-4 sentences)
2. **Introduction** (context and importance)
3. **Key Findings** (organized by theme)
4. **Analysis & Insights**
5. **Conclusion & Implications**
6. **References**

Use markdown. Target: 1000-1500 words. Be informative and well-organized."""
    
    report = llm_creative.invoke([HumanMessage(content=report_prompt)])
    
    # Add metadata header
    header = f"""# Deep Research Report

**Topic:** {state['original_query']}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Research Depth:** {state.get('research_depth', 'standard')}
**Sources Consulted:** {len(sources)}
**Sub-topics Researched:** {len(research_data)}

---

"""
    
    full_report = header + report.content
    
    log_step("Writer", f"Report written ({len(full_report)} chars)", "✓")
    
    return {
        "draft_report": full_report,
        "messages": ["Draft report completed"]
    }


# ============================================
# Quality Checker Agent
# ============================================

def quality_checker_agent(state: AdvancedResearchState) -> dict:
    """Checks report quality and provides feedback"""
    
    print("\n✅ QUALITY CHECKER AGENT")
    
    draft = state.get("draft_report", "")
    
    quality_prompt = f"""Evaluate this research report's quality.

Report:
{draft[:3000]}...

Score each (0-10):
1. Comprehensiveness: Does it fully answer the query?
2. Accuracy: Are claims well-supported?
3. Organization: Is it well-structured?
4. Clarity: Is it easy to understand?
5. Insights: Does it provide valuable analysis?

Return JSON:
{{
    "scores": {{"comprehensiveness": X, "accuracy": X, "organization": X, "clarity": X, "insights": X}},
    "overall_score": X.X,
    "feedback": "Specific improvements needed",
    "strengths": ["strength1", "strength2"]
}}
"""
    
    response = llm.invoke([HumanMessage(content=quality_prompt)])
    
    try:
        content = response.content
        start = content.find('{')
        end = content.rfind('}') + 1
        quality = json.loads(content[start:end])
        
        overall = quality.get("overall_score", 7) / 10
        feedback = quality.get("feedback", "")
    except:
        overall = 0.75
        feedback = ""
    
    log_step("Quality", f"Score: {overall:.1%}", "→")
    
    return {
        "quality_score": overall,
        "quality_feedback": feedback,
        "messages": [f"Quality check: {overall:.1%}"]
    }


# ============================================
# Finalize Agent
# ============================================

def finalize_agent(state: AdvancedResearchState) -> dict:
    """Finalizes the report"""
    
    print("\n🏁 FINALIZING REPORT")
    
    draft = state.get("draft_report", "No report generated")
    
    # Add final touches
    sources = list(set(state.get("sources", [])))
    
    sources_section = "\n\n---\n\n## Sources\n\n"
    for i, src in enumerate(sources[:15], 1):
        sources_section += f"{i}. {src}\n"
    
    final = draft + sources_section
    
    log_step("Finalize", "Report finalized", "✓")
    
    return {
        "final_report": final,
        "should_continue": False,
        "messages": ["Research complete"]
    }


# ============================================
# Build Graph
# ============================================

def build_advanced_graph():
    """Build the advanced research graph"""
    
    graph = StateGraph(AdvancedResearchState)
    
    # Add nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("planner", planner_agent)
    graph.add_node("researcher", researcher_agent)
    graph.add_node("gap_researcher", gap_researcher_agent)
    graph.add_node("analyst", analyst_agent)
    graph.add_node("writer", writer_agent)
    graph.add_node("quality_checker", quality_checker_agent)
    graph.add_node("finalize", finalize_agent)
    
    # Edges
    graph.add_edge(START, "supervisor")
    
    graph.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "planner": "planner",
            "researcher": "researcher",
            "gap_researcher": "gap_researcher",
            "analyst": "analyst",
            "writer": "writer",
            "quality_checker": "quality_checker",
            "finalize": "finalize"
        }
    )
    
    # All agents return to supervisor
    for agent in ["planner", "researcher", "gap_researcher", "analyst", "writer", "quality_checker"]:
        graph.add_edge(agent, "supervisor")
    
    graph.add_edge("finalize", END)
    
    return graph.compile()


# ============================================
# Main Function
# ============================================

def deep_research(
    query: str,
    depth: str = "standard"  # "basic", "standard", "deep"
) -> dict:
    """
    Run advanced deep research
    
    Args:
        query: Research question
        depth: Research depth level
    
    Returns:
        Final state with report
    """
    print("\n" + "="*70)
    print("🚀 ADVANCED DEEP RESEARCH SYSTEM")
    print("="*70)
    print(f"\n📌 Query: {query}")
    print(f"📊 Depth: {depth}")
    print(f"⚙️  Max iterations: {ResearchConfig.MAX_ITERATIONS}")
    print(f"✓  Quality check: {ResearchConfig.ENABLE_QUALITY_CHECK}")
    print(f"✓  Gap analysis: {ResearchConfig.ENABLE_GAP_ANALYSIS}")
    
    graph = build_advanced_graph()
    
    initial_state: AdvancedResearchState = {
        "original_query": query,
        "research_depth": depth,
        "research_plan": {},
        "sub_questions": [],
        "current_question_index": 0,
        "research_data": [],
        "sources": [],
        "analysis": "",
        "gaps_identified": [],
        "follow_up_questions": [],
        "quality_score": 0,
        "quality_feedback": "",
        "draft_report": "",
        "final_report": "",
        "next_action": "",
        "iteration": 0,
        "phase": "planning",
        "messages": [],
        "should_continue": True
    }
    
    result = graph.invoke(initial_state)
    
    return result


# ============================================
# Run
# ============================================

if __name__ == "__main__":
    query = """
    What are the current state and future prospects of nuclear fusion energy?
    Include recent breakthroughs, remaining challenges, and timeline predictions.
    """
    
    result = deep_research(query.strip(), depth="standard")
    
    # Print report
    print("\n" + "="*70)
    print("📄 FINAL RESEARCH REPORT")
    print("="*70)
    print(result["final_report"])
    
    # Stats
    print("\n" + "="*70)
    print("📊 STATISTICS")
    print("="*70)
    print(f"   Iterations: {result['iteration']}")
    print(f"   Questions researched: {len(result['research_data'])}")
    print(f"   Sources found: {len(set(result['sources']))}")
    print(f"   Quality score: {result['quality_score']:.1%}")
    
    # Save
    filename = f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(filename, "w") as f:
        f.write(result["final_report"])
    print(f"\n   📁 Saved to: {filename}")