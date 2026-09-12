"""
ResearchMind - LangGraph orchestration with a real critic -> writer feedback loop.

Replaces the old linear pipeline.py. Flow:

    search -> read -> write -> critique --(score < threshold and retries left)--> write
                                    |
                                    --(score >= threshold OR out of retries)--> END

Run directly:  python graph_pipeline.py
Import into Streamlit: from graph_pipeline import run_research_pipeline
"""

import re
from typing import TypedDict, Annotated
from operator import add

from langgraph.graph import StateGraph, END

from agents import (
    build_search_agent,
    build_reader_agent,
    writer_chain,
    revision_chain,
    critic_chain,
)

MAX_REVISIONS = 2          # writer gets at most this many revision passes
PASS_THRESHOLD = 7         # critic score out of 10 needed to stop revising


# --------------------------------------------------
# STATE
# --------------------------------------------------

class ResearchState(TypedDict):
    topic: str
    search_results: str
    scraped_content: str
    report: str
    feedback: str
    score: int
    revision_count: int
    history: Annotated[list[str], add]   # log of what happened, step by step


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def _extract_score(critic_text: str) -> int:
    """Pull the numeric score out of the critic's 'Score: X/10' line. Defaults to 0 if not found."""
    match = re.search(r"Score:\s*(\d+)\s*/\s*10", critic_text)
    return int(match.group(1)) if match else 0


# --------------------------------------------------
# NODES
# --------------------------------------------------

def search_node(state: ResearchState) -> dict:
    agent = build_search_agent()
    result = agent.invoke({
        "messages": [("user", f"Find recent, reliable and detailed information about: {state['topic']}")]
    })
    content = result["messages"][-1].content
    return {
        "search_results": content,
        "history": [f"[search] gathered {len(content)} chars of search results"],
    }


def read_node(state: ResearchState) -> dict:
    agent = build_reader_agent()
    result = agent.invoke({
        "messages": [("user",
            f"Based on the following search results about '{state['topic']}', "
            f"pick the most relevant URL and scrape it for deeper content.\n\n"
            f"Search Results:\n{state['search_results'][:800]}"
        )]
    })
    content = result["messages"][-1].content
    return {
        "scraped_content": content,
        "history": [f"[read] scraped {len(content)} chars of source content"],
    }


def write_node(state: ResearchState) -> dict:
    # First pass: write from scratch. Later passes: revise using critic feedback.
    if state.get("report") and state.get("feedback"):
        report = revision_chain.invoke({
            "topic": state["topic"],
            "previous_report": state["report"],
            "feedback": state["feedback"],
        })
        log = f"[write] revised report (pass {state['revision_count'] + 1})"
    else:
        research_combined = (
            f"SEARCH RESULTS:\n{state['search_results']}\n\n"
            f"DETAILED SCRAPED CONTENT:\n{state['scraped_content']}"
        )
        report = writer_chain.invoke({
            "topic": state["topic"],
            "research": research_combined,
        })
        log = "[write] drafted initial report"

    return {"report": report, "history": [log]}


def critique_node(state: ResearchState) -> dict:
    feedback = critic_chain.invoke({"report": state["report"]})
    score = _extract_score(feedback)
    return {
        "feedback": feedback,
        "score": score,
        "revision_count": state["revision_count"] + 1,
        "history": [f"[critique] score={score}/10 (revision {state['revision_count'] + 1})"],
    }


# --------------------------------------------------
# CONDITIONAL EDGE
# --------------------------------------------------

def should_revise(state: ResearchState) -> str:
    if state["score"] >= PASS_THRESHOLD:
        return "end"
    if state["revision_count"] >= MAX_REVISIONS:
        return "end"
    return "revise"


# --------------------------------------------------
# GRAPH ASSEMBLY
# --------------------------------------------------

def build_graph():
    graph = StateGraph(ResearchState)

    graph.add_node("search", search_node)
    graph.add_node("read", read_node)
    graph.add_node("write", write_node)
    graph.add_node("critique", critique_node)

    graph.set_entry_point("search")
    graph.add_edge("search", "read")
    graph.add_edge("read", "write")
    graph.add_edge("write", "critique")

    graph.add_conditional_edges(
        "critique",
        should_revise,
        {
            "revise": "write",
            "end": END,
        },
    )

    return graph.compile()


research_graph = build_graph()


# --------------------------------------------------
# PUBLIC ENTRY POINT (used by app.py)
# --------------------------------------------------

def run_research_pipeline(topic: str) -> dict:
    initial_state: ResearchState = {
        "topic": topic,
        "search_results": "",
        "scraped_content": "",
        "report": "",
        "feedback": "",
        "score": 0,
        "revision_count": 0,
        "history": [],
    }

    final_state = research_graph.invoke(initial_state)

    for line in final_state["history"]:
        print(line)

    return final_state


if __name__ == "__main__":
    topic = input("\nEnter a research topic: ")
    result = run_research_pipeline(topic)

    print("\n" + "=" * 50)
    print(f"FINAL REPORT (score {result['score']}/10 after {result['revision_count']} pass(es))")
    print("=" * 50)
    print(result["report"])
    print("\n--- Final critic feedback ---")
    print(result["feedback"])