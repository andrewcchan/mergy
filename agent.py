import os
import json
from typing import List, Dict, Any, Optional, Annotated, Literal, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

import operator

# Ensure API Key is available
if not os.environ.get("OPENAI_API_KEY"):
    pass # In actual run, need to set this. For now just placeholder.

class ResearchVector(BaseModel):
    id: str = Field(description="A unique identifier for the research vector.")
    title: str = Field(description="The title of the research vector (e.g. Financial Health).")
    description: str = Field(description="A brief description of what needs to be researched.")
    queries: List[str] = Field(description="A list of specific questions to answer.")

class Plan(BaseModel):
    vectors: List[ResearchVector] = Field(description="A list of distinct research vectors to execute.")

class Finding(BaseModel):
    vector_id: str = Field(description="The ID of the research vector this finding corresponds to.")
    markdown_content: str = Field(description="The comprehensive finding content in Markdown format, with citations.")

# Sub-Agent State (used within the execution phase for a specific vector)
class SubAgentState(TypedDict):
    company: str
    vector: ResearchVector
    messages: Annotated[list, add_messages]
    notes: List[str]
    draft_finding: Optional[str]

# Global State
class GlobalState(TypedDict):
    company: str
    plan: Optional[Plan]
    findings: Annotated[List[Finding], operator.add]
    report: Optional[str]

# --- Tools ---
from langchain_core.tools import tool
from duckduckgo_search import DDGS
import requests
from bs4 import BeautifulSoup
import markdownify

@tool
def search_tool(query: str) -> str:
    """Searches the web for the given query and returns snippets and links."""
    try:
        results = DDGS().text(query, max_results=5)
        if not results:
            return "No results found."
        formatted_results = []
        for r in results:
            formatted_results.append(f"Title: {r['title']}\nURL: {r['href']}\nSnippet: {r['body']}\n")
        return "\n".join(formatted_results)
    except Exception as e:
        return f"Search failed: {e}"

@tool
def scrape_tool(url: str) -> str:
    """Fetches the webpage at the given URL, strips out boilerplate, and returns clean Markdown."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # Remove nav, footer, scripts, styles
        for element in soup(["nav", "footer", "script", "style", "header", "aside"]):
            element.extract()

        markdown_text = markdownify.markdownify(str(soup), heading_style="ATX")
        # limit length to avoid context overflow
        return markdown_text[:10000]
    except Exception as e:
        return f"Scraping failed: {e}"

@tool
def notebook_tool(note: str) -> str:
    """Jot down facts as you find them."""
    return f"Note saved: {note}"

# --- Nodes ---
from langchain_openai import ChatOpenAI

def planner_node(state: GlobalState):
    """Phase 1: Planning. Creates a structured research plan based on the company."""
    company = state["company"]
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(Plan)

    prompt = f"""You are the Planner Agent. Your goal is to analyze the target company ({company})
and create a strategic research plan consisting of a list of distinct research vectors.
Typically, for a company, this includes:
- Financial health and funding history.
- Core products and technological stack.
- Leadership team and organizational structure.
- Competitors and market position.
- Recent news and public sentiment.

Generate the plan as a structured list of these vectors."""

    plan = structured_llm.invoke(prompt)
    return {"plan": plan}

def orchestrator_node(state: GlobalState):
    """Maps research vectors from the plan to sub-agent executions."""
    plan = state["plan"]
    company = state["company"]
    sends = []
    if plan and plan.vectors:
        for vector in plan.vectors:
            initial_message = SystemMessage(
                content=f"""You are a specialized Sub-Agent. Your objective is to map out {company}'s {vector.title}.
Description: {vector.description}
Queries to answer: {', '.join(vector.queries)}
Use the Search Tool to find official and credible sources.
Use the Scrape Tool to read pages and extract facts.
Use the Notebook Tool to save facts as you go.
When you are done, output your final comprehensive draft finding."""
            )
            sends.append(Send("parallel_execution_node", {
                "company": company,
                "vector": vector,
                "messages": [initial_message],
                "notes": [],
                "draft_finding": None
            }))
    return sends

from deepagents import create_deep_agent

# Refactored parallel node that wraps both to return to GlobalState
def parallel_execution_node(state: SubAgentState) -> Dict[str, Any]:
    # We use deep agents for the execution inner loop
    tools = [search_tool, scrape_tool, notebook_tool]

    # create_deep_agent expects a model string like provider:model
    agent = create_deep_agent(
        model="openai:gpt-4o-mini",
        tools=tools,
        system_prompt="Use tools to gather facts and answer queries."
    )

    # Run the deep agent
    response = agent.invoke({"messages": state["messages"]})
    draft = response["messages"][-1].content

    # Critic check (lightweight, single pass for simplicity)
    critic_prompt = f"""You are the Credibility Critic.
Review the following draft finding for the vector '{state['vector'].title}' about '{state['company']}'.
Draft Finding:
{draft}

Ensure it looks like a finding and has some references. If it lacks citations, add a note saying [Needs better citations].
Output the finalized markdown content."""

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    critic_response = llm.invoke([SystemMessage(content=critic_prompt)])
    final_content = critic_response.content

    finding = Finding(vector_id=state["vector"].id, markdown_content=final_content)

    return {"findings": [finding]}

def synthesizer_node(state: GlobalState):
    """Phase 3: Synthesis. Aggregates findings and generates a final report."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    company = state["company"]
    findings = state.get("findings", [])

    findings_text = "\n\n".join([f"--- Vector: {f.vector_id} ---\n{f.markdown_content}" for f in findings])

    prompt = f"""You are the Synthesizer Writer Agent.
Your objective is to write a comprehensive, highly structured, cited report about '{company}' based ONLY on the following findings.
Resolve any conflicting data and ensure a narrative flow.
Ensure strict adherence to inline citations linking back to the crawled URLs mentioned in the findings.

Findings:
{findings_text}

Output the final report in Markdown format."""

    response = llm.invoke([SystemMessage(content=prompt)])

    return {"report": response.content}

# --- Graph Wiring ---
builder = StateGraph(GlobalState)

builder.add_node("planner_node", planner_node)
builder.add_node("parallel_execution_node", parallel_execution_node)
builder.add_node("synthesizer_node", synthesizer_node)

builder.add_edge(START, "planner_node")
# Orchestrator uses `Send` to route to parallel execution nodes
builder.add_conditional_edges("planner_node", orchestrator_node)
# All parallel branches converge back to synthesizer
builder.add_edge("parallel_execution_node", "synthesizer_node")
builder.add_edge("synthesizer_node", END)

graph = builder.compile()
