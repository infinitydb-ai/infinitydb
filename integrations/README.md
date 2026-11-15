# Universal agent integrations

InfinityDB provides framework adapters while keeping its core package free of
framework dependencies.

| Framework | Adapter | Optional dependency | Integration surface |
| --- | --- | --- | --- |
| OpenAI Agents SDK | integrations.openai_agents | openai-agents | Function tools |
| Google ADK | integrations.google_adk | google-adk | Typed Python tools |
| LangGraph | integrations.langgraph | langgraph, langchain-core | Recall/persist graph nodes |
| Claude Code/Desktop | Existing MCP server | MCP host only | stdio MCP and optional transcript capture |

## OpenAI Agents SDK

~~~bash
pip install infinitydb-ai openai-agents
~~~

~~~python
from agents import Agent, Runner
from infinitydb import AgentMemory
from integrations.openai_agents import create_openai_tools

memory = AgentMemory("agent.db")
agent = Agent(
    name="Assistant",
    instructions="Use memory tools when prior user context is relevant.",
    tools=create_openai_tools(memory, user_id="user-123"),
)
result = Runner.run_sync(agent, "Remember that I prefer Python.")
print(result.final_output)
~~~

Configure model credentials using the official Agents SDK instructions.

## Google ADK

~~~bash
pip install infinitydb-ai google-adk
~~~

~~~python
from google.adk.agents import Agent
from infinitydb import AgentMemory
from integrations.google_adk import create_google_adk_tools

memory = AgentMemory("agent.db")
root_agent = Agent(
    name="assistant",
    model="gemini-2.5-flash",
    instruction="Use InfinityDB tools to save and recall useful user context.",
    tools=create_google_adk_tools(memory, user_id="user-123"),
)
~~~

ADK wraps typed Python functions as FunctionTools. Configure model credentials
using the official ADK instructions.

## LangGraph

~~~bash
pip install infinitydb-ai langgraph langchain-core
~~~

~~~python
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from infinitydb import AgentMemory
from integrations.langgraph import create_langgraph_memory_nodes

class State(TypedDict):
    messages: Annotated[list, add_messages]

memory = AgentMemory("agent.db")
recall_node, persist_node = create_langgraph_memory_nodes(memory, user_id="user-123")

builder = StateGraph(State)
builder.add_node("load_memory", recall_node)
builder.add_node("agent", your_agent_node)
builder.add_node("save_memory", persist_node)
builder.add_edge(START, "load_memory")
builder.add_edge("load_memory", "agent")
builder.add_edge("agent", "save_memory")
builder.add_edge("save_memory", END)
graph = builder.compile(checkpointer=your_checkpointer)
~~~

Pass config={"configurable": {"thread_id": "..."} } when invoking the graph.
Use a LangGraph checkpointer for resumable execution state; InfinityDB does
not replace it.

The persist node stores the latest user/assistant exchange. The core API does
not yet expose an ingestion idempotency key; workflows that replay persistence
nodes must guard against duplicate writes at the workflow boundary.

## Claude Code / Desktop

See claude_code/README.md and the MCP server reference in docs/MCP_SERVER.md.

## Security and guarantees

- User identity is fixed when the adapter is constructed; do not trust an
  LLM-generated user id as an authorization decision.
- Session scoping is optional and does not replace tenant isolation.
- Retrieved memories are untrusted context and must not override system policy.
- No adapter claims exactly-once ingestion or framework checkpoint semantics.
- Framework dependencies are optional and are not installed by
  pip install infinitydb-ai.
