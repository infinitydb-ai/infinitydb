# Universal agent integrations

This coordinated integration release adds adapter surfaces for OpenAI Agents
SDK, Google ADK, LangGraph, and Claude Code/Desktop while keeping the core
package free of framework dependencies.

## Compatibility boundaries

| Framework | Integration model | Framework-owned capabilities |
| --- | --- | --- |
| OpenAI Agents SDK | Two typed FunctionTools for save and recall | Agent loop, handoffs, tracing, approvals |
| Google ADK | Two typed Python functions auto-wrapped as FunctionTools | Session state, invocation lifecycle, runner |
| LangGraph | Recall and persist graph nodes | Checkpointing, graph state, reducers, retries |
| Claude Code/Desktop | Existing stdio MCP server and optional transcript hook | MCP host lifecycle, permissions, UI |

InfinityDB does not replace framework checkpointers, hosted conversation
state, or execution recovery.

## Install

Core-only installation remains unchanged:

~~~bash
pip install infinitydb-ai
~~~

Install only the optional dependencies needed by your application:

~~~bash
pip install "openai-agents" "google-adk" "langgraph" "langchain-core"
~~~

Claude connects to the existing MCP server and needs no additional Python
framework package.

## Shared API contract

Adapters accept either AgentMemory or the lower-level InfinityDB engine.
They scope operations with an application-supplied user_id and optional
session_id. User identity must come from trusted application authentication,
not an LLM-generated tool argument.

The adapters use the engine's add(conversation=...) and
recall(query=..., n=..., reinforce=False) APIs. Tool results are reduced to
JSON-safe ids, facts, timestamps, scores, and policy rules.

## Framework behavior

### OpenAI Agents SDK

integrations.openai_agents.create_openai_tools(...) returns two function_tool
instances for Agent(tools=[...]). The optional SDK import is deferred until
the factory is called.

### Google ADK

integrations.google_adk.create_google_adk_tools(...) returns typed Python
functions. ADK inspects their signatures and docstrings and wraps them as
FunctionTools when supplied in an agent's tools list.

### LangGraph

integrations.langgraph.create_langgraph_memory_nodes(...) returns
(recall_node, persist_node). Place recall before the model node and persist
after it. Supply invocation config to scope by
configurable.thread_id. Continue to configure a LangGraph checkpointer for
short-term execution state.

The persist node writes the latest user/assistant exchange. The current core
API does not expose an ingestion idempotency key; graphs that replay
persistence nodes must guard against duplicate writes at the workflow boundary.

### Claude Code and Claude Desktop

Use python -m infinitydb.mcp_server as a stdio MCP server. For opt-in
transcript capture, use python -m infinitydb.capture --transcript ... from a
Claude Code Stop hook. See integrations/claude_code/README.md.

## Tests

~~~bash
pytest -q tests/test_framework_integrations.py
pytest
~~~

The adapter contract tests use a fake engine and do not require network access,
model credentials, or optional framework installations. For full compatibility
certification, run examples against the exact framework versions in your
application lockfile and record those versions.

## Production checklist

- Derive user_id from authenticated application identity.
- Enforce and test tenant isolation.
- Treat retrieved memories as untrusted data, not instructions.
- Obtain consent before capturing transcripts.
- Test deletion and backup/replica retention against your policy.
- Use framework checkpointers for execution recovery.
- Pin framework versions and test upgrades.
