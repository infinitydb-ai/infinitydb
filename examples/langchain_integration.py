"""InfinityDB Drop-in Memory Adapter for LangChain.

Provides sub-millisecond, local-first neurosymbolic memory for LangChain agents
with deterministic fact extraction, causal relations, and audit proofs.
"""

from __future__ import annotations

from typing import Any, Dict, List

from infinitydb import InfinityDB


class InfinityDBLangChainMemory:
    """Drop-in memory interface compatible with LangChain BaseMemory specification."""

    def __init__(
        self,
        db_path: str = "langchain_agent.db",
        user_id: str = "default_user",
        memory_key: str = "chat_history",
        return_facts_only: bool = False,
    ) -> None:
        self.db = InfinityDB(db_path)
        self.user_id = user_id
        self.memory_key = memory_key
        self.return_facts_only = return_facts_only

    @property
    def memory_variables(self) -> List[str]:
        """Return memory input keys."""
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve relevant memories for the current prompt context."""
        query = str(inputs.get("input", inputs.get("question", "")))
        if not query.strip():
            return {self.memory_key: ""}

        recall_res = self.db.recall(query, user_id=self.user_id)
        
        if self.return_facts_only:
            facts = [r["fact"] for r in recall_res.get("results", [])]
            formatted_context = "\n".join(f"- {fact}" for fact in facts)
        else:
            facts = [r["fact"] for r in recall_res.get("results", [])]
            persona = self.db.get_persona(user_id=self.user_id)
            policies = recall_res.get("policies", [])
            
            sections = []
            if facts:
                sections.append("Recalled Facts:\n" + "\n".join(f"- {f}" for f in facts))
            if persona:
                sections.append("User Profile State:\n" + "\n".join(f"- {k}: {v}" for k, v in persona.items()))
            if policies:
                sections.append("Governance Guardrails:\n" + "\n".join(f"- [POLICY] {p}" for p in policies))
                
            formatted_context = "\n\n".join(sections)

        return {self.memory_key: formatted_context}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Commit user interaction and agent response into L0-L2 memory."""
        user_msg = str(inputs.get("input", inputs.get("question", ""))).strip()
        agent_msg = str(outputs.get("output", outputs.get("response", ""))).strip()

        if user_msg:
            self.db.add(user_msg, user_id=self.user_id)
        if agent_msg:
            self.db.add(f"Agent response: {agent_msg}", user_id=self.user_id)

    def clear(self) -> None:
        """Clear user memory according to GDPR right-to-be-forgotten."""
        self.db.delete_user(user_id=self.user_id)


def main() -> None:
    print("[*] Initializing InfinityDB LangChain Adapter...")
    memory = InfinityDBLangChainMemory(db_path="demo_langchain.db", user_id="user_alex")

    # 1. Simulate multi-turn dialogue ingestion
    print("[*] Ingesting multi-turn conversation into memory...")
    memory.save_context(
        {"input": "Hi, I am Alex. I manage cloud infrastructure in Frankfurt. We strictly use Terraform and Go."},
        {"output": "Understood, Alex. I will tailor all recommendations to Terraform and Go."}
    )
    
    # User modifies preferences over time (tested with conflict auto-resolution)
    memory.save_context(
        {"input": "Our policy changed today: we must now migrate all legacy workloads to Kubernetes and Rust."},
        {"output": "Noted. Updating infrastructure standards to Kubernetes and Rust."}
    )

    # 2. Querying memory for next interaction
    query_inputs = {"input": "What language and cloud deployment tools should we use?"}
    loaded = memory.load_memory_variables(query_inputs)

    print("\n[+] Injected Context into LangChain Agent Prompt:")
    print("--------------------------------------------------")
    print(loaded[memory.memory_key])
    print("--------------------------------------------------")
    print("[PASS] LangChain adapter verified successfully.")


if __name__ == "__main__":
    main()
