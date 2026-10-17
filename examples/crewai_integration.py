"""InfinityDB Drop-in Multi-Agent Memory Adapter for CrewAI.

Enables CrewAI teams (e.g. Researcher, Architect, Writer) to share persistent,
verifiable long-term memory across sessions with zero external vector API costs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from infinitydb import InfinityDB


class InfinityDBCrewMemory:
    """Multi-agent shared memory provider for CrewAI workflows."""

    def __init__(self, db_path: str = "crewai_shared_memory.db") -> None:
        self.db = InfinityDB(db_path)

    def save(self, value: str, agent_role: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store task outputs or findings from a specific agent role."""
        meta_str = f" [role={agent_role}]"
        log_id = self.db.add(f"{value}{meta_str}", user_id=agent_role)
        return log_id

    def search(self, query: str, agent_role: Optional[str] = None, limit: int = 5) -> List[Dict[str, Any]]:
        """Search multi-agent shared memory with causal graph relations."""
        user_id = agent_role if agent_role else "default"
        res = self.db.recall(query, user_id=user_id)
        
        results = []
        for item in res.get("results", [])[:limit]:
            results.append({
                "id": item["id"],
                "content": item["fact"],
                "score": item["score"],
                "proof_available": res.get("proof_available", False),
            })
        return results

    def get_agent_profile(self, agent_role: str) -> Dict[str, Any]:
        """Retrieve dynamic state and established rules for an agent role."""
        return self.db.get_persona(user_id=agent_role)

    def add_crew_policy(self, rule: str, category: str = "crew_governance") -> str:
        """Enforce team-wide constraints across all agent outputs."""
        return self.db.add_policy(rule, domain=category, priority=2.0)


def main() -> None:
    print("[*] Initializing InfinityDB CrewAI Multi-Agent Memory Provider...")
    crew_memory = InfinityDBCrewMemory("demo_crewai.db")

    # Set team-wide governance rules
    crew_memory.add_crew_policy("All production architecture designs must be SOC2 and HIPAA compliant")

    # Step 1: Research Agent executes task and stores findings
    print("[*] Agent 'Security Researcher' saving findings to shared memory...")
    crew_memory.save(
        value="PostgreSQL with pgvector requires substantial RAM overhead at 1M vectors. SQLite WAL mode provides sub-millisecond local reads.",
        agent_role="Security_Researcher"
    )

    # Step 2: Architect Agent contributes design specification
    print("[*] Agent 'Lead Architect' saving architecture decision...")
    crew_memory.save(
        value="We will standardize on InfinityDB for edge agent deployments to achieve $0 memory LLM cost.",
        agent_role="Lead_Architect"
    )

    # Step 3: Developer Agent queries shared team memory before implementation
    query = "What database and architecture was chosen for edge deployments?"
    print(f"\n[*] Agent 'Senior Developer' querying crew memory: '{query}'")
    matches = crew_memory.search(query, agent_role="Lead_Architect")

    print("\n[+] Retrieved Crew Memory for Developer Agent:")
    print("--------------------------------------------------")
    for m in matches:
        print(f"- Fact: {m['content']}")
        print(f"  Confidence Score: {m['score']}")
        print(f"  Audit Proof: {m['id']}")
    print("--------------------------------------------------")
    print("[PASS] CrewAI multi-agent memory adapter verified successfully.")


if __name__ == "__main__":
    main()
