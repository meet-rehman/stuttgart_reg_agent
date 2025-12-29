# coordinator_agent.py

import yaml
from pathlib import Path
from precomputed_rag import EnhancedPrecomputedRAGSystem as PrecomputedRAGSystem

class CoordinatorAgent:
    def __init__(self, rag_system: PrecomputedRAGSystem, tasks_path="tasks.yaml", agents_path="agents.yaml"):
        self.rag_system = rag_system
        self.tasks = self._load_yaml(tasks_path)
        self.agents = self._load_yaml(agents_path)

    def _load_yaml(self, file_path):
        p = Path(file_path)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        return {}

    def route(self, query: str):
        """
        Naive routing: checks query keywords and selects an agent/task.
        Later you can make this smarter with embeddings or LLM.
        """
        query_lower = query.lower()

        if any(word in query_lower for word in ["zoning", "plot", "bebauungsplan"]):
            agent = "zoning_agent"
            task = "analyze_zoning"
        elif any(word in query_lower for word in ["height", "lbo", "building code", "compliance"]):
            agent = "building_code_agent"
            task = "check_building_code"
        elif any(word in query_lower for word in ["accessibility", "barrierefrei"]):
            agent = "accessibility_agent"
            task = "assess_accessibility"
        else:
            agent = "coordinator_agent"
            task = None

        # Always fetch RAG context
        context = self.rag_system.get_context_for_query(query, max_tokens=1500)

        return {
            "agent": agent,
            "task": task,
            "context": context
        }
