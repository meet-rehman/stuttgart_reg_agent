from pathlib import Path
import yaml
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.memory import LongTermMemory, ShortTermMemory, EntityMemory
from crewai.memory.storage.rag_storage import RAGStorage
from crewai.memory.storage.ltm_sqlite_storage import LTMSQLiteStorage
from stuttgart_reg_agent.schemas import (
    PlotDetails,
    ZoningRecommendation,
    BuildingCodeCompliance,
    AccessibilityAnalysis,
)

@CrewBase
class StuttgartRegAgent:
    _config_dir = Path(__file__).parent / "config"
    agents_config = str(_config_dir / "agents.yaml")
    tasks_config_file = _config_dir / "tasks.yaml"

    def __init__(self):
        with open(self.tasks_config_file, "r", encoding="utf-8") as f:
            self.tasks_config = yaml.safe_load(f)

    @agent
    def coordinator_agent(self) -> Agent:
        return Agent(
            role="Coordinator agent",
            goal="Manage and delegate tasks to Zoning, Building Code, and Accessibility agents.",
            backstory="I oversee the workflow of multiple agents for Stuttgart building regulations.",
            allow_delegation=True,
            memory=True,
            tools=[]
        )

    @agent
    def zoning_agent(self) -> Agent:
        return Agent(
            role="Zoning expert agent",
            goal="Analyze zoning regulations for a plot in Stuttgart based on location, size, type, floors, and height.",
            backstory="I am specialized in Stuttgart zoning laws, LBO/BauGB rules, and municipal regulations.",
            memory=True,
            tools=[]
        )

    @agent
    def building_code_agent(self) -> Agent:
        return Agent(
            role="Building code expert",
            goal="Check building compliance according to LBO, BauGB, and Stuttgart municipal rules.",
            backstory="I am specialized in building code regulations, ensuring projects comply with legal requirements.",
            memory=True,
            tools=[]
        )

    @agent
    def accessibility_agent(self) -> Agent:
        return Agent(
            role="Accessibility expert",
            goal="Assess accessibility and barrierefreies Bauen compliance.",
            backstory="I am an expert in accessibility regulations for residential and commercial buildings in Stuttgart.",
            memory=True,
            tools=[]
        )

    @task
    def analyze_zoning(self) -> Task:
        return Task(
            config=self.tasks_config.get('analyze_zoning'),
            expected_output="A zoning recommendation with allowed building type, max floors, max height, and notes.",
            output_pydantic=ZoningRecommendation,
        )

    @task
    def check_building_code(self) -> Task:
        return Task(
            config=self.tasks_config.get('check_building_code'),
            expected_output="A building code compliance check showing whether the project is compliant and listing violations.",
            output_pydantic=BuildingCodeCompliance,
        )

    @task
    def assess_accessibility(self) -> Task:
        return Task(
            config=self.tasks_config.get('assess_accessibility'),
            expected_output="An accessibility analysis showing compliance and listing recommendations for improvements.",
            output_pydantic=AccessibilityAnalysis,
        )

    @crew
    def crew(self) -> Crew:
        manager = self.coordinator_agent()
        other_agents = [a for a in self.agents if a != manager]

        return Crew(
            agents=other_agents,
            tasks=self.tasks,
            process=Process.hierarchical,
            verbose=True,
            manager_agent=manager,
            memory=True,
            long_term_memory=LongTermMemory(
                storage=LTMSQLiteStorage(db_path=str(Path.cwd() / "memory" / "long_term_memory_storage.db"))
            ),
            short_term_memory=ShortTermMemory(
                storage=RAGStorage(
                    embedder_config={"provider": "huggingface", "config": {"model": "all-MiniLM-L6-v2"}},
                    type="short_term",
                    path=str(Path.cwd() / "memory" / "short_term")
                )
            ),
            entity_memory=EntityMemory(
                storage=RAGStorage(
                    embedder_config={"provider": "huggingface", "config": {"model": "all-MiniLM-L6-v2"}},
                    type="short_term",
                    path=str(Path.cwd() / "memory" / "entity")
                )
            )
        )
