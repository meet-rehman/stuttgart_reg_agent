# Clean search_agent.py implementation
from dataclasses import dataclass

@dataclass
class WebSearchTool:
    name: str = "web_search"
    description: str = "Search the web"

class Agent:
    def __init__(self, tools=None):
        self.tools = tools or []

class ModelSettings:
    def __init__(self, model="gpt-3.5-turbo"):
        self.model = model

search_agent = Agent(tools=[WebSearchTool()])
