from crewai import Agent, Crew, Task, Process, LLM
from crewai.project import CrewBase, agent, task, crew
from crewai_tools import ScrapeWebsiteTool, SerperDevTool
from pydantic import BaseModel


llm = LLM(model="ollama/gemma3:4b")

class Episode(BaseModel):
    """
    Summary of the Episodes
    """
    title: str
    content: str


@CrewBase
class Write:
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def topic_researcher(self) -> Agent:
        return Agent(config=self.agents_config["topic_researcher"],
                     tools=[SerperDevTool(n_results=5), ScrapeWebsiteTool()],
                     llm=llm)

    @task
    def research_topic(self) -> Task:
        return Task(config=self.tasks_config["research_topic"])

    @agent
    def writer(self) -> Agent:
        return Agent(config=self.agents_config["writer"],
                     llm=llm)

    @task
    def write_chapter(self) -> Task:
        return Task(config=self.tasks_config["write_chapter"],
                    output_pydantic=Episode)

    @crew
    def crew(self) -> Crew:
        """Creates the Research Crew"""

        return Crew(agents=self.agents,
                    tasks=self.tasks,
                    process=Process.sequential,
                    verbose=True)