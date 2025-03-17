from crewai import Agent, Crew, Task, Process, LLM
from crewai.project import CrewBase, agent, task, crew
from crewai_tools import ScrapeWebsiteTool, SerperDevTool
from pydantic import BaseModel, Field


llm = LLM(model="ollama/gemma3:4b")

class SeriesOutline(BaseModel):
    """
    Summary of the Episodes
    """
    total_episodes: int
    titles: list[str]


@CrewBase
class OutlineCrew:
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def research_agent(self) -> Agent:
        return Agent(
            config = self.agents_config["research_agent"],
            tools = [SerperDevTool(n_results=5), ScrapeWebsiteTool()],
            llm=llm
        )

    @task
    def research_task(self) -> Task:
        return Task(config=self.tasks_config["research_task"])

    @agent
    def outline_writer(self) -> Agent:
        return Agent(config=self.agents_config["outline_writer"],
                     llm=llm)

    @task
    def write_outline(self) -> Task:
        return Task(config=self.tasks_config["write_outline"],
                    output_pydantic=SeriesOutline)

    @crew
    def crew(self) -> Crew:
        """Creates the Outline Crew"""

        return Crew(agents=self.agents,
                    tasks=self.tasks,
                    process=Process.sequential,
                    verbose=True)