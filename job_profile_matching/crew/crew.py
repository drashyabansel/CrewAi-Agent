from crewai import Agent, Task, Crew, LLM
from crewai.project import CrewBase, agent, task, crew
from crewai_tools import SerperDevTool, ScrapeWebsiteTool, FileReadTool, PDFSearchTool

llm = LLM(model="ollama/gemma3:4b")

@CrewBase
class JobProfilling:
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"
    read_resume = FileReadTool("./DRASHYA BANSEL Resume.pdf")
    semantic_serach_tool = PDFSearchTool(
        pdf="./DRASHYA BANSEL Resume.pdf",
        config=dict(
            llm=dict(
                provider="ollama",
                config=dict(
                    model="gemma3:4b"
                ),
            ),
            embedder=dict(
                provider="ollama",
                config=dict(
                    model="gemma3:4b",
                ),
            ),
        )
    )

    @agent
    def tech_job_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config["tech_job_researcher"],
            llm=llm,
        )

    @task
    def tech_job_research_task(self) -> Task:
        return Task(
            config = self.tasks_config["tech_job_research_task"]
        )

    @agent
    def job_profiler(self) -> Agent:
        return Agent(
            config = self.agents_config["job_profiler"],
            llm=llm,
            tools=[SerperDevTool(), ScrapeWebsiteTool(), self.read_resume, self.semantic_serach_tool]
        )

    @task
    def job_profiller_task(self):
        return Task(config = self.tasks_config["job_profiller_task"])

    @agent
    def resume_strategist(self) -> Agent:
        return Agent(
            config = self.agents_config["resume_strategist"],
            llm = llm,
            tools=[SerperDevTool(), ScrapeWebsiteTool(), self.read_resume, self.semantic_serach_tool]
        )

    @task
    def resume_strategy_task(self):
        return Task(config = self.tasks_config["resume_strategy_task"])

    @agent
    def interview_preparer(self) -> Agent:
        return Agent(
            config = self.agents_config["interview_preparer"],
            llm=llm,
            tools=[SerperDevTool(), ScrapeWebsiteTool(), self.read_resume, self.semantic_serach_tool]
        )

    @task
    def interview_preparation_task(self):
        return Task(config=self.tasks_config["interview_preparation_task"])

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            verbose=True
        )