import os
import asyncio

from pydantic import BaseModel, Field
from crewai.flow import Flow, listen, start

from webseries_writer.crew.outline.outline import OutlineCrew
from webseries_writer.crew.write.write import Write

from dotenv import load_dotenv

load_dotenv()



class Episode(BaseModel):
    title : str = ""
    content : str = ""

class WebSeries(BaseModel):
    topic : str = "Artificial Intelligence"
    total_episodes : int = 0
    titles : list[str] = []
    episodes : list[Episode] = []

class WebSeriesFlow(Flow[WebSeries]):
    @start()
    def generate_outline(self):
        print("Generating outline")
        outline = OutlineCrew().crew().kickoff(inputs={"topic": self.state.topic})
        self.state.total_episodes = outline.pydantic.total_episodes
        self.state.titles = outline.pydantic.titles

    @listen(generate_outline)
    async def generate_episodes(self):
        print("Generating chapters")
        tasks = []

        async def write_single_episode(title: str):
            result = (
                Write()
                .crew()
                .kickoff(inputs={
                    "title": title,
                    "topic": self.state.topic,
                    "episodes": [episode.title for episode in self.state.episodes]
                })
            )
            return result.pydantic

        # Create tasks for each chapter
        for i in range(self.state.total_episodes):
            task = asyncio.create_task(
                write_single_episode(
                    self.state.titles[i]
                )
            )
            tasks.append(task)

        # Wait for all chapters to be generated concurrently
        episodes = await asyncio.gather(*tasks)
        print(f"Generated {len(episodes)} episodes")
        self.state.episodes.extend(episodes)

    @listen(generate_episodes)
    def save_webseries(self):
        print("Saving Series")
        with open("webseries.md", "w") as f:
            for episode in self.state.episodes:
                f.write("# " + episode.title + "\n")
                f.write(episode.content + "\n")

def kickoff():
    book_flow = WebSeriesFlow()
    asyncio.run(book_flow.kickoff_async())


def plot():
    book_flow = WebSeriesFlow()
    book_flow.plot()


if __name__ == "__main__":
    kickoff()
    # plot()