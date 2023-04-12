import polars as pl
import os
from langchain.agents import create_csv_agent
from langchain.llms import OpenAI

openai_key = os.environ["OPENAI_API_KEY"]


df = pl.read_csv("taxi.csv")

agent = create_csv_agent(OpenAI(temperature=0),
                         "taxi.csv",
                         verbose = True)

#agent.run("Can you tell me what is the day with the most amount of rides and show me the top 5 in ascending order")
agent.run("What is the average amount of passengers per ride, show me that per each day")
