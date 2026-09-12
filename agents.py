from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools import web_search, scrape_url
from dotenv import load_dotenv
import os
import streamlit as st

load_dotenv()

# Model setup
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

# 1st agent
def build_search_agent():
    return create_agent(
        model=llm,
        tools=[web_search]
    )

# 2nd agent
def build_reader_agent():
    return create_agent(
        model=llm,
        tools=[scrape_url]
    )


# --------------------------------------------------
# WRITER CHAIN (first draft)
# --------------------------------------------------

writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research writer. Write clear, structured and insightful reports."),
    ("human", """Write a detailed research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Structure the report as:
- Introduction
- Key Findings (minimum 3 well-explained points)
- Conclusion
- Sources (list all URLs found in the research)

Be detailed, factual and professional."""),
])

writer_chain = writer_prompt | llm | StrOutputParser()


# --------------------------------------------------
# REVISION CHAIN (uses critic feedback to improve the report)
# --------------------------------------------------

revision_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research writer revising your own work based on editorial feedback. "
               "Address every point raised. Do not ignore any critique."),
    ("human", """Topic: {topic}

Here is the previous version of the report:
{previous_report}

Here is the critic's feedback on that version:
{feedback}

Rewrite the full report, incorporating the feedback directly. Keep the same structure
(Introduction, Key Findings, Conclusion, Sources), but improve depth, accuracy, and clarity
wherever the critic pointed out a weakness. Output the complete revised report only."""),
])

revision_chain = revision_prompt | llm | StrOutputParser()


# --------------------------------------------------
# CRITIC CHAIN
# --------------------------------------------------

critic_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a sharp and constructive research critic. Be honest and specific."),
    ("human", """Review the research report below and evaluate it strictly.

Report:
{report}

Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

One line verdict:
..."""),
])

critic_chain = critic_prompt | llm | StrOutputParser()