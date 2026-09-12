from langchain.tools import tool
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient
import os
from dotenv import load_dotenv

load_dotenv()

tavily = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)


# =========================
# WEB SEARCH TOOL
# =========================

class WebSearchInput(BaseModel):
    query: str = Field(
        description="The search query to find recent and reliable information."
    )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def _tavily_search(query: str, max_results: int = 5):
    return tavily.search(query=query, max_results=max_results)


@tool(args_schema=WebSearchInput)
def web_search(query: str) -> str:
    """
    Search the web using Tavily and return titles,
    URLs, and snippets. Retries on transient failures.
    """
    try:
        results = _tavily_search(query)
    except Exception as e:
        return f"Search failed after retries: {str(e)}"

    out = []
    for r in results["results"]:
        out.append(
            f"Title: {r['title']}\n"
            f"URL: {r['url']}\n"
            f"Snippet: {r['content'][:300]}"
        )

    return "\n----\n".join(out)


# =========================
# SCRAPE URL TOOL
# =========================

class ScrapeURLInput(BaseModel):
    url: str = Field(
        description="The complete URL of the webpage to scrape."
    )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(requests.exceptions.RequestException),
    reraise=True,
)
def _fetch(url: str):
    response = requests.get(
        url,
        timeout=8,
        headers={"User-Agent": "Mozilla/5.0"}
    )
    response.raise_for_status()
    return response


@tool(args_schema=ScrapeURLInput)
def scrape_url(url: str) -> str:
    """
    Scrape a webpage and return clean text content
    for deeper reading. Retries on transient network failures.
    """
    try:
        response = _fetch(url)

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        for tag in soup([
            "script",
            "style",
            "nav",
            "footer",
            "header"
        ]):
            tag.decompose()

        return soup.get_text(
            separator=" ",
            strip=True
        )[:2500]

    except Exception as e:
        return f"Could not scrape URL after retries: {str(e)}"