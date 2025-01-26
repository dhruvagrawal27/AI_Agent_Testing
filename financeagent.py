# Install required dependencies: Run `pip install yfinance dotenv`

from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from dotenv import load_dotenv

load_dotenv()

def get_company_symbol(company: str) -> str:
    """
    Retrieves the stock symbol for a given company.
    
    Args:
        company (str): The name of the company.
    
    Returns:
        str: The stock symbol for the company, or "Unknown" if not found.
    """
    symbols = {
        "Reliance Power": "RPOWER.NS",
        "Sterling and Wilson Solar": "SWSOLAR.NS",
        "Suzlon": "SUZLON.NS",
        "Reliance Industries": "RELIANCE.NS",
        "Tata Power": "TATAPOWER.NS",
        "Adani Green Energy": "ADANIGREEN.NS",
    }
    return symbols.get(company, "Unknown")

agent = Agent(
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            stock_fundamentals=True
        ), 
        get_company_symbol
    ],
    instructions=[
        "Display data in well-formatted tables for better understanding.",
        "Use the get_company_symbol tool to fetch company symbols for analysis.",
        "Focus on important energy and solar module companies in the Indian stock market.",
    ],
    show_tool_calls=True,
    markdown=True,
    debug_mode=True,
)

try:
    agent.print_response(
        "Summarize stock price trends, analyst recommendations, and fundamentals for Reliance Power, Sterling and Wilson Solar, and Suzlon. Show in tables.",
        stream=True
    )
except Exception as e:
    print(f"Error: {e}")
