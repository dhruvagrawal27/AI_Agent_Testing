import os
from typing import List, Optional
from dotenv import load_dotenv
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools

# Load environment variables
load_dotenv()

def create_web_agent(
    model_id: str = "llama3-70b-8192", 
    search_tool: Optional[object] = None
) -> Agent:
    """
    Create a web research agent with configurable search tool.
    """
    if search_tool is None:
        search_tool = DuckDuckGo()
    
    return Agent(
        name="Web Research Agent",
        model=Groq(id=model_id),
        tools=[search_tool],
        instructions=[
            "Always include credible sources",
            "Provide concise and accurate information",
            "Cross-reference multiple sources"
        ],
        show_tool_calls=True,
        markdown=True,
        max_tokens=4096
    )

def create_finance_agent(
    model_id: str = "llama3-70b-8192"
) -> Agent:
    """
    Create a finance data agent.
    """
    return Agent(
        name="Finance Intelligence Agent",
        role="Comprehensive financial analysis",
        model=Groq(id=model_id),
        tools=[YFinanceTools(
            stock_price=True, 
            analyst_recommendations=True, 
            company_info=True,
            key_financial_ratios=True
        )],
        instructions=[
            "Use markdown tables for data presentation",
            "Provide context for financial metrics",
            "Highlight key insights and trends"
        ],
        show_tool_calls=True,
        markdown=True,
        max_tokens=4096
    )

def create_multi_agent_team(
    agents: Optional[List[Agent]] = None,
    model_id: str = "llama3-70b-8192"
) -> Agent:
    """
    Create a multi-agent team with configurable agents.
    """
    if agents is None:
        agents = [
            create_web_agent(),
            create_finance_agent()
        ]
    
    return Agent(
        name="Multi-Agent Research Team",
        model=Groq(id=model_id),
        team=agents,
        instructions=[
            "Collaborate and synthesize information",
            "Ensure comprehensive and accurate reporting",
            "Use clear and professional communication"
        ],
        show_tool_calls=True,
        markdown=True,
        max_tokens=8192
    )

def main():
    """
    Main execution function for multi-agent research.
    """
    try:
        # Create multi-agent team
        agent_team = create_multi_agent_team()
        
        # Example queries
        queries = [
            "Summarize analyst recommendations and latest news for SUZLON",
            "Provide financial analysis for Reliance Power (RPOWER)"
        ]
        
        # Execute queries
        for query in queries:
            print(f"\n--- Researching: {query} ---")
            agent_team.print_response(query, stream=True)
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()