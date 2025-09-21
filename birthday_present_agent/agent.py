"""Agent definition for the birthday present concierge."""

from google.adk.agents import Agent
from google.adk.tools import FunctionTool

from .prompt import birthday_present_agent_instruction
from .tools import fetch_product_details, fetch_social_profile, shopping_search

root_agent = Agent(
    model="gemini-2.5-flash",
    name="birthday_present_agent",
    instruction=birthday_present_agent_instruction,
    tools=[
        FunctionTool(func=shopping_search),
        FunctionTool(func=fetch_product_details),
        FunctionTool(func=fetch_social_profile),
    ],
)
