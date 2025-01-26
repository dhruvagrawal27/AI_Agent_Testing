from phi.agent import Agent
from phi.model.groq import Groq
from dotenv import load_dotenv

load_dotenv()

agent = Agent(
    model=Groq(id="llama-3.3-70b-versatile")
)

agent.print_response("Share a 3 sentence closing statement for love between dosa and samosa as imagine yourself as Mike Ross from Suits and represting as attorney for dosa and samosa infront of judge as Harvey Specter")