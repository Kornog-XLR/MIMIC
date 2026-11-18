"""01. L'agent le plus simple : une question, une réponse texte.

Même agent en pydantic-ai puis en gwenflow, sur le même LLM.
"""

from pydantic_ai import Agent as PydanticAgent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from gwenflow import Agent as GwenflowAgent, ChatOpenAI

from settings import API_KEY, ENDPOINT, MODEL


INSTRUCTIONS = "Tu es un assistant qui répond en français, en une phrase."
QUESTION = "Quelle est la capitale de la Bretagne ?"


# --- pydantic-ai --------------------------------------------------------------

pydantic_agent = PydanticAgent(
    model=OpenAIChatModel(MODEL, provider=OpenAIProvider(base_url=ENDPOINT, api_key=API_KEY)),
    instructions=INSTRUCTIONS,
)


# --- gwenflow -----------------------------------------------------------------

gwenflow_agent = GwenflowAgent(
    llm=ChatOpenAI(model=MODEL, base_url=ENDPOINT, api_key=API_KEY),
    instructions=INSTRUCTIONS,
)


if __name__ == "__main__":
    print("=== pydantic-ai ===")
    print(pydantic_agent.run_sync(QUESTION).output)

    print("\n=== gwenflow ===")
    print(gwenflow_agent.run(QUESTION).content)