import argparse
import uvicorn
from dotenv import load_dotenv
load_dotenv()

from google.adk.agents import Agent
from google.adk.a2a.utils.agent_to_a2a import to_a2a

from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)

SYSTEM_PROMPT = """
You are a browser_use-style automation agent that specializes in solving CAPTCHAs on web pages.

You do NOT need real browser tools. Pretend you can drive a headless browser with actions like go_to_url, wait_for_selector, click, type, screenshot, and OCR. Narrate the steps you would take and keep hallucinated details plausible.

When asked to solve a CAPTCHA:
- Briefly outline the page state you expect and the actions you'd perform to surface the challenge.
- Imagine reading the CAPTCHA (text, checkbox, image grid, or slider) and describe the intended interaction.
- Provide a concise action log (as if using browser_use) and finish with `Captcha solution: <your best guess>`.
- Stay within one response; do not ask for clarification or external resources.
"""

def main():
    parser = argparse.ArgumentParser(description="Run a browser_use-style CAPTCHA solver agent.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9019, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="External URL to provide in the agent card")
    args = parser.parse_args()

    root_agent = Agent(
        name="captcha_solver",
        model="gemini-2.0-flash",
        description="Acts like a browser_use agent that narrates how it would solve a CAPTCHA and returns the guessed solution.",
        instruction=SYSTEM_PROMPT,
    )

    skill = AgentSkill(
        id="solve_captcha_browser_use",
        name="Pretend browser_use CAPTCHA solver",
        description="Narrate browser-use style steps to solve a CAPTCHA and provide the final guessed solution.",
        tags=["captcha", "browser_use"],
        examples=[
            "Navigate to https://example.com/captcha, describe how you would load the widget, imagine the grid or text, and reply with the final Captcha solution: XY7KQ."
        ],
    )

    agent_card = AgentCard(
        name="captcha_solver",
        description="Browser-use style agent that pretends to solve CAPTCHAs and reports the guessed answer.",
        url=args.card_url or f'http://{args.host}:{args.port}/',
        version='1.0.0',
        default_input_modes=['text'],
        default_output_modes=['text'],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )

    a2a_app = to_a2a(root_agent, agent_card=agent_card)
    uvicorn.run(a2a_app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
