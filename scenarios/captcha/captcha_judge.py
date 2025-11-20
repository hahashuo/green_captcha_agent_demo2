import argparse
import asyncio
import contextlib
import json
import logging
import random
import shlex
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

import uvicorn
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    Part,
    TaskState,
    TextPart,
)
from a2a.utils import new_agent_text_message

from agentbeats.green_executor import GreenAgent, GreenExecutor
from agentbeats.models import EvalRequest
from agentbeats.tool_provider import ToolProvider


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("captcha_judge")


class CaptchaJudgeResult(BaseModel):
    puzzle_url: str
    solver_response: str


def captcha_judge_agent_card(agent_name: str, card_url: str) -> AgentCard:
    skill = AgentSkill(
        id="start_and_share_captcha",
        name="Start CAPTCHA app and share puzzle",
        description="Starts a local CAPTCHA Flask app and sends a puzzle URL to a participant.",
        tags=["captcha", "coordination"],
        examples=[
            """
{
  "participants": {
    "captcha_solver": "http://127.0.0.1:9019"
  },
  "config": {
    "captcha_type": "Dart_Count",
    "puzzle_id": "dart_puzzle_1.json",
    "captcha_host": "127.0.0.1",
    "captcha_port": 7861
  }
}
"""
        ],
    )
    agent_card = AgentCard(
        name=agent_name,
        description="Starts the CAPTCHA Flask service and coordinates sending a puzzle URL to a solver agent.",
        url=card_url,
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )
    return agent_card


class CaptchaJudge(GreenAgent):
    def __init__(self):
        self._required_roles = ["captcha_solver"]
        # captcha_host/captcha_port have defaults; puzzle selection can be random if not provided.
        self._required_config_keys: list[str] = []
        self._tool_provider = ToolProvider()

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        missing_roles = set(self._required_roles) - set(request.participants.keys())
        if missing_roles:
            return False, f"Missing roles: {missing_roles}"
        try:
            int(request.config.get("captcha_port", 7861))
        except Exception as exc:
            return False, f"Can't parse captcha_port: {exc}"
        return True, "ok"

    async def run_eval(self, req: EvalRequest, updater: TaskUpdater) -> None:
        app_host = req.config.get("captcha_host", "127.0.0.1")
        app_port = int(req.config.get("captcha_port", 7861))
        configured_type = req.config.get("captcha_type")
        configured_puzzle_id = req.config.get("puzzle_id")
        app_cmd = req.config.get("captcha_app_cmd")
        default_app_path = "/Users/chenhaishuo/Documents/OPENCAPTCHAWORLD/app2.py"
        app_path = Path(req.config.get("captcha_app_path", default_app_path))

        logger.info(
            "Starting CAPTCHA judge with host=%s port=%s type=%s puzzle=%s app_cmd=%s app_path=%s",
            app_host,
            app_port,
            configured_type,
            configured_puzzle_id,
            app_cmd,
            app_path,
        )

        process: Optional[asyncio.subprocess.Process] = None
        try:
            process = await self._start_flask_app(app_cmd, app_path, app_host, app_port, updater)
            await self._ensure_app_ready(app_host, app_port, updater)

            captcha_type, puzzle_id = await self._pick_random_puzzle(
                app_host,
                app_port,
                configured_type=configured_type,
                configured_puzzle_id=configured_puzzle_id,
                updater=updater,
            )

            puzzle_url = f"http://{app_host}:{app_port}/get_puzzle?type={captcha_type}&id={puzzle_id}"
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"CAPTCHA app running. Sharing puzzle URL {puzzle_url} with solver."),
            )

            solver_prompt = (
                "A local CAPTCHA Flask app is now running. "
                f"Open {puzzle_url} in your browser, follow the puzzle instructions, "
                "and return with the downloaded JSON result. "
                "Narrate briefly what you did and include the puzzle answer."
            )
            solver_response = await self._tool_provider.talk_to_agent(
                solver_prompt,
                str(req.participants["captcha_solver"]),
                new_conversation=False,
            )
            logger.info("Solver response: %s", solver_response)

            result = CaptchaJudgeResult(puzzle_url=puzzle_url, solver_response=solver_response)
            await updater.add_artifact(
                parts=[
                    Part(root=TextPart(text=result.model_dump_json())),
                    Part(root=TextPart(text=solver_response)),
                ],
                name="CaptchaResult",
            )
            await updater.update_status(
                TaskState.succeeded,
                new_agent_text_message("Solver responded. See CaptchaResult artifact for details."),
            )
        except Exception as exc:
            logger.exception("CaptchaJudge failed: %s", exc)
            await updater.update_status(TaskState.failed, new_agent_text_message(f"CaptchaJudge failed: {exc}"))
            raise
        finally:
            await self._shutdown_process(process)
            self._tool_provider.reset()

    async def _start_flask_app(
        self,
        app_cmd: Optional[str],
        app_path: Path,
        host: str,
        port: int,
        updater: TaskUpdater,
    ) -> asyncio.subprocess.Process:
        if app_cmd:
            cmd = shlex.split(app_cmd)
        else:
            if not app_path.exists():
                raise RuntimeError(f"CAPTCHA app path not found: {app_path}")
            cmd = ["python", str(app_path), "--host", host, "--port", str(port)]

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"Launching CAPTCHA Flask app with command: {' '.join(cmd)}"),
        )
        logger.info("Launching Flask app: %s", cmd)
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(f"Failed to start Flask app: {exc}") from exc

        asyncio.create_task(self._log_process_output(process))
        return process

    async def _pick_random_puzzle(
        self,
        host: str,
        port: int,
        configured_type: Optional[str],
        configured_puzzle_id: Optional[str],
        updater: TaskUpdater,
    ) -> tuple[str, str]:
        api_base = f"http://{host}:{port}"
        types_url = f"{api_base}/api/types"
        await updater.update_status(TaskState.working, new_agent_text_message(f"Fetching CAPTCHA types from {types_url}"))
        types_payload = await self._fetch_json(types_url)
        types = types_payload.get("types", [])
        if not types:
            raise RuntimeError("No CAPTCHA types returned from API.")

        if configured_type and configured_type in types:
            chosen_type = configured_type
        else:
            chosen_type = random.choice(types)
        logger.info("Chosen CAPTCHA type: %s (configured=%s)", chosen_type, configured_type)

        puzzles_url = f"{api_base}/api/list_puzzles?type={chosen_type}"
        await updater.update_status(TaskState.working, new_agent_text_message(f"Fetching puzzles from {puzzles_url}"))
        puzzles_payload = await self._fetch_json(puzzles_url)
        puzzles = puzzles_payload.get("puzzles", [])
        if not puzzles:
            raise RuntimeError(f"No puzzles returned for type {chosen_type}")

        if configured_puzzle_id and configured_puzzle_id in puzzles:
            chosen_puzzle = configured_puzzle_id
        else:
            chosen_puzzle = random.choice(puzzles)
        logger.info("Chosen puzzle: %s (configured=%s)", chosen_puzzle, configured_puzzle_id)

        return chosen_type, chosen_puzzle

    async def _fetch_json(self, url: str, timeout: float = 10.0) -> dict:
        def _blocking_fetch() -> dict:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read().decode("utf-8")
                return json.loads(data)

        try:
            return await asyncio.to_thread(_blocking_fetch)
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Failed to fetch JSON from {url}: {exc}") from exc

    async def _ensure_app_ready(self, host: str, port: int, updater: TaskUpdater, timeout: float = 15.0) -> None:
        await updater.update_status(TaskState.working, new_agent_text_message("Waiting for CAPTCHA app to be ready..."))
        start = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start < timeout:
            try:
                reader, writer = await asyncio.open_connection(host, port)
                writer.close()
                await writer.wait_closed()
                logger.info("CAPTCHA app responded on %s:%s", host, port)
                return
            except Exception:
                await asyncio.sleep(0.5)
        raise TimeoutError(f"CAPTCHA app did not become ready on {host}:{port} within {timeout} seconds.")

    async def _shutdown_process(self, process: Optional[asyncio.subprocess.Process]) -> None:
        if not process:
            return
        if process.returncode is not None:
            return
        logger.info("Terminating CAPTCHA Flask app...")
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), 5)
        except asyncio.TimeoutError:
            logger.warning("CAPTCHA Flask app did not exit gracefully; killing.")
            process.kill()
            await process.wait()

    async def _log_process_output(self, process: asyncio.subprocess.Process) -> None:
        if not process.stdout:
            return
        async for raw_line in process.stdout:
            line = raw_line.decode().rstrip()
            logger.info("[captcha-app] %s", line)


async def main():
    parser = argparse.ArgumentParser(description="Run the CAPTCHA judge.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9020, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="External URL to provide in the agent card")
    parser.add_argument(
        "--cloudflare-quick-tunnel",
        action="store_true",
        help="Use a Cloudflare quick tunnel. Requires cloudflared. This will override --card-url",
    )
    args = parser.parse_args()

    if args.cloudflare_quick_tunnel:
        from agentbeats.cloudflare import quick_tunnel

        agent_url_cm = quick_tunnel(f"http://{args.host}:{args.port}")
    else:
        agent_url_cm = contextlib.nullcontext(args.card_url or f"http://{args.host}:{args.port}/")

    async with agent_url_cm as agent_url:
        agent = CaptchaJudge()
        executor = GreenExecutor(agent)
        agent_card = captcha_judge_agent_card("CaptchaJudge", agent_url)

        request_handler = DefaultRequestHandler(
            agent_executor=executor,
            task_store=InMemoryTaskStore(),
        )

        server = A2AStarletteApplication(
            agent_card=agent_card,
            http_handler=request_handler,
        )

        uvicorn_config = uvicorn.Config(server.build(), host=args.host, port=args.port)
        uvicorn_server = uvicorn.Server(uvicorn_config)
        await uvicorn_server.serve()


if __name__ == "__main__":
    asyncio.run(main())
