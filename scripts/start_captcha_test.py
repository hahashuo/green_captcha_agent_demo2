#!/usr/bin/env python3
"""
Script to start Cloudflare tunnel and captcha_judge agent with public card URL.
This script runs the captcha_judge agent with Cloudflare tunnel enabled and uses
configuration from scenarios/captcha/scenario.toml.
"""

import argparse
import asyncio
import os
import sys
import subprocess
import time
from pathlib import Path
import tomllib
import httpx

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from a2a.client import A2ACardResolver


def parse_scenario_config():
    """Parse the scenario.toml file to get configuration."""
    scenario_path = project_root / "scenarios" / "captcha" / "scenario.toml"
    
    if not scenario_path.exists():
        raise FileNotFoundError(f"Scenario file not found: {scenario_path}")
    
    with open(scenario_path, 'rb') as f:
        data = tomllib.load(f)
    
    return data


async def wait_for_agent_ready(endpoint: str, timeout: int = 30) -> bool:
    """Wait for the agent to be ready by checking its agent card."""
    print(f"Waiting for agent to be ready at {endpoint}...")
    start_time = time.time()
    
    async def check_endpoint() -> bool:
        try:
            async with httpx.AsyncClient(timeout=2) as client:
                resolver = A2ACardResolver(httpx_client=client, base_url=endpoint)
                await resolver.get_agent_card()
                return True
        except Exception as e:
            return False
    
    while time.time() - start_time < timeout:
        if await check_endpoint():
            print("Agent is ready!")
            return True
        print("  Waiting...")
        await asyncio.sleep(1)
    
    print(f"Timeout: Agent did not become ready after {timeout}s")
    return False


def start_captcha_judge_with_cloudflare():
    """Start the captcha_judge agent with Cloudflare tunnel."""
    # Parse scenario configuration
    scenario_config = parse_scenario_config()
    
    # Get the green agent configuration
    green_agent = scenario_config.get("green_agent", {})
    config = scenario_config.get("config", {})
    
    # Extract host and port from endpoint
    endpoint = green_agent.get("endpoint", "http://127.0.0.1:9020")
    host_port = endpoint.replace("http://", "").split("/")[0]
    host, port = host_port.split(":")
    port = int(port)
    
    # Build the command to start captcha_judge with Cloudflare tunnel
    captcha_judge_script = project_root / "scenarios" / "captcha" / "captcha_judge.py"
    
    cmd = [
        sys.executable,
        str(captcha_judge_script),
        "--host", host,
        "--port", str(port),
        "--cloudflare-quick-tunnel"
    ]
    
    print(f"Starting captcha_judge with Cloudflare tunnel...")
    print(f"Command: {' '.join(cmd)}")
    print(f"Configuration: {config}")
    
    # Start the process
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    
    return process, f"http://{host}:{port}"


async def monitor_process(process, local_url: str):
    """Monitor the process output and wait for it to be ready."""
    try:
        # Read and print output in real-time
        if process.stdout:
            for line in process.stdout:
                print(f"[captcha_judge] {line}", end='')
                
                # Check for Cloudflare tunnel URL in output
                if "https://" in line and ".trycloudflare.com" in line:
                    print(f"\n🎉 Cloudflare tunnel URL detected: {line.strip()}")
        
        # Wait for the agent to be ready
        ready = await wait_for_agent_ready(local_url)
        if ready:
            print(f"\n✅ captcha_judge is running with Cloudflare tunnel!")
            print(f"   Local URL: {local_url}")
            print(f"   Public URL: (check Cloudflare output above)")
            print(f"\nThe agent is ready for testing!")
            print(f"Configuration from scenario.toml:")
            scenario_config = parse_scenario_config()
            for key, value in scenario_config.get("config", {}).items():
                print(f"  - {key}: {value}")
        else:
            print(f"\n❌ Failed to start captcha_judge agent")
            
    except Exception as e:
        print(f"Error monitoring process: {e}")
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()


async def main():
    parser = argparse.ArgumentParser(description="Start captcha_judge with Cloudflare tunnel")
    parser.add_argument("--timeout", type=int, default=60, help="Timeout for agent readiness check (seconds)")
    args = parser.parse_args()
    
    try:
        # Start the captcha_judge agent with Cloudflare tunnel
        process, local_url = start_captcha_judge_with_cloudflare()
        
        # Monitor the process
        await monitor_process(process, local_url)
        
    except KeyboardInterrupt:
        print("\n🛑 Script interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())