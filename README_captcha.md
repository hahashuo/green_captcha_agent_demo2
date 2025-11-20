## Quickstart

2. Install dependencies
```
uv sync
```
3. Set environment variables
```
cp sample.env .env
```
Add your Google API key to the .env file

4. Change captcha_app_path in scenarios/captcha/scenario.toml to path of maosheng's app2.py

4. Run the [captcha example](#example)
```
uv run agentbeats-run scenarios/captcha/scenario.toml
```
