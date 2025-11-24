#!/bin/bash
# Script to start Cloudflare tunnel and captcha_judge agent with public card URL
# This script runs the captcha_judge agent with Cloudflare tunnel enabled

set -e

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "🚀 Starting captcha_judge with Cloudflare tunnel..."

# Check if Python script exists
CAPTCHA_JUDGE_SCRIPT="scenarios/captcha/captcha_judge.py"
if [ ! -f "$CAPTCHA_JUDGE_SCRIPT" ]; then
    echo "❌ Error: captcha_judge.py not found at $CAPTCHA_JUDGE_SCRIPT"
    exit 1
fi

# Check if cloudflared is installed
if ! command -v cloudflared &> /dev/null; then
    echo "❌ Error: cloudflared is not installed. Please install it first:"
    echo "  brew install cloudflared  # macOS"
    echo "  or visit: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/install-and-setup/tunnel-guide/local/"
    exit 1
fi

echo "📋 Using configuration from scenarios/captcha/scenario.toml:"
cat scenarios/captcha/scenario.toml | grep -E "(endpoint|cmd|config)" | sed 's/^/  /'

echo ""
echo "🔧 Starting captcha_judge with Cloudflare tunnel..."
echo "   This will create a public URL for testing the agent"

# Run the captcha_judge with Cloudflare tunnel
python "$CAPTCHA_JUDGE_SCRIPT" \
    --host 127.0.0.1 \
    --port 9020 \
    --cloudflare-quick-tunnel

echo ""
echo "✅ Script completed. The captcha_judge agent should now be running with a public Cloudflare URL."
echo "   Check the output above for the public URL (look for 'https://*.trycloudflare.com')"