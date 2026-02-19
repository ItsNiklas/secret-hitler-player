#!/bin/bash
# Launch a human experiment game: 1 LLM vs 4 humans on external website.
# The operator relays messages and inputs choices from the website.
#
# Usage:
#   ./human.sh                                   # Use config-human.yaml defaults
#   ./human.sh --player-names LLMBot,John,Jane,Mike,Sarah
#   ./human.sh --first-president 2               # Set starting president by index

set -e
cd "$(dirname "$0")/simulator"

echo "=============================================="
echo "  SECRET HITLER – HUMAN EXPERIMENT MODE"
echo "=============================================="
echo ""
echo "  1 LLM player  vs  4 human players (website)"
echo "  You are the middleman."
echo ""
echo "  Workflow per turn:"
echo "    1. Input human actions from the website"
echo "    2. Copy LLM decisions to the website"
echo "    3. Input drawn policies from the website deck"
echo ""
echo "=============================================="
echo ""

python HitlerGame.py --config ../config-human.yaml "$@"
