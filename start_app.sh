#!/bin/bash

# Start LiveKit Server
livekit-server --dev &

# Start Day 1 - Assistant Agent
(cd backend && uv run python src/assistant_agent.py dev) &

# Start Day 2 - Barista Agent
(cd backend && uv run python src/barista_agent.py dev) &

# Start Frontend
(cd frontend && pnpm dev) &

# Wait for all background jobs
wait
