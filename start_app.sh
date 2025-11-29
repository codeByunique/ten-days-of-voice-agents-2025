#!/bin/bash

# Start LiveKit Server
livekit-server --dev &

# Start Day 1 - Assistant Agent
(cd backend && uv run python src/assistant_agent.py dev) &

# Start Day 2 - Barista Agent
(cd backend && uv run python src/barista_agent.py dev) &

# Start Day 3 - Wellness Agent
(cd backend && uv run python src/wellness_agent.py dev) &

# Day 4 - Tutor Agent
(cd backend && uv run python src/tutor_agent.py dev) &

# Day 5 - SDR Agent
(cd backend && uv run python src/sdr_agent.py dev) &

# Day 6 - Fraud detection Agent
(cd backend && uv run python src/fraud_detection_agent.py dev) &

# Day 7 - Food & Grocery Ordering Agent
(cd backend && uv run python src/food_agent.py dev) &

# Day 8 – Game Master Agent
(cd backend && uv run python src/game_master_agent.py dev) &

# Start Frontend
(cd frontend && pnpm dev) &

# Wait for all background jobs
wait
