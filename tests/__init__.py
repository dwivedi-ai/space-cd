"""XO Cowork API tests."""

import os

# Decision-model routing stays off in tests, whatever the checkout's .env says.
# A developer's .env may set XO_INTELLIGENCE_MODE=shadow and a real
# LEVANTO_API_KEY, and importing server.py loads it. load_dotenv() never
# overrides a non-empty export, and server.py refills a blank one from .env,
# so both are pinned to non-empty values here, before any test imports it.
# Tests that need a mode or a key patch them in.
os.environ["XO_INTELLIGENCE_MODE"] = "off"
os.environ["LEVANTO_API_KEY"] = "test-placeholder-not-a-key"
