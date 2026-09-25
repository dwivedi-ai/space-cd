"""Per-request intelligence: which model and effort an agent runs a chat with.

An agent opts in by shipping ``config/agents/<name>/intelligence.json``
(``profiles.py`` reads it). Core code here never names an agent; an agent
without the file runs exactly as before.
"""
