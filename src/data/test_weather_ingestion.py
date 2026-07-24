"""Self-check for the Open-Meteo weighted call-cost accounting (see _call_cost).
Run: uv run -m src.data.test_weather_ingestion
"""

from src.data.weather_ingestion import _call_cost

# Open-Meteo's own documented example: 2 weeks x 15 variables, 1 location = 1.5 calls.
assert abs(_call_cost(1, "2024-01-01", "2024-01-15", 15) - 1.5) < 1e-9
# 4 weeks x 15 variables, 1 location = 3.0 calls.
assert abs(_call_cost(1, "2024-01-01", "2024-01-29", 15) - 3.0) < 1e-9
# At or under both thresholds (<=2 weeks, <=10 vars), a single location costs 1 call.
assert _call_cost(1, "2024-01-01", "2024-01-15", 10) == 1.0
assert _call_cost(1, "2024-01-01", "2024-01-08", 5) == 1.0
# Cost scales linearly with batched locations.
assert _call_cost(5, "2024-01-01", "2024-01-15", 15) == 5 * 1.5

print("weather_ingestion self-check passed: _call_cost matches Open-Meteo's documented multiplier")
