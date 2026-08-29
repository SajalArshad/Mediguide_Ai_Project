"""
config.py
Application configuration: API key handling, model defaults, and static
form options used to build the Streamlit UI.
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load variables from a local .env file (if present) into the environment.
load_dotenv()

# ---------------------------------------------------------------------------
# API KEY HANDLING
# ---------------------------------------------------------------------------
# MediGuide AI lets a visitor paste their own OpenAI API key in the sidebar
# so the app can be shared/demoed without exposing a shared project key.
# If the visitor doesn't provide one, we fall back to OPENAI_API_KEY from
# the environment / .env file (useful for local development).

ENV_API_KEY = os.getenv("OPENAI_API_KEY", "")


def resolve_api_key(user_supplied_key: Optional[str]) -> str:
    """Return the API key to use: the visitor's key takes priority."""
    if user_supplied_key and user_supplied_key.strip():
        return user_supplied_key.strip()
    return ENV_API_KEY


# ---------------------------------------------------------------------------
# MODEL DEFAULTS
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "gpt-4o-mini"
AVAILABLE_MODELS = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
DEFAULT_TEMPERATURE = 0.3

# ---------------------------------------------------------------------------
# FORM OPTIONS
# ---------------------------------------------------------------------------
GENDER_OPTIONS = ["Female", "Male", "Other", "Prefer not to say"]

SYMPTOM_OPTIONS = [
    "Fever", "Cough", "Sore throat", "Runny nose", "Headache",
    "Fatigue", "Shortness of breath", "Chest pain", "Nausea",
    "Vomiting", "Diarrhea", "Abdominal pain", "Muscle aches",
    "Dizziness", "Rash", "Joint pain", "Loss of appetite",
    "Chills", "Sweating", "Loss of taste or smell",
]

DURATION_OPTIONS = [
    "Less than 1 day", "1-3 days", "4-7 days",
    "1-2 weeks", "More than 2 weeks",
]

LANGUAGE_OPTIONS = ["English", "Urdu", "Roman Urdu", "Spanish", "French", "Arabic"]

CACHE_OPTIONS = ["No caching", "In-memory cache", "SQLite cache"]

URGENCY_COLORS = {
    "LOW": "\U0001F7E2",       # green circle
    "MEDIUM": "\U0001F7E1",    # yellow circle
    "HIGH": "\U0001F7E0",      # orange circle
    "EMERGENCY": "\U0001F534",  # red circle
}

APP_NAME = "MediGuide AI"
APP_TAGLINE = "Educational AI symptom guidance assistant \u2014 not a substitute for a doctor."
