"""
Configuration module for loading environment variables.

This module loads environment variables from .env file and exposes
configuration values for OpenAI API and other services.
"""

import os
from dotenv import load_dotenv
load_dotenv()

# OpenAI configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PDF_MODEL = os.getenv("PDF_MODEL", "gpt-4o")
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "gpt-4o")
ENABLE_OPENAI = os.getenv("ENABLE_OPENAI", "true").lower() in ("true", "1", "yes", "on")
