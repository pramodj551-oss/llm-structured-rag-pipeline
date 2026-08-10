import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

from src.schemas import ExtractedSupportTicket

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY is not configured."
    )

client = genai.Client(api_key=API_KEY)


def extract_support_ticket(
    ticket_text: str,
) -> ExtractedSupportTicket:

    prompt = f"""
You are a support-ticket classification and extraction system.

Extract the following information from the support ticket:

- ticket_id
- category
- urgency
- sentiment
- one_line_summary

You MUST follow the provided structured schema.

Do not invent information.

Support Ticket:
{ticket_text}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ExtractedSupportTicket,
        ),
    )

    return ExtractedSupportTicket.model_validate_json(
        response.text
  )
