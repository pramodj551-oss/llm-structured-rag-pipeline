from enum import Enum
from pydantic import BaseModel, Field, field_validator

class UrgencyLevel(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"

class TicketCategory(str, Enum):
    BILLING = "Billing"
    TECHNICAL = "Technical Support"
    ACCOUNT = "Account Management"
    FEATURE_REQUEST = "Feature Request"
    SECURITY = "Security"

class SentimentLevel(str, Enum):
    POSITIVE = "Positive"
    NEUTRAL = "Neutral"
    NEGATIVE = "Negative"

class ExtractedSupportTicket(BaseModel):
    ticket_id: str = Field(description="Unique identifier of the ticket")
    category: TicketCategory = Field(description="Assigned category of the ticket")
    urgency: UrgencyLevel = Field(description="Assigned urgency level")
    sentiment: SentimentLevel = Field(description="Detected customer sentiment")
    one_line_summary: str = Field(description="A concise single-sentence summary of the user issue", max_length=150)

    @field_validator('one_line_summary')
    @classmethod
    def must_be_single_line(cls, v: str) -> str:
        if "\n" in v:
            raise ValueError("Summary must be a single line without line breaks.")
        return v.strip()
