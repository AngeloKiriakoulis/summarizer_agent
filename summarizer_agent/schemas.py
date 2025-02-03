from pydantic import BaseModel
from typing import Union, Dict, Any, List, Optional

class InputSchema(BaseModel):
    tool_name: str
    tool_input_data: Optional[Union[Dict[str, Any], List[Dict[str, Any]], str]] = None

class SystemPromptSchema(BaseModel):
    """Schema for system prompts."""
    role: str = "You are a helpful AI assistant. You will summarize the given text in 3 sentences max."
    persona: Optional[Union[Dict, BaseModel]] = None