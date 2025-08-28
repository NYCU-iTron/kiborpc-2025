from dataclasses import dataclass
from typing import Optional

@dataclass
class GeneratorConfig:
    model_name: str = "gpt-3.5-turbo"
    max_retries: int = 3
    temperature: float = 0.7
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    system_prompt: Optional[str] = None
    user_prompt: Optional[str] = None