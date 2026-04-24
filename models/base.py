from dataclasses import dataclass
from typing import Dict, Optional, Any
from datetime import datetime
import re


@dataclass
class ReasoningResult:
    reasoning: str
    answer: str
    model_name: str
    verification_status: Dict = None
    timestamp: datetime = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.verification_status is None:
            self.verification_status = {}
        if self.timestamp is None:
            self.timestamp = datetime.now()


class BaseAdvancedModel:
    def __init__(self, model_config, advanced_config):
        self.model_config = model_config
        self.config = advanced_config
        self.model = None
        self.tokenizer = None
        self.device = model_config.device
        self.total_tokens = 0
        self.total_time = 0
        self.cache = {}
        import logging
        self.logger = logging.getLogger(f"{self.__class__.__name__}.{model_config.name}")
