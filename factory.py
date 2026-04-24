import logging
from medperturb_eval.models.base import BaseAdvancedModel
from medperturb_eval.config import ModelType, AdvancedConfig, ModelConfig


class ModelFactory:
    @staticmethod
    def create_model(mc: ModelConfig, cfg: AdvancedConfig) -> BaseAdvancedModel:
        if mc.type == ModelType.BEDROCK:
            from medperturb_eval.models.bedrock import BedrockChatModel
            return BedrockChatModel(mc, cfg)
        elif mc.type == ModelType.TX_AGENT:
            from medperturb_eval.models.txagent import TxAgentModel
            return TxAgentModel(mc, cfg)
        else:
            raise ValueError(f"Unsupported model type: {mc.type}")
