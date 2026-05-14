import logging
from models.base import BaseAdvancedModel
from config import ModelType, AdvancedConfig, ModelConfig


class ModelFactory:
    @staticmethod
    def create_model(mc: ModelConfig, cfg: AdvancedConfig) -> BaseAdvancedModel:
        if mc.type == ModelType.BEDROCK:
            from models.bedrock import BedrockChatModel
            return BedrockChatModel(mc, cfg)
        elif mc.type == ModelType.TX_AGENT:
            from models.txagent import TxAgentModel
            return TxAgentModel(mc, cfg)
        elif mc.type == ModelType.MEDREASON:
            from models.medreason import MedReasonModel
            return MedReasonModel(mc, cfg)
        elif mc.type == ModelType.MEDPATH_AGENT:
            from models.medpathagent import MedPathAgentModel
            return MedPathAgentModel(mc, cfg)
        elif mc.type == ModelType.AGENT_MD:
            from models.agentmd import AgentMDModel
            return AgentMDModel(mc, cfg)
        elif mc.type == ModelType.TOOLUNI:
            from models.tooluni import ToolUniModel
            return ToolUniModel(mc, cfg)
        else:
            raise ValueError(f"Unsupported model type: {mc.type}")
