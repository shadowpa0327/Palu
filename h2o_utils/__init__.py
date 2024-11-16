from .llama_patch import monkey_patch_h2o as h2o_patch_llama
from .utils import reset_h2o as trigger_h2o_reset

from loguru import logger

def apply_h2o(model, model_name, h2o_comp_rate):
    logger.info(f"Applying H2O to {model_name} with compression rate {h2o_comp_rate}")
    heavy_ratio = h2o_comp_rate / 2
    recent_ratio = h2o_comp_rate / 2
    # log heavy ratio and recent_ratio
    logger.info(f"* Heavy ratio: {heavy_ratio}")
    logger.info(f"* Recent ratio: {recent_ratio}")
    if "llama" in model_name.lower():
        h2o_patch_llama(model, heavy_ratio, recent_ratio)
    else:
        raise NotImplementedError