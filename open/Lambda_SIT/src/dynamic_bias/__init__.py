from tensorrt_llm.dynamic_bias import router_manager

dynamic_bias_step = router_manager.dynamic_bias_step
get_dynamic_bias_manager = router_manager.get_dynamic_bias_manager

__all__ = ["dynamic_bias_step", "get_dynamic_bias_manager", "router_manager"]
