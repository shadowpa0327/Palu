def reset_h2o(model):
    for name, module in model.named_modules():
        if hasattr(module, '_reset_masks'):
            module._reset_masks()