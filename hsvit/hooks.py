import torch

class FeatureHook:
    def __init__(self, module, retain_grad=False):
        self.features = None
        self.retain_grad = retain_grad
        self.hook = module.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        if self.retain_grad:
            output.retain_grad()
        self.features = output.detach().cpu()

    def close(self):
        self.hook.remove()

