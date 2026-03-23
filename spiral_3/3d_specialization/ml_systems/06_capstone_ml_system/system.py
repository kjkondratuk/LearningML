"""ML System capstone: serving pipeline."""

class InferenceServer:
    def __init__(self, model_path: str, quantization: str = "int4"):
        raise NotImplementedError
    def generate(self, prompt: str, max_tokens: int = 256):
        raise NotImplementedError
