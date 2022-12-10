from .utils import save_json
import os
import numpy as np

class JsonSaver_IF:
    def __init__(self, base_path, influence_model):
        self.base_path = base_path
        self.influence_model = influence_model
        self.jsons = {}
    
    def save_result(self, influences, filename):
        influences = np.array(influences)
        helpful = np.argsort(influences)
        harmful = helpful[::-1]
        result = {}
        result["helpful"] = helpful[:500].tolist()
        result["harmful"] = harmful[:500].tolist()
        result["influence"] = influences.tolist()
        path = os.path.join(self.base_path, filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_json(result, path)

    def save_influence(self, influences, epoch):
        self.save_result(influences, os.path.join(self.influence_model, f"epoch{epoch}.json"))
    
    def save_first_iter_grad(self, first_iter_grad):
        self.save_result(first_iter_grad, "first_iteration_grad.json")