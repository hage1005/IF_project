import json
import os 
from .utils import save_json
import numpy as np

class Path_IF:
    def __init__(self, args):
        self.base_path = os.path.join(
            "outputs",
            args.dataset_name,
            args.classification_model,
            "dev_id_" + str(args.dev_id_num),
            f"pretrain{args.max_checkpoint_epoch}epoch"
        )

        self.ckpt_dir = os.path.join("checkpoints/fenchel", args.dataset_name, args.classification_model) 
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.inv_hessian_json_path = os.path.join(self.base_path, "Percy.json")
        self.identity_json_path = os.path.join(self.base_path, "Identity.json")

        self.pretrain_ckpt_path = os.path.join(self.ckpt_dir,
        f"epoch{args.max_pretrain_epoch}_lr{args.pretrain_classification_lr}_{args._pretrain_ckpt_name}")
        self.inv_hessian_path = os.path.join(self.ckpt_dir, "numpy_inv_hessian_" + os.path.basename(self.pretrain_ckpt_path))

        self.influence_model = args.influence_model
    
    def get_inv_hessian_influences(self):
        with open (self.inv_hessian_json_path, "r") as f:
            result_true = json.loads(f.read())
        return result_true
    
    def get_identity_influences(self):
        with open (self.identity_json_path, "r") as f:
            result_identity = json.loads(f.read())
        return result_identity

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

    def save_percy_influence(self, percy_influence):
        self.save_result(percy_influence, "Percy.json")
    
    def save_identity_influence(self, identity_influences):
        self.save_result(identity_influences, "Identity.json")