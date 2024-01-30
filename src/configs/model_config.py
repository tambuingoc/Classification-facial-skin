import torch


class ModelConfig:
    def __init__(self):
        self.learning_rate = 0.001
        self.batch_size = 32
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.epochs = 5
        self.log_interval = 2  # Log every 2 batches => number of items is 32*2 = 64
        
        # Wandb config
        self.wandb = True
        self.wandb_project = "template-pytorch-model"
        self.wandb_entity = "nguyen"
        self.wandb_api_key = ""
        
        # Model 1 config
        # self.model1_path = "models/model1.pt"
        self.modelFore_path = "models/ForeGAN_32R.pt"
        self.modelSmile_path = "models/Smile85Gan_32R.pt"
        self.modelEye_path = "models/Eye85Gan_32R.pt"
        # self.modelPig_path = "models/last_model.pt"
        self.modelPore_path = "models/best_model_Pore65Res.pt"
        self.modelPig_path = "models/piggan69_32R.pt"

    def get_config(self):
        return self
config = ModelConfig().get_config()