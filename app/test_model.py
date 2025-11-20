import torch
from torchvision import models

state_dict = torch.load("../models/thar_wrangler.pth", map_location="cpu")
print("Loaded keys:", list(state_dict.keys())[:5])
