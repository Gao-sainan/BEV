from torchstat import stat
from bev import Bev
from config import Config
import torch

c = Config()
model_path = c.cp_dir

model = Bev(c.N)
model.load_state_dict(torch.load(c.cp_dir))

model.to(c.device)

stat(model, ((3, 480, 640), (3, 480, 640)))