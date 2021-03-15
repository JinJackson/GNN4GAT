import os
import enum
from torch.utils.tensorboard import SummaryWriter

class LayerType(enum.Enum):
    IMP1 = 0,
    IMP2 = 1,
    IMP3 = 2
