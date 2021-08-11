from torchvision import transforms
import npy2image
import torch
from torch.utils.data import Dataset

class G2Dataset(Dataset):
    def __init__(self,path,target=None):
        self.path = path
        if target is not None:
            self.is_train=True
        else:
            self.is_train=False
        self.target = target

    def __len__(self):
        return len(self.path)

    def __getitem__(self,idx):
        image = npy2image.increase_dimension(self.path[idx])
        trans = transforms.ToTensor()
        image = trans(image)
        if self.is_train is not None:
            target = torch.tensor(self.target[idx], dtype=torch.float32)
            return {"image": image,
                    "target": target}
        else:
            return {"image": image}
