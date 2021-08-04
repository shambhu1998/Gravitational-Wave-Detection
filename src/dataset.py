from torchvision import transforms
from npy2image import increase_dimension
import torch

class G2Net:
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
        image = increase_dimension(self.path[idx], self.is_train)
        trans = transforms.ToTensor()
        image = trans(image)
        if self.is_train is not None:
            target = torch.tensor(self.target[idx], dtype=torch.float32)
            return {"image": image,
                    "target": target}
        else:
            return {"image": image}
