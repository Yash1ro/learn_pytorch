from torch.utils.data import Dataset
from PIL import Image
import os


class MyData(Dataset):

    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.path, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_path)


ants_root_dir = "data/hymenoptera_data/hymenoptera_data/train"
ants_label_dir = "ants"
bees_root_dir = "data/hymenoptera_data/hymenoptera_data/train"
bees_label_dir = "bees"
ants_dataset = MyData(ants_root_dir, ants_label_dir)
bees_dataset = MyData(bees_root_dir, bees_label_dir)

# reload plus sign
# we can use this way to put dataset together
train_dataset = ants_dataset + bees_dataset
