import os
import imageio
from torch.utils.data import Dataset

class BentleyBlizzardBlossoms(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        labels_path = os.path.join(root, 'labels.tsv')
        with open(labels_path) as f:
            lines = f.readlines()
            lines = [x.split('\t') for x in lines]
            self.db = {int(x[0]): x[1] for x in lines}
        self.keys = sorted(list(self.db.keys()))
        self.classes = sorted(set(self.db.values()))
    
    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, i):
        key = self.keys[i]
        labl = self.db[key]
        labl = self.classes.index(labl)
        feat = imageio.imread(os.path.join(self.root, '%04d.png' % key))
        if self.transform is not None:
            feat = self.transform(feat)
        return {'img': feat, 'label': labl}
