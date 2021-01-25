from torch.utils.data import Dataset
from PIL import Image

class My_Dataset(Dataset):

    def __init__(self, root='./data', train=True, transform=None, target_transform=None):

        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        if self.train:
            img_index = root + '/train.txt'
            img_folder = root + '/images/train/'
        else:
            img_index = root + '/test.txt'
            img_folder = root + '/images/test/'

        self.img_dir = []
        #self.labels = []
        self.img_folder = img_folder
        with open(img_index) as fp:
            data = fp.read()
            temp = data.split()
            for i in temp:
                self.img_dir.append(img_folder + i + '.jpg')


    def __getitem__(self, index):

        img = Image.open(self.img_dir[index])

        label = self.target_transform(img)
        out = self.transform(img)

        return out, label


    def __len__(self):
        return len(self.img_dir)