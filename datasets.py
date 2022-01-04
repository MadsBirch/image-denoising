from torch.utils.data import Dataset
from PIL import Image


class TransformedDataset(Dataset):
    def __init__(self, dataset, transform, input_transform):
        self.dataset = dataset
        self.transform = transform
        self.input_transform = input_transform
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x, y = self.dataset[index]
        x_in = self.input_transform(x)
        x_tran = self.transform(x)
        return x_tran, y, x_in

class CatsAndDogs(Dataset):
    def __init__(self, dataset, transform, input_transform):
        self.dataset = self.checkChannel(dataset) # some images are CMYK, Grayscale, check only RGB 
        self.transform = transform
        self.input_transform = input_transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, item):
        x = Image.open(self.dataset[item][0])
        y = self.dataset[item][1]

        x_in = self.input_transform(x)
        x_tran = self.transform(x)

        return x_tran, y, x_in

    
    def checkChannel(self, dataset):
        datasetRGB = []
        for index in range(len(dataset)):
            if (Image.open(dataset[index][0]).getbands() == ("R", "G", "B")): # Check Channels
                datasetRGB.append(dataset[index])
        return datasetRGB