from torch.utils.data import Dataset
from PIL import Image
from PIL import ImageFile

import os
from glob import glob
from torchvision import transforms
from torch.utils.data.dataset import Dataset

import torch
import pdb






ImageFile.LOAD_TRUNCATED_IMAGES = True


import os
from fnmatch import fnmatch


class VimeoAndArod(Dataset):
    def __init__(self, data_dir,text_dir ,data_dir_arod, image_size=256):
        self.text_dir  = text_dir
        self.data_dir = data_dir
        self.data_dir_arod = data_dir_arod
        self.image_size = image_size
        pattern = ".png"
        self.image_path = [] #sorted(glob(os.path.join(self.data_dir, "*.*")))
        cn = 0
        for path, _, files in os.walk(self.data_dir):
            for name in files:
                cn = cn + 1

                if ".png" in os.path.join(path, name):   
                    self.image_path.append(os.path.join(path, name))
        
        self.image_path = self.image_path[:15000]
        
        self.arod_files = os.listdir(self.data_dir_arod)
        
        
        for i in range(len(self.arod_files)):
            self.image_path.append(os.path.join(self.data_dir_arod, self.arod_files[i]))

        print(len(self.image_path))
        #self.image_path = sorted(glob(os.path.join(self.data_dir, "*.*")))
        
    def __getitem__(self, item):
        image_ori = self.image_path[item]

        #image = cv2.imread(image_ori)

        image = Image.open(image_ori).convert('RGB')
        transform = transforms.Compose([
          # transforms.Grayscale(num_output_channels=1),
            #transforms.RandomCrop(self.image_size),
            transforms.RandomResizedCrop(self.image_size),
            #transforms.RandomHorizontalFlip(),
            #transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
           # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        return transform(image)
        
    def __len__(self):
        return len(self.image_path)




class VimeoDatasets(Dataset):
    def __init__(self, data_dir, text_dir , image_size=256):
        self.text_dir  = text_dir # tri_trainlist.txt
        self.data_dir = data_dir #vimeo_arod/
        self.image_size = image_size
        pattern = ".png"
        self.image_path = [] #sorted(glob(os.path.join(self.data_dir, "*.*")))
        cn = 0
        self.total_dir = os.path.join(self.data_dir,"sequences") #data_dir/sequences
        
        file = open(os.path.join(self.data_dir,self.text_dir),"r")
        lines = file.readlines()
        
        for index, line in enumerate(lines):
            if index> 30001:
                break


            c = line.strip()
            tmp = os.path.join(self.data_dir,c)
            if os.path.isdir(tmp):
                d = [os.path.join(tmp,f) for f in os.listdir(tmp)]
                self.image_path += d
        
        file.close()
        self.image_path = self.image_path[:30000]
        
        
        """
        for path, _, files in os.walk(self.data_dir):
            for name in files:
                cn = cn + 1
                if ".png" in os.path.join(path, name):   
                    self.image_path.append(os.path.join(path, name))
                if cn%20000==0:
                    print(len(self.image_path))
        print(len(self.image_path)," lunghezza")
        """

        #self.image_path = sorted(glob(os.path.join(self.data_dir, "*.*")))
        
    def __getitem__(self, item):
        image_ori = self.image_path[item]

        #image = cv2.imread(image_ori)

        image = Image.open(image_ori).convert('RGB')
        transform = transforms.Compose([
          # transforms.Grayscale(num_output_channels=1),
            #transforms.RandomCrop(self.image_size),
            transforms.RandomResizedCrop(self.image_size),
            #transforms.RandomHorizontalFlip(),
            #transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
           # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        return transform(image)
        
    def __len__(self):
        return len(self.image_path)


def get_loader(train_data_dir, test_data_dir, image_size, batch_size):
    train_dataset = Datasets(train_data_dir, image_size)
    test_dataset = Datasets(test_data_dir, image_size)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)
    return train_loader, test_loader




class JPGAROD(Dataset):
    def __init__(self, data_dir, image_size=256):
        self.data_dir = data_dir
        self.image_size = image_size
        self.image_path = sorted(glob(os.path.join(self.data_dir, "*.*")))
        
    def __getitem__(self, item):
        image_ori = self.image_path[item]

        #image = cv2.imread(image_ori)

        image = Image.open(image_ori).convert('RGB')
        transform = transforms.Compose([
          # transforms.Grayscale(num_output_channels=1),
            #transforms.RandomCrop(self.image_size),
            transforms.RandomResizedCrop(self.image_size),
            #transforms.RandomHorizontalFlip(),
            #transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
           # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        return transform(image)
        
    def __len__(self):
        return len(self.image_path)



class Datasets(Dataset):
    def __init__(self, data_dir, image_size=256):
        self.data_dir = data_dir
        self.image_size = image_size
        self.image_path = sorted(glob(os.path.join(self.data_dir, "*.*")))
        
    def __getitem__(self, item):
        image_ori = self.image_path[item]

        image = cv2.imread(image_ori)

        image = Image.open(image_ori).convert('RGB')
        transform = transforms.Compose([
          # transforms.Grayscale(num_output_channels=1),
            #transforms.RandomCrop(self.image_size),
            transforms.RandomResizedCrop(self.image_size),
            #transforms.RandomHorizontalFlip(),
            #transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
           # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        return transform(image)
        
    def __len__(self):
        return len(self.image_path)


def get_loader(train_data_dir, test_data_dir, image_size, batch_size):
    train_dataset = Datasets(train_data_dir, image_size)
    test_dataset = Datasets(test_data_dir, image_size)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)
    return train_loader, test_loader


def get_train_loader(train_data_dir, image_size, batch_size):
    train_dataset = Datasets(train_data_dir, image_size)
    torch.manual_seed(3334)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True)
    return train_dataset, train_loader

class TestKodakDataset(Dataset):
    def __init__(self, data_dir, names = False):
        self.names = names
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            raise Exception(f"[!] {self.data_dir} not exitd")
        self.image_path = sorted(glob(os.path.join(self.data_dir, "*.*")))

    def __getitem__(self, item):
        image_ori = self.image_path[item]
        image = Image.open(image_ori).convert('RGB')
        transform = transforms.Compose([
          
            transforms.Resize((256,256)),
            transforms.ToTensor(),
        ])
        if self.names:
            return transform(image), image_ori.split("/")[-1].split(".")[0]
        else:
            return transform(image)

    def __len__(self):
        return len(self.image_path)

def build_dataset():
    train_set_dir = '/data1/liujiaheng/data/compression/Flick_patch/'
    dataset, dataloader = get_train_loader(train_set_dir, 256, 4)
    for batch_idx, (image, path) in enumerate(dataloader):
        pdb.set_trace()
