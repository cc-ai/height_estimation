import os.path
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import random

def default_loader(path):
    return Image.open(path).convert("RGB")

def array_loader(path):
    """
    loader to be used for ground truth height numpy arrays
    """
    return(torch.from_numpy(np.load(path)))
    
def default_flist_reader(flist):
    """
    file list reader
    """
    imlist = []
    with open(flist, "r") as rf:
        for line in rf.readlines():
            impath = line.strip()
            imlist.append(impath)
    return imlist

class ImageFilelist(data.Dataset):
    def __init__(
        self,
        root,
        flist,
        transform=None,
        flist_reader=default_flist_reader,
        loader=default_loader,
    ):
        self.root = root
        self.imlist = flist_reader(flist)
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        impath = self.imlist[index]
        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.imlist)

class HeightDataset(data.Dataset):
    def __init__(
        self,
        img_root,
        gt_root,
        flist,
        gtlist, 
        flist_reader=default_flist_reader,
        floader=default_loader,
        gtloader=array_loader,
        transform_params = None
    ):
        self.img_root = img_root
        self.gt_root = gt_root
        self.imlist = flist_reader(flist)
        self.gtlist = flist_reader(gtlist)
        self.floader = floader
        self.gtloader = gtloader
        self.resize_size = None
        self.crop_size = None
        if not transform_params is None : 
            self.resize_size = transform_params['resize_size']
            self.crop_size = transform_params['crop_size']      
        
    
    def __getitem__(self, index):
        impath = self.imlist[index]
        gtpath = self.gtlist[index]
        img = self.floader(os.path.join(self.img_root, impath))
        gt = self.gtloader(os.path.join(self.gt_root, gtpath)).unsqueeze(0).unsqueeze(0)
        if not self.transform is None:
            return(self.transform(img, gt))
        return {'image': transform.ToTensor()(img), 'mask' : gt}
    
    def __len__(self):
        return (len(self.imlist))
    
    def set_transform(self, transform):
        """This method will replace the current transformation for the
        dataset.

        :param transform: the new transformation
        """
        self.transform = transform
        
    def get_rcrop_params(self, img):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (tensor): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        h, w = img.shape[1:]
        th, tw =  self.crop_size
        if w == tw and h == th:
            return (0, 0, h, w)
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return (i, j, th, tw)
    
    def transform(self, image, mask):

        image_ = transforms.ColorJitter(brightness=0.5, contrast=0.5)(image)
        img = transforms.ToTensor()(image_).unsqueeze(0)
        
        if not self.resize_size is None : 
            # Resize
  
            img =  F.interpolate(img, size = (self.resize_size[0], self.resize_size[1]) , mode = "nearest")

            # h and w are swapped for landmarks because for images,
            # x and y axes are axis 1 and 0 respectively
            mask = F.interpolate(mask, size = (self.resize_size[0], self.resize_size[1]), mode = "nearest")
            img = img.squeeze(0)
            mask = mask.squeeze(0)
        if not self.crop_size is None :
            # Random crop
            i, j, h, w = self.get_rcrop_params(img)
            img = img[:,i:i+h, j:j+w]
            mask = mask[:,i:i+h, j:j+w]
            #print(img.shape)
            
        # Random vertical flipping
        if random.random() > 0.5:
            img = torch.flip(img, (2, ))
            mask = torch.flip(mask, (2, ))
            #print(img.shape)
        return {'image': img, 'mask' : mask}#transforms.ToPILImage()(img.squeeze(0)), 'mask' : mask}
    
class HeightDataset_fromlist(data.Dataset):
    def __init__(
        self,
        flist,
        gtlist, 
        flist_reader=default_flist_reader,
        floader=default_loader,
        gtloader=array_loader,
        transform_params = None
    ):
        self.imlist = flist_reader(flist)
        self.gtlist = flist_reader(gtlist)
        self.floader = floader
        self.gtloader = gtloader
        self.resize_size = None
        self.crop_size = None
        if not transform_params is None : 
            self.resize_size = transform_params['resize_size']
            self.crop_size = transform_params['crop_size']      
        
    
    def __getitem__(self, index):
        impath = self.imlist[index]
        gtpath = self.gtlist[index]
        img = self.floader(impath)
        gt = self.gtloader(gtpath).unsqueeze(0)
        if not self.transform is None:
            return(self.transform(img, gt))
        return {'image': transform.ToTensor()(img), 'mask' : gt}
    
    def __len__(self):
        return (len(self.imlist))
    
    def set_transform(self, transform):
        """This method will replace the current transformation for the
        dataset.

        :param transform: the new transformation
        """
        self.transform = transform
        
    def get_rcrop_params(self, img):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (tensor): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        h, w = img.shape[1:]
        th, tw =  self.crop_size
        if w == tw and h == th:
            return (0, 0, h, w)
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return (i, j, th, tw)
    
    def transform(self, image, mask):

        image_ = transforms.ColorJitter(brightness=0.5, contrast=0.5)(image)
        img = transforms.ToTensor()(image_)#.unsqueeze(0)
        #print(img.shape)
        if not self.resize_size is None : 
            # Resize
  
            img =  F.interpolate(img, size = self.resize_size, mode = "nearest")

            # h and w are swapped for landmarks because for images,
            # x and y axes are axis 1 and 0 respectively
            mask = F.interpolate(mask, size = self.resize_size, mode = "nearest")
        
        if not self.crop_size is None :
            # Random crop
            i, j, h, w = self.get_rcrop_params(img)
            img = img[:, i:i+h, j:j+w]
            mask = mask[ :,i:i+h, j:j+w]
            #print(img.shape)
            
        # Random vertical flipping
        if random.random() > 0.5:
            img = torch.flip(img, (2, ))
            mask = torch.flip(mask, (2, ))
            #print(img.shape)
        return {'image': img, 'mask' : mask}
         
def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), "%s is not a valid directory" % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images