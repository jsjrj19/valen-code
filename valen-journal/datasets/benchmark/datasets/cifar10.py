from PIL import Image
import os
import os.path
import sys
import torch
import numpy as np
import pickle 
import torch.utils.data as data
from copy import deepcopy
from utils.utils_algo import binarize_class, partialize, check_integrity, download_url

class cifar10(data.Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    base_folder = 'cifar-10-batches-py' 
    url = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    filename = 'cifar-10-python.tar.gz'
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a' 
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]
    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]


    def __init__(self, root, train_or_not=True, download=False, transform=None, target_transform=None,
                 partial_type='binomial', partial_rate=0.1, random_state=0):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train_or_not
        self.dataset = 'cifar10'
        self.partial_type = partial_type

        if download:
            self.download()
        #检查数据集完整性
        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')
        #如果是训练集
        if self.train:
            self.train_data = []
            self.train_labels = []
            #遍历训练集的所有批次文件
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(self.root, self.base_folder, f)
                fo = open(file, 'rb')
                entry = pickle.load(fo, encoding='latin1') 
                self.train_data.append(entry['data'])
                self.train_labels += entry['labels']
                fo.close()
            #合并所有批次的数据
            self.train_data = np.concatenate(self.train_data) 
            #重塑数据形状
            self.train_data = self.train_data.reshape((50000, 3, 32, 32))
            #转置维度
            self.train_data = self.train_data.transpose((0, 2, 3, 1)) 
            #转换为pytorch张量
            self.train_data = torch.from_numpy(self.train_data)
            self.train_labels = torch.tensor(self.train_labels, dtype=torch.long)
            #如果部分标签不为0，处理部分标签
            if partial_rate != 0.0:
                #标签二值化
                y = binarize_class(self.train_labels)
                #部分化标签
                self.train_final_labels, self.average_class_label = partialize(y, self.train_labels, partial_type, partial_rate)    

            else:
                #如果不使用部分标签，直接使用二值化标签
                self.train_final_labels = binarize_class(self.train_labels).float()
            #深拷贝标签分布，用于后续处理
            self.train_label_distribution = deepcopy(self.train_final_labels)

        else:
            f = self.test_list[0][0]#获取测试集文件名
            file = os.path.join(self.root, self.base_folder, f)#构建完整文件路径
            fo = open(file, 'rb')#以二进制读模式打开文件

            #根据python版本选择不同的加载方式
            if sys.version_info[0] == 2: 
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.test_data = entry['data']
            self.test_labels = entry['labels']
            fo.close()
            self.test_data = self.test_data.reshape((10000, 3, 32, 32))
            self.test_data = self.test_data.transpose((0, 2, 3, 1)) 
            
            self.test_data = torch.from_numpy(self.test_data)
            self.test_labels = torch.tensor(self.test_labels, dtype=torch.long)
            

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        #根据是训练集还是测试集获取对应的数据
        if self.train:
            img, target, true, distr = self.train_data[index], self.train_final_labels[index], self.train_labels[index], self.train_label_distribution[index]
        else:
            img, target, true, distr = self.test_data[index], self.test_labels[index], self.test_labels[index], self.test_labels[index]
        #将张量转换为PIL图像
        img = Image.fromarray(img.numpy(), mode=None) 
        #应用图像变换
        if self.transform is not None:
            img = self.transform(img)
        #应用标签变换
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, true, distr, index


    def __len__(self):
        #返回数据集的长度
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)


    def _check_integrity(self):
        #检查数据集文件的完整性
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5): 
                return False
        return True
        

    def download(self):
        import tarfile 

        if self._check_integrity():
            return

        #下载数据集文件
        download_url(self.url, self.root, self.filename, self.tgz_md5) 

        cwd = os.getcwd()#保存当前工作目录 
        tar = tarfile.open(os.path.join(self.root, self.filename), "r:gz")#打开下载的tar.gz文件
        os.chdir(self.root)#切换到根目录
        tar.extractall()#解压所有文件
        tar.close()#关闭tar文件
        os.chdir(cwd)#切换回原工作目录


    def __repr__(self):
        #返回数据集的字符串表示
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
