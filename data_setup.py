import torch
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import ConcatDataset


def Combine_datasets(transform, list_dataset:str):
    datasets_list = []

    for folder_list in list_dataset:
        if os.path.isdir(folder_list):
            print(folder_list)
            dataset = datasets.ImageFolder(root=folder_list, transform=transform)
            print(folder_list)
            img, label = dataset[0][0], dataset[0][1]
            print(f"Image tensor:\n{img}")
            print(f"Image shape: {img.shape}")
            print(f"Image datatype: {img.dtype}")
            print(f"Image label: {label}")
            print(f"Label datatype: {type(label)}")
            datasets_list.append(dataset)
        
    return datasets_list

# def Combine_datasets(root:str, transform, list_dataset):
#     datasets_list = []

#     for root_folder in os.listdir(root):
#         root_path = os.path.join(root, root_folder)
#         print(root_path)
#         for folder_name in os.listdir(root_path):
#             if folder_name in list_dataset:
#                 folder_path = os.path.join(root_path, folder_name)
#                 if os.path.isdir(folder_path):
#                     print(folder_path)
#                     dataset = datasets.ImageFolder(root=folder_path, transform=transform)
#                     datasets_list.append(dataset)
        
#     return datasets_list

class CustomConcatDataset(Dataset):
    def __init__(self, datasets, batch:int=64):
        TRAIN_PERCENT = 0.8
        self.datasets = datasets
        self.batch = batch
        self.cumulative_sizes = [0] + [len(d) for d in self.datasets]
        self.class_to_index_map = self._create_class_to_index_map()
        self.classes = self._get_classes()
        self.combined_dataset = ConcatDataset(self.datasets)
        self.train_size = int(TRAIN_PERCENT * len(self.combined_dataset))
        self.test_size = int(len(self.combined_dataset) - self.train_size)
        self.train_data, self.test_data = torch.utils.data.random_split(self.combined_dataset,
                                                                        [self.train_size, self.test_size])
         
        
    def _create_class_to_index_map(self):
        class_to_index_map = {}
        current_index = 0
        for dataset in self.datasets:
            classes = dataset.classes
            for class_name in classes:
                class_to_index_map[class_name] = current_index
                current_index += 1
        return sorted(class_to_index_map.items(), key=lambda x:x[1])
    
    def _get_classes(self):
        all_classes = []
        for dataset in self.datasets:
            all_classes.extend(dataset.classes)
        return sorted(set(all_classes))
    
    def __len__(self):
        return sum(len(d) for d in self.datasets)
    
    def __getitem__(self, index):
        dataset_index = 0
        while index >= self.cumulative_sizes[dataset_index + 1]:
            dataset_index += 1
        return self.datasets[dataset_index][index - self.cumulative_sizes[dataset_index]]
    
    def class_to_index(self, class_name):
        return self.class_to_index_map[class_name]
    
    def get_train_dataloader(self):
        return DataLoader(self.train_data,
                          batch_size=self.batch,
                          shuffle=False,
                          num_workers=1,
                          pin_memory=True,
                          sampler=DistributedSampler(self.train_data)
                          )
    
    def get_test_dataloader(self):
        return DataLoader(self.test_data,
                          batch_size=self.batch,
                          shuffle=False,
                          num_workers=1,
                          pin_memory=True,
                          sampler=DistributedSampler(self.test_data)
                          )