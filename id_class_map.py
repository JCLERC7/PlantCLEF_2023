import argparse
import os
import torch
import pandas as pd
import data_setup, engine, utils, gen_model
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torchvision import datasets, transforms

list_dataset = ["./data/PlantCLEF2022_Training/web/images",
                "../PlantCLEF_2024/data/Training_data/dataset"]

transform = transforms.Compose([
	# Resize the images to 224x224
	transforms.Resize(size=(224,224)),
	# Flip the images randomly on the horizontal
	transforms.RandomHorizontalFlip(p=0.5),  # p = probability of flip, 0.5 = 50% chance
	# Add your custom augmentation here
	transforms.RandomApply([transforms.TrivialAugmentWide(num_magnitude_bins=31)], p=0.5),
	# Turn the image into a torch.Tensor
	transforms.ToTensor(),
	# Normalize the image
	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

datasets_list = data_setup.Combine_datasets(transform=transform, list_dataset=list_dataset)
    
combine_dataset = data_setup.CustomConcatDataset(datasets_list)

df = pd.DataFrame(combine_dataset.class_to_index_map, columns=['class_name', 'class_id'])

output_csv = 'class_map.csv'
df.to_csv(output_csv, index=False, sep=";")

print(f'Class-ID mapping saved to {output_csv}')