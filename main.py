"""
Trains a PyTorch image classification model using device-agnostic code.
"""
import argparse
import os
import torch
import data_setup, engine, utils, gen_model
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torchvision import datasets, transforms

def ddp_setup():
    init_process_group(backend="nccl")
    
def main (epochs: int,
          batch_size: int,
          lr: float,
          save_every: int,
          snapshot_path: str,
          fully_trained_model: str):
    
    # Function to set the seed for reproducibility (default seed = 42)
    utils.set_seed()
    
    # Setup target device
    ddp_setup()
    
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
    
    classes = combine_dataset.classes
    
    print(combine_dataset.combined_dataset)
    
    nbr_classes = len(classes)
    
    training_dataloader = combine_dataset.get_train_dataloader()
    test_dataloader = combine_dataset.get_test_dataloader()
    
    writer = utils.create_writer("Run_batch",
                                 "vit_small_patch14_reg4_dinov2",
                                 f"lr-{lr}_epoch-{epochs}_batch-{batch_size}_light_dataset")
    
    loss_fn = torch.nn.CrossEntropyLoss()
    
    model_creator = gen_model.vit_small_eva02
    model = model_creator.creat_model(nbr_classes=nbr_classes)
    

    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    
    trainer = engine.Trainer(model=model,
                             train_data=training_dataloader,
                             test_data=test_dataloader,
                             optimizer=optimizer,
                             save_every=save_every,
                             snapshot_path=snapshot_path,
                             loss_fn=loss_fn,
                             scheduler=scheduler,
                             writer=writer)
    
    trainer.train(max_epochs=epochs)
    destroy_process_group()

    # Save the model with help from utils.py
    utils.save_model(model=model,
                    target_dir="models",
                    model_name=fully_trained_model)
    
if __name__ == "__main__":
    import sys
    parser = argparse.ArgumentParser(description="Simple example of training script using EVA02.")
    parser.add_argument("-e", "--epochs", required=False, type=int, default=50, help="The number of training Epochs")
    parser.add_argument("--batch", required=False, type=int, default=48, help="The size of the batch")
    parser.add_argument("--lr", required=False, type=float, default=8.0e-05, help="The learning rate used for the training")
    parser.add_argument("--save_every", required=False, type=int, default=2, help="How often the model is saved per epochs during the trainning")
    parser.add_argument("--snapshot_path", required=False, type=str, default="models/snapshot/snapshot.pt", help="File location of the intermadiate saved model")
    parser.add_argument("--fully_trained_model", required=False, type=str, default="Small_EVA02_trained_Vx.pth")
    args = parser.parse_args()
    
    main(epochs=args.epochs,
         batch_size=args.batch,
         lr=args.lr,
         save_every=args.save_every,
         snapshot_path=args.snapshot_path,
         fully_trained_model=args.fully_trained_model)