import torch
from models.model import ExampleCNN
from datasets.dataloader import make_test_dataloader

import os
from tqdm import tqdm

import argparse

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# data path and weight path
base_path = os.path.dirname(os.path.abspath(__file__))
train_data_path = os.path.join(base_path, "data", "train")
test_data_path = os.path.join(base_path, "data", "test")
weight_path = os.path.join(base_path, "weights", "weight.pth")
img_path = os.path.join(base_path, "result")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Set epochs and learning rate')
    parser.add_argument('--img_path', type=str, default=img_path)
    parser.add_argument('--weight_path', type=str, default=img_path)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    img_path = args.img_path
    weight_path = args.weight_path
    batch_size = args.batch_size


txt_path = os.path.join(img_path, "acc.txt")
# load model and use weights we saved before
model = ExampleCNN()
model.load_state_dict(torch.load(weight_path))
model = model.to(device)

# make dataloader for test data
test_loader = make_test_dataloader(test_data_path, batch_size)

predict_correct = 0
model.eval()
with torch.no_grad():
    for data, target in tqdm(test_loader, desc="Testing"):
        data, target = data.to(device), target.to(device)

        output = model(data)
        predict_correct += (output.data.max(1)[1] == target.data).sum()
        
    accuracy = 100. * predict_correct / len(test_loader.dataset)
print(f'Test accuracy: {accuracy:.4f}%')
with open(txt_path, 'w') as file:
    file.write(f'Test accuracy: {accuracy:.4f}%\n')