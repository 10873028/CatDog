import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import CatDog
from efficientnet_pytorch import EfficientNet
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32

TRANSFORM = A.Compose(
    [
        A.Resize(height=448, width=448),
        A.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0
        ),
        ToTensorV2(),
    ]
)


def save_features(model, loader, filename, output_size=(1, 1)):
    model.eval()
    images = []
    labels = []
    for x, y in loader:
        x = x.to(DEVICE)
        with torch.no_grad():
            features = model.extract_features(x)
            features = F.adaptive_avg_pool2d(features, output_size=output_size)
        images.append(features.reshape(x.shape[0], -1).detach().cpu().numpy())
        labels.append(y.numpy())

    np.save(f"x_{filename}.npy", np.concatenate(images, axis=0))
    np.save(f"y_{filename}.npy", np.concatenate(labels, axis=0))
    model.train()


def main():
    model = EfficientNet.from_pretrained("efficientnet-b7")
    trainDataset = CatDog(root="CATDOG/train", transform=TRANSFORM)
    testDataset = CatDog(root="CATDOG/test1", transform=TRANSFORM)
    trainLoader = DataLoader(trainDataset, shuffle=True, batch_size=BATCH_SIZE)
    testLoader = DataLoader(testDataset, shuffle=False, batch_size=BATCH_SIZE)
    model = model.to(DEVICE)
    save_features(model, trainLoader, filename="train", output_size=(1, 1))
    save_features(model, testLoader, filename="test", output_size=(1, 1))


if __name__ == "__main__":
    main()
