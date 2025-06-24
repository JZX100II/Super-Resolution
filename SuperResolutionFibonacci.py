import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

def add_coord_encoding(xy): 
    frequencies = [-13, -8, -5, -3, -2, -1, -1, 1, 1, 2, 3, 5, 8, 13]
    
    embeddings = [xy]
    for freq in frequencies:
        for fn in [torch.sin, torch.cos]:
            embeddings.append(fn(xy * np.pi * freq))
    return torch.cat(embeddings, dim=-1)

class NeuralTexture(nn.Module): 
    def __init__(self, input_dim=2, hidden_dim=128, output_dim=3, layers=4):
        super().__init__()
        net = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(layers - 1):
            net += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        net += [nn.Linear(hidden_dim, output_dim), nn.Sigmoid()]  # Sigmoid to output [0, 1] for RGB
        self.model = nn.Sequential(*net)

    def forward(self, x):
        return self.model(x)

def get_coords(w, h):
    xs = torch.linspace(-1, 1, steps=w)
    ys = torch.linspace(-1, 1, steps=h)
    grid = torch.stack(torch.meshgrid(xs, ys, indexing='xy'), dim=-1)  # (W, H, 2)
    return grid.reshape(-1, 2)  # (W*H, 2)

def train_texture_model(img_tensor, epochs=500, lr=1e-3):
    H, W = img_tensor.shape[:2]
    coords = get_coords(W, H)  # (N, 2)
    coords = add_coord_encoding(coords)
    pixels = img_tensor.reshape(-1, 3)

    model = NeuralTexture(input_dim=coords.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        pred = model(coords)
        loss = loss_fn(pred, pixels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.6f}")

    return model

def render_texture(model, res=256):
    coords = get_coords(res, res)
    coords_encoded = add_coord_encoding(coords)
    with torch.no_grad():
        preds = model(coords_encoded)
    return preds.reshape(res, res, 3).cpu().numpy()

if __name__ == "__main__":
    from PIL import Image
    import torchvision.transforms as T

    ds = CIFAR10(root="data", train=True, download=True, transform=T.ToTensor())
    random_index = random.randint(0, 8000)
    img, _ = ds[random_index]
    img = img.permute(1, 2, 0) # H, W, C: Height, Width, Color

    model = train_texture_model(img, epochs=2000)

    base_img, _ = ds[random_index]
    base_img = base_img.permute(1, 2, 0)
    plt.imshow(base_img)
    plt.title(f"Base Image {random_index}")
    plt.axis('off')
    plt.show()

    out_img = render_texture(model, res=2048) # Super Resolution
    plt.imshow(out_img)
    plt.title("Neural Texture Output")
    plt.axis('off')
    plt.show()