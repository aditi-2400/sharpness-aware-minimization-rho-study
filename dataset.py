import os, numpy as np, torch
from typing import Tuple, Dict, List
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

def get_cifar100_loaders(data_dir: str, batch_size: int = 128, num_workers: int = 4, use_aug: bool = True) -> Tuple[DataLoader, DataLoader]:
    """Returns train and test dataloaders for CIFAR-100."""
    norm = transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), norm
    ]) if use_aug else transforms.Compose([transforms.ToTensor(), norm])
    test_tf = transforms.Compose([transforms.ToTensor(), norm])

    train_ds = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=train_tf)
    test_ds = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader

class CIFAR100C(Dataset):
    """CIFAR-100-C style dataset."""

    CORRUPTIONS: List[str] = [
        "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
        "snow", "frost", "fog", "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression",
    ]

    def __init__(self, root: str, corruption: str, severity: int = 1, transform=None):
        assert corruption in self.CORRUPTIONS, f"Unknown corruption: {corruption}"
        assert 1 <= severity <= 5, "severity must be in [1,5]"
        self.root, self.corruption, self.severity, self.transform = root, corruption, severity, transform

        imgs_path, labels_path = os.path.join(root, f"{corruption}.npy"), os.path.join(root, "labels.npy")
        self.images, self.labels = np.load(imgs_path), np.load(labels_path)
        n, start, end = 50000, (severity - 1) * 50000, severity * 50000
        self.images = self.images[start:end]
        assert len(self.images) == len(self.labels)

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        img, label = torch.from_numpy(self.images[idx]).permute(2, 0, 1).float() / 255.0, int(self.labels[idx])
        if self.transform: img = self.transform(img)
        return img, label


def get_cifar100c_loaders(data_dir: str, batch_size: int = 128, num_workers: int = 4, severities: List[int] = [1, 2, 3, 4, 5]) -> Dict[str, List[DataLoader]]:
    """Returns corruption_name -> list of dataloaders per severity."""
    norm = transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
    dl_dict: Dict[str, List[DataLoader]] = {}
    for corr in CIFAR100C.CORRUPTIONS:
        dl_dict[corr] = [
            DataLoader(CIFAR100C(root=data_dir, corruption=corr, severity=s, transform=norm),
                       batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
            for s in severities
        ]
    return dl_dict

