import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------- #
#    ResNet-18 Model Architecture for CIFAR-100
#    As specified in Proposal Section 3.1
# ---------------------------------------------------------------------------- #

class BasicBlock(nn.Module):
    """
    Defines the fundamental ResNet "BasicBlock" (used in ResNet-18/34).
    This block consists of two 3x3 convolutions and a "skip connection".
    """
    # Class variable: states that the output channels are the same as input channels.
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        """
        Initializes the block's layers.
        - in_planes: number of input channels
        - planes: number of output channels (for the conv layers)
        - stride: stride of the first convolution, used for downsampling
        - downsample: a function/layer to process the skip connection if dimensions change
        """
        super(BasicBlock, self).__init__()
        # First convolutional layer (3x3 kernel)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # Second convolutional layer (3x3 kernel)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # The 'downsample' layer is for the skip connection, to match dimensions
        # if the stride is not 1 (i.e., spatial downsampling) or if channels increase.
        self.downsample = downsample

    def forward(self, x):
        """
        Defines the forward pass of the block.
        """
        # Store the input for the skip connection
        identity = x

        # Path 1: Main convolutional path
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Path 2: Skip connection path
        if self.downsample is not None:
            identity = self.downsample(x) # Apply downsampling (if needed)

        # Add the skip connection to the main path's output
        # This is the core "residual" idea.
        out += identity
        out = F.relu(out) # Final activation
        return out


class ResNet(nn.Module):
    """
    Assembles the BasicBlocks into the full ResNet architecture.
    """
    def __init__(self, block, num_blocks, num_classes=100):
        """
        Initializes the full network architecture.
        - block: The block type to use (i.e., BasicBlock)
        - num_blocks: A list of integers specifying how many blocks per stage (e.g., [2, 2, 2, 2])
        - num_classes: Number of output classes (100 for CIFAR-100)
        """
        super(ResNet, self).__init__()
        # 'in_planes' is a helper to track input channels for the next layer
        self.in_planes = 64

        # --- CIFAR-100 Modification (per proposal) ---
        # Standard ResNet uses a 7x7 kernel and maxpool here, designed for 224x224 ImageNet images.
        # For 32x32 CIFAR images, we use a single 3x3 conv to preserve dimensions.
        # Input: (B, 3, 32, 32)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # Output: (B, 64, 32, 32)
        self.bn1 = nn.BatchNorm2d(64)
        
        # --- ResNet Stages ---
        # Stage 1: No spatial change
        # Input: (B, 64, 32, 32)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        # Output: (B, 64, 32, 32)
        
        # Stage 2: Halves height/width, doubles channels
        # Input: (B, 64, 32, 32)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        # Output: (B, 128, 16, 16)
        
        # Stage 3: Halves height/width, doubles channels
        # Input: (B, 128, 16, 16)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        # Output: (B, 256, 8, 8)
        
        # Stage 4: Halves height/width, doubles channels
        # Input: (B, 256, 8, 8)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # Output: (B, 512, 4, 4)
        
        # --- Classifier Head ---
        # Global Average Pooling: Takes the 4x4 feature map and averages it to 1x1
        # Input: (B, 512, 4, 4)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Output: (B, 512, 1, 1)
        
        # Final Fully Connected (Linear) Layer
        # Input: (B, 512) (after flattening)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        # Output: (B, 100)

    def _make_layer(self, block, planes, num_blocks, stride):
        """
        Helper function to create a "stage" (a series of blocks).
        """
        # The first block in a stage might have a stride of 2 (to downsample)
        # and will need a 'downsample' layer for its skip connection.
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        # Add the first block (which handles the downsampling)
        layers.append(block(self.in_planes, planes, stride, downsample))
        
        # Update in_planes for the subsequent blocks
        self.in_planes = planes * block.expansion
        
        # Add the remaining blocks (all with stride=1)
        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Defines the forward pass for the entire network.
        Input x: (B, 3, 32, 32)
        """
        # Initial conv layer
        out = F.relu(self.bn1(self.conv1(x))) # (B, 64, 32, 32)
        
        # ResNet stages
        out = self.layer1(out) # (B, 64, 32, 32)
        out = self.layer2(out) # (B, 128, 16, 16)
        out = self.layer3(out) # (B, 256, 8, 8)
        out = self.layer4(out) # (B, 512, 4, 4)
        
        # Classifier head
        out = self.avgpool(out)    # (B, 512, 1, 1)
        out = torch.flatten(out, 1) # (B, 512)
        out = self.linear(out)      # (B, 100)
        
        return out


def resnet18_cifar():
    """
    A "factory" function to easily instantiate the ResNet-18 model
    configured for CIFAR-100.
    
    ResNet-18 is defined by using [2, 2, 2, 2] blocks.
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=100)


# This block only runs if you execute this file directly (e.g., `python model.py`)
# It's a "sanity check" to make sure the model can be built and run.
if __name__ == '__main__':
    print("--- Testing ResNet-18 for CIFAR-100 ---")
    
    # 1. Create an instance of the model
    model = resnet18_cifar()
    print(f"Model: {model.__class__.__name__}")
    
    # 2. Create a dummy input tensor
    # (Batch size 2, 3 color channels, 32x32 pixels)
    sample_input = torch.randn(2, 3, 32, 32)
    print(f"Input shape:  {sample_input.shape}")
    
    # 3. Perform a forward pass
    output = model(sample_input)
    
    # 4. Check the output shape
    # Should be (Batch size 2, 100 classes)
    print(f"Output shape: {output.shape}")
    assert output.shape == (2, 100)
    
    print("--- Model test passed. ---")