import torch
import torch.nn as nn
import torchvision 
# from torchvision.models import ResNet18_Weights
from torchvision.models import resnet50
from torchvision.models import ResNet50_Weights
from models.base_model import BaseModel


# temporary parameters
#model_properties = {'color_channels': 3, 'num_classes': 10, 'image_size': 224}


class ResNetMusic(BaseModel):
    """
    Custom model ResNetMusic, using pretrained model ResNet18. 
    """
    def __init__(self, model_properties: dict) -> None:
        super().__init__(model_properties)
        
        # Validating model_properties
        assert 'color_channels' in model_properties, "Missing 'color_channels' in model_properties"
        assert 'num_classes' in model_properties, "Missing 'num_classes' in model_properties"
        assert 'image_size' in model_properties, "Missing 'image_size' in model_properties"

        # Setting model properties
        self.color_channels = model_properties['color_channels']
        self.num_classes = model_properties['num_classes']
        self.image_size = model_properties['image_size']

        # Initializing the model itself
        self.init_model()


    def init_model(self) -> None:
        """
        Initializes model
        """

        # self.model = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT) # Loading pretrained ResNet18 with the latest available weights
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)

        # Fine-tune layers instead of freezing them
        for param in self.model.parameters():
            param.requires_grad = True

        # Modifying pretrained ResNet18
        self.model.conv1 = nn.Conv2d(in_channels=self.color_channels, 
                                     out_channels=64, 
                                     kernel_size=7, 
                                     stride=2, 
                                     padding=3
                                    )
        
        # self.model.fc = nn.Linear(self.model.fc.in_features, out_features=self.num_classes) # modifying last layer of ResNet (input from previous layer, output 10 classes(genres))
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),  # Add dropout with 50% rate
            nn.Linear(self.model.fc.in_features, out_features=self.num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Models forward function
        """
        return self.model(x)


