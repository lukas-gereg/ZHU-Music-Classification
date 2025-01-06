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
        self.model = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)

        # Fine-tune layers instead of freezing them
        # for param in self.model.parameters():
        #     param.requires_grad = True

        # Modifying pretrained ResNet18
        # self.model.conv1 = nn.Conv2d(in_channels=self.color_channels,
        #                              out_channels=64,
        #                              kernel_size=7,
        #                              stride=2,
        #                              padding=3
        #                             )
        
        # self.model.fc = nn.Linear(self.model.fc.in_features, out_features=self.num_classes) # modifying last layer of ResNet (input from previous layer, output 10 classes(genres))
        fc_input = self.model.fc.in_features

        self.model.fc = nn.Identity()

        self.drop = nn.Dropout(p=0.2)

        self.fc1 = nn.Sequential(
            nn.Linear(fc_input, 128),
            nn.LeakyReLU(inplace=True)
        )

        # self.fc2 = nn.Sequential(
        #     nn.Linear(self.fc1.get_submodule('0').out_features, 256),
        #     nn.LeakyReLU(inplace=True)
        # )

        self.fc3 = nn.Sequential(
            nn.Linear(self.fc1.get_submodule('0').out_features, self.num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Models forward function
        """

        x = self.model(x)

        x = self.drop(x)

        x = self.fc1(x)
        # x = self.fc2(x)
        x = self.fc3(x)

        return x
