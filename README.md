# Pokemon-Classifier

### Overview
This project uses a ResNet-18 image classification pipeline.  
The goal is to classify images of pokémons into various classes/species (e.g Bulbasaur).

### Project structure

├── README.md              
├── requirements.txt               
├── utils.py              
├── train.py              
├── visual.ipynb        
├── silly.py              
├── augmentation_results.ipynb              
├── model_testing_results.ipynb   
├── pokemon_resnet18.pth
└── eda.ipynb 

- ``requirements.txt``: Required packages to run this project.  
- ``utils.py``: Utilities - dataset, augmentation, and model for classification.  
- ``train.py``: Training script with PyTorch which imports model and dataloaders from ``utils.py``. Also evaluates the model.  
- ``visual.ipynb``: Notebook for visualization.  
- ``silly.py``: Script for single image inference (where we tested images of ourselves to see which pokémon we looked like xD).  
- ``augmentation_results.ipynb``: Notebook summarizing the results from testing various transforms.  
- ``model_testing_results.ipynb``: Notebook summarizing the results from testing various models.  
- ``pokemon_resnet18.pth``: The saved model.  
- ``eda.ipynb``: Exploratory Data Analysis to understand dataset structure, attribute distributions, and class imbalances.

### Dataset
We used [this](https://www.kaggle.com/datasets/noodulz/pokemon-dataset-1000/data) dataset from Kaggle.  
It consists of more than 26000 images of 1000 pokémons, split into test, train, and validation, including a csv file with metadata.

### Results
After experimenting with various transforms we decided to go with these (using the Albumentations library):
- **HueSaturationValue** 
- **RandomBrightnessContrast**
- **GaussNoise** 
- **ColorJitter**

We also experimented with different models and decided to go with a pre-trained ResNet18 model:

    class PokemonResNet(nn.Module):
        def __init__(self, num_classes):
            super(PokemonResNet, self).__init__()
            self.base_model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
            in_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Linear(in_features, num_classes)

        def forward(self, x):
            x = self.base_model(x)
            return x

With these augmentations and this model we were able to achieve a testing accuracy of **90.52%**.