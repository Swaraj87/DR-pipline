#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import sys
import json
import pickle
import copy
import numpy as np
import optuna
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# In[3]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F


# In[4]:


import xgboost as xgb
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report,
                           roc_auc_score, cohen_kappa_score)
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image


# In[5]:


class Config:
    def __init__(self):
        #dataset pathways
        self.data_dir = Path('aptos2019-blindness-detection')
        self.train_csv = self.data_dir/'train.csv'
        self.train_images_dir = self.data_dir/'train_images'
        self.test_images_dir = self.data_dir/'test_images'

        #Model path
        self.model_dir = Path('savemodels')
        self.model_features = Path('extracted_features')
        self.model_results = Path('results')

        #Create Directories 
        for dir_path in [self.model_dir, self.model_features, self.model_results]:
            dir_path.mkdir(parents=True, exist_ok=True)

        self.results_dir = Path('results')
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Hyperparameters
        self.batch_size = 32
        self.num_epochs = 5
        self.learning_rate = 5e-5
        self.num_classes = 5 #(0-4 severity)
        self.img_size = (512, 512) # for highlightling retina images

        # Unfreezing strategy
        self.unfreeze_blocks = {
            'resnet50': ['layer4', 'fc'],
            'inception_v3': ['Mixed_7c', 'Mixed_7b', 'Mixed_7a', 'fc'],
            'densenet121': ['denseblock4', 'classifier']
        }

        # Models
        self.pretrained_models = {
            'resnet50': models.resnet50,
            'densenet121': models.densenet121,
            'inceptionV3': models.inception_v3
        }

        # XGBoost parameters
        self.xgb_params = {
            'objective': 'multi:softmax',
            'num_class': 5,
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }

        self.n_splits = 5
        self.random_seed = 42

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def to_dict(self):
        return {k:v for k,v in self.__dict__.items() if not k.startswith('_')}


# In[6]:


class DataAugmentation:
    "performing Data augmentation for our images"

    @staticmethod
    def get_train_transform():

        return transforms.Compose([
            transforms.Resize((512,512)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1,0.1), scale=(0.9,1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    @staticmethod
    def get_val_transform():
        return transforms.Compose([
            transforms.Resize((512,512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    @staticmethod
    def preprocess_image(image_path):
        transfrom = transforms.Compose([
            transforms.Resize((512,512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = Image.open(image_path).convert('RGB')
        return transfrom(image).unsqueeze(0)


# In[7]:


class CustomAptos(Dataset):
    #Customizing the AptosDataset for our use
    def __init__(self, dataframe, image_dir, transform = None, is_test = False):

        self.dataframe = dataframe
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx]['id_code']
    
        # Handle different image extensions
        image_paths = [
            self.image_dir / f"{img_name}.png",
            self.image_dir / f"{img_name}.jpg",
            self.image_dir / f"{img_name}.jpeg"
        ]

        image_path = next((p for p in image_paths if p.exists()), None)
        if image_path is None:
            raise FileNotFoundError(f"Image not found for {img_name}")
    
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        if self.is_test:
            return image, img_name
            
        label = self.dataframe.iloc[idx]['diagnosis']
        return image, label


# In[8]:


class DRModelManager:
    #Manages multiple pre-trained models for diabetic retinopathy

    def __init__(self, config, model_name, tuning_params):
        self.config = config
        self.model_name = model_name
        self.device = config.device
        #Support for dynamic hyperparameters(Optuna)
        #Default Params to use if tuning_params is None.
        self.params = tuning_params if tuning_params else{
            'fc_dim': 512,
            'dropout':0.5,
            'fc_layers':2
        }
        self.model = None
        self.feature_extractor = None
        self._initialize_model_finetune()

    # Helper method to build variable-size classification heads
    def _build_dynamic_head(self, in_features):
        layers = []

        #layer1
        layers.append(nn.Dropout(self.params['dropout']))
        layers.append(nn.Linear(in_features, self.params['fc_dim']))
        layers.append(nn.BatchNorm1d(self.params['fc_dim']))
        layers.append(nn.LeakyReLU(inplace=True))

        #Optional Layer 2 (Controlled by tuning_params)
        if self.params.get('fc_layers', 2) == 2:
            layers.append(nn.Dropout(self.params['dropout'] * 0.5))
            layers.append(nn.Linear(self.params['fc_dim'], self.params['fc_dim'] // 2))
            layers.append(nn.BatchNorm1d(self.params['fc_dim'] // 2))
            layers.append(nn.LeakyReLU(inplace=True))
            last_dim = self.params['fc_dim'] // 2

        else:
            last_dim = self.params['fc_dim']

        #Final Classification Layer
        layers.append(nn.Dropout(0.2))
        layers.append(nn.Linear(last_dim, self.config.num_classes))

        
    def _initialize_model_finetune(self):
        """Initialize pre-trained model with fine-tuning on last blocks only"""
        
        if self.model_name == 'resnet50':
            self._initialize_resnet50_finetune()
            
        elif self.model_name == 'inceptionV3':
            self._initialize_inception_v3_finetune()
            
        elif self.model_name == 'densenet121':
            self._initialize_densenet121_finetune()
        
        else:
            raise ValueError(f"Model {self.model_name} not supported")
        
        self.model.to(self.device)
        
        # Create feature extractor (all layers except the final classifier)
        self._feature_extractor()
    
    def _initialize_resnet50_finetune(self):
        """Fine-tune ResNet50: Freeze all, unfreeze layer4 and FC"""
        self.model = models.resnet50(pretrained = True)

        # CONCEPT: Freezing the Backbone
        # We start by turning off gradient calculation for ALL layers.
        # This locks the weights of the feature extractor (layers 1-3)
        # so they act as a static "retinal feature detector"
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.layer4.parameters():
            param.requires_grad = True

        # Also unfreeze the BatchNorm layers in the last block
        for module in self.model.layer4.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.train()  # Set to training mode
                for param in module.parameters():
                    param.requires_grad = True

        # Using dynamic head builder instead of hardcoded Sequential
        num_feature = self.model.fc.in_features
        self.model.fc = self._build_dynamic_head(num_feature)

        for param in self.model.fc.parameters():
            param.requires_grad = True

    def _initialize_inception_v3_finetune(self):
        """Fine-tune InceptionV3: Unfreeze Mixed_7 blocks"""
        # Note: aux_logits=True is required for stable Inception training
        self.model = models.inception_v3(pretrained=True, aux_logits=True)

        for param in self.model.parameters():
            param.requires_grad = False
        # InceptionV3 architecture: Unfreeze from Mixed_7c onward (last few blocks)
        # Mixed_6a to Mixed_7c are the later blocks

        for name, param in self.model.named_parameters():
            if 'Mixed_7' in name or 'Mixed_6e' in name or 'Mixed_6d' in name: # type: ignore
                param.requires_grad = True
            if 'bn' in name or 'BatchNorm' in name:  # Unfreeze BatchNorm in unfrozen blocks
                if 'Mixed_7' in name or 'Mixed_6' in name:
                    param.requires_grad = True

        #Handle the Auxiliary Classifier such that it does not return 1000 classe 
        #intsead it return our 5 class output
        if self.model.AuxLogits is not None:
            num_aux_ftrs = self.model.AuxLogits.fc.in_features
            self.model.AuxLogits.fc = nn.Linear(num_aux_ftrs, self.config.num_classes)
            for param in self.model.AuxLogits.fc.parameters():
                param.requires_grad = True
                
        # Replace the final FC layer
        num_features = self.model.fc.in_features
        self.model.fc = self._build_dynamic_head(num_features)
        
        # Set FC layer to trainable
        for param in self.model.fc.parameters():
            param.requires_grad = True

    def _initialize_densenet121_finetune(self):
        """Fine-tune DenseNet121: only train last dense block"""
        self.model = models.densenet121(pretrained=True)
        
        # FREEZE ALL LAYERS FIRST
        for param in self.model.parameters():
            param.requires_grad = False
        
        # DenseNet121: Unfreeze only the last dense block (denseblock4)
        # and transition layer before it
        for name, param in self.model.named_parameters():
            if 'denseblock4' in name or 'norm5' in name:
                param.requires_grad = True
            if 'transition3' in name:  # The transition before last block
                param.requires_grad = True
        
        # Unfreeze BatchNorm layers in the unfrozen blocks
        for name, module in self.model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                if 'denseblock4' in name or 'norm5' in name:
                    module.train()
                    for param in module.parameters():
                        param.requires_grad = True
        
        # Get the original classifier's input features
        num_features = self.model.classifier.in_features
        
        # **FIX: Create a new Sequential classifier and replace the old one**
        # DenseNet's classifier is a single Linear layer, so we need to wrap our
        # custom layers in a Sequential and assign to classifier
        
        # First, let's check what type of classifier we have
        print(f"DenseNet classifier type: {type(self.model.classifier)}")
        print(f"DenseNet classifier: {self.model.classifier}")
        
        # **CRITICAL: Replace the classifier with our Sequential**
        self.model.classifier = self._build_dynamic_head(num_features)
        
        # Set classifier to trainable
        for param in self.model.classifier.parameters():
            param.requires_grad = True
        
        print(f"Created new classifier with {num_features} input features")
        print(f"New classifier architecture: {self.model.classifier}")

    def _feature_extractor(self):
        """Creates a version of the model that outputs embeddings"""
        
        # First, verify the model exists
        if self.model is None:
            raise ValueError(f"Model is None! Check initialization for {self.model_name}")
        
        if self.model_name == 'resnet50':
            # Original ResNet50 structure:
            # [conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool, fc]

            # We take everything EXCEPT the final FC layer:
            self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1])
            # This gives us: [conv1 → layer4 → avgpool]
            # Output shape: (batch_size, 2048, 1, 1) after avgpool
        
        # InceptionV3 has different structure
        # We need to add AdaptiveAvgPool2d because Inception's pooling might vary
        elif self.model_name == 'inceptionV3':
            # --- FIX FOR INCEPTION V3 ---
            # We cannot simple use nn.Sequential because Inception has a complex graph.
            # Instead, we copy the model and replace the final classification layer (fc)
            # with an Identity layer. This preserves the internal graph while outputting features.
            
            # 1. Create a shallow copy of the model structure to avoid breaking the original
            self.feature_extractor = copy.deepcopy(self.model)
            
            # 2. Disable AuxLogits to prevent tuple outputs ((logits, aux)) during inference
            self.feature_extractor.aux_logits = False
            
            # 3. Replace the final FC layer with Identity
            # This makes the model output the 2048-dim feature vector directly
            self.feature_extractor.fc = nn.Identity()

        # DenseNet structure is different: features + classifier
        elif self.model_name == 'densenet121':
            # FIX: Ensure model has features attribute
            if not hasattr(self.model, 'features'):
                raise AttributeError(f"DenseNet121 model doesn't have 'features' attribute")
            
            self.feature_extractor = nn.Sequential(
                self.model.features,
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1,1))
            )
            # Output shape: (batch_size, 1024, 1, 1)

        else:
            raise ValueError(f"Unsupported model for feature extraction: {self.model_name}")

        #Before Extraction of(CNN Output):
        #For a batch of 16 images: Shape: (16, 2048, 7, 7)
        # 2048 channels, 7x7 spatial grid

        #After AdaptiveAvgPool2d((1,1)):
        #Shape: (16, 2048, 1, 1)  # Each channel averaged to single value

        #Shape: (16, 2048, 1, 1)  # Each channel averaged to single value
        #Shape: (16, 2048)  # 2048-dimensional feature vector per image

        self.feature_extractor.to(self.device)
        # Set to evaluation mode for inference
        self.feature_extractor.eval()

    def print_trainable_parameters(self):
        """Print which layers are trainable - useful for debugging"""
        print(f"\n{'='*60}")
        print(f"Trainable parameters for {self.model_name}:")
        print('='*60)

        total_parameter = 0
        trainable_parameter = 0 

        for name, param in self.model.named_parameters():
            total_parameter += param.numel()
            if param.requires_grad:
                trainable_parameter += param.numel()
                print(f"✓ TRAINABLE: {name}")
            else:
                print(f"  Frozen: {name}")
        
        print(f"\nTotal parameters: {total_parameter:,}")
        print(f"Trainable parameters: {trainable_parameter:,}")
        print(f"Percentage trainable: {100 * trainable_parameter / total_parameter:.2f}%")
        print('='*60)
        
        return trainable_parameter, total_parameter
    
    def get_model(self):
        return self.model
    
    def get_feature_extractor(self):
        if self.feature_extractor is None:
            raise ValueError("Feature extractor not initialized. Call _feature_extractor() first.")
        return self.feature_extractor
    
    def save_model(self, path):
        """Save the important parameters and model details to use even after the training is done"""
        
        if self.model is None:
            raise ValueError("Cannot save: model is not initialized")
            
        if self.feature_extractor is None:
            raise ValueError("Cannot save: feature extractor is not initialized")

        trainable_names = [name for name, p in self.model.named_parameters() if p.requires_grad]

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_name': self.model_name,
            'config': self.config.to_dict(),
            'trainable_layers': trainable_names,
            'feature_extractor_state_dict': self.feature_extractor.state_dict()
        }, path)
        
        print(f"Model saved to {path}")
                
    def load_model(self, path):
        """Loading the saved model"""
        
        # First, ensure the model architecture is initialized
        if self.model is None:
            self._initialize_model_finetune()
            
        # FIX: Add weights_only=False to allow loading Config objects
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        # Load model state dictionary
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Verify model name matches
        if 'model_name' in checkpoint and checkpoint['model_name'] != self.model_name:
            print(f"Warning: Loading {checkpoint['model_name']} into {self.model_name}")

        # Loading the trainable parameter if in the model
        if 'trainable_layers' in checkpoint:
            # First freeze all
            for param in self.model.parameters():
                param.requires_grad = False
            # Then unfreeze saved trainable layers
            for name, param in self.model.named_parameters():
                if name in checkpoint['trainable_layers']:
                    param.requires_grad = True
        
        self._feature_extractor()
        
        print(f"Model loaded from {path}")
        return self.model
    
    def debug_model_state(self):
        """Debug method to check model initialization"""
        print(f"\n{'='*60}")
        print(f"Debug: {self.model_name}")
        print(f"{'='*60}")
        print(f"1. Model is None: {self.model is None}")
        print(f"2. Feature extractor is None: {self.feature_extractor is None}")
        
        if self.model is not None:
            print(f"3. Model type: {type(self.model)}")
            print(f"4. Model has children: {hasattr(self.model, 'children')}")
            
            if hasattr(self.model, 'children'):
                children = list(self.model.children())
                print(f"5. Number of children: {len(children)}")
                print(f"6. First 3 children types:")
                for i, child in enumerate(children[:3]):
                    print(f"   [{i}] {type(child).__name__}")
                if len(children) > 3:
                    print(f"   ... and {len(children)-3} more")
        
        print(f"{'='*60}")

        


# In[ ]:


class DRTrainer:
    "Training engine for our fine tune CNNs"

    def __init__(self, config, model_manager, trial=None):
        self.config = config
        self.model_manager = model_manager
        self.model = self.model_manager.get_model()
        self.device = self.config.device
        self.trial = trial

        #Printing the Trainable parameter information
        self.model_manager.print_trainable_parameters()

        training_params = [p for p in self.model.parameters() if p.requires_grad]

        if training_params == 0:
            raise ValueError("No training parameters found. Check the fine tuning.")
        
        print(f"\nOptimizing {len(training_params)} parameter groups")
        
        # Different learning rates for fine-tuned layers vs new layers
        # Higher LR for new layers, lower LR for fine-tuned pretrained layers
        
        # Group parameters by type
        new_layers = []
        finetune_layers = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'fc' in name or 'classifier' in name:
                    new_layers.append(param)  # New classifier layers
                else:
                    finetune_layers.append(param)  # Fine-tuned pretrained layers
        
        # Create parameter groups with different learning rates
        # We pass these groups to the optimizer
        param_groups = [
            {'params': finetune_layers, 'lr': config.learning_rate * 0.1},
            {'params': new_layers, 'lr': config.learning_rate}  
        ]
        self.optimizer = optim.AdamW(param_groups, lr=config.learning_rate)
        
        #lr scheduler for countinuouly chaning learning and then restarting with higher after some epochs
        # Scheduler: OneCycleLR is a smart learning rate scheduler that follows a specific policy:
        # Warm up: Gradually increase learning rate from low to high
        #Annealing: Gradually decrease learning rate from high to low
        #Single cycle: All done in one complete cycle (hence the name)

        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer, 
            max_lr=[config.learning_rate * 0.1, config.learning_rate],
            epochs=config.num_epochs,
            steps_per_epoch=115, # Approx batches (3662 / 32)
            pct_start=0.3
        )

        # Loss fucntion with class wieght imbalance
        self.criterion = self._get_weighted_loss()

        #initialize GradScaler for mixed precision training if CUDA is available
        self.scaler = torch.GradScaler('cuda') if torch.cuda.is_available() else None

        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'learning_rates': []
        }

    def _get_weighted_loss(self):
        """
    Dynamically calculates class weights based on the training data.
    Higher weights are assigned to rare classes (like Severe DR) to prevent bias.
    """
        #Used to assign more weight to less frequency labels in the dataset to avoid baises
        #Calculate the weight of each class by - Total sample / no.of classes * count of item in class i
        if not self.config.train_csv.exists():
            print("Warning: Train CSV not found for weight calc. Using default weights.")
            return nn.CrossEntropyLoss()
        

        df = pd.read_csv(self.config.train_csv)
        # Count samples per class
        counts = df["diagnosis"].value_counts().sort_index()
        class_counts = counts.values
        # Calculate weights: Total / (Num_Classes * Class_Count)
        # This is the standard "Balanced" formula
        total_samples = sum(class_counts)
        num_classes = len(class_counts)
        weights = total_samples / (num_classes * class_counts)

        class_weights = torch.tensor(weights, dtype=torch.float32)
        #trunsout to be tensor([0.4058, 1.9795, 0.7331, 3.7948, 2.4827])
        #Normalize weights
        class_weights = class_weights / class_weights.sum()
        class_weights = class_weights.to(self.device)

        print(f"Computed Class Weights: {class_weights}")
        # Expected Output for APTOS: tensor([0.05, 0.22, 0.08, 0.41, 0.24]) approx
        return nn.CrossEntropyLoss(weight= class_weights)
    
    def train_epoch(self, train_loader, epoch):
        self.model.train()

        #Handling the BatchNorm blocks in fine tunning to make sure they are in traning mode
        for name, module in self.model.named_modules():
            if isinstance(module, nn.BatchNorm2d) and hasattr(module, 'weight'):
                if module.weight.requires_grad:
                    module.train()

        running_loss = 0
        correct = 0
        total = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.config.num_epochs}')
        for batch_idx, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            #Handling the InceptionV3 duo outputs during training (output, aux_output)
            # Adding autocast context for mixed precision compatibility
            with torch.autocast('cuda', enabled=(self.scaler is not None)):
                if self.model_manager.model_name == 'inceptionV3':
                    outputs, aux_outputs = self.model(inputs)
                    # outputs: Main prediction from final layer
                    # aux_outputs: Auxiliary prediction from middle layer

                    loss1 = self.criterion(outputs, labels)
                    loss2 = self.criterion(aux_outputs, labels)
                    loss = loss1 + 0.4 * loss2  # Weighted sum as in original paper
                else:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)

            # CONCEPT: Gradient Clipping
            # Fine-tuning can sometimes produce large gradients that destabilize the
            # pre-trained weights. We clip the gradient norm to 1.0 to ensure smooth updates.
            #for faster training we use mix precision training where we use FP16 and Fp32
            if self.scaler:  # If we have a GPU that supports mixed precision
                # 1. Scale up the loss (prevents underflow)
                self.scaler.scale(loss).backward()
                # Loss is multiplied by e.g., 65536 before backward pass
                
                # 2. Unscale gradients before optimizer step
                self.scaler.unscale_(self.optimizer)
                # Now gradients are back to normal scale
                
                # 3. Clip gradients (prevent overflow)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad], 
                    max_norm=1.0
                )
                
                # 4. Optimizer step with scaling
                self.scaler.step(self.optimizer)
                
                # 5. Update scale factor for next iteration
                self.scaler.update()

            else:
                loss.backward()

                #gradient Clipping 
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad], 
                    max_norm=1.0)

                self.optimizer.step()

            #statistics
            #running_loss: Sum of all batch losses in the current epoch
            #Example: If 100 batches with losses [0.5, 0.4, 0.3, ...], running_loss = 0.5 + 0.4 + 0.3 + ...
            running_loss += loss.item()

            #getting the prediction outputs where we recive 5 output and only choose max value from each iteration
            _, predicted = outputs.max(1)
            total += labels.size(0)

            #Gettting the total correctly predicted labels in each iteration
            correct += predicted.eq(labels).sum().item()

            #showing the progress bar to monitor the performance
            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })

        #Managing the loss per epcoh
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total

        self.history['train_loss'].append(epoch_loss)
        self.history['train_acc'].append(epoch_acc)

        return epoch_loss, epoch_acc
    
    def validate(self, val_loader):
        #Performing the validation for our trained model
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                #For InceptionV3 in eval mode, no aux output
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                running_loss = loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100* correct/ total

        self.history['val_loss'].append(epoch_loss)
        self.history['val_acc'].append(epoch_acc)

        return epoch_loss, epoch_acc, all_preds, all_labels
    
    def save_checkpoint(self, epoch, best_acc, save_path):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_acc': best_acc,
            'history': self.history
        }, save_path)

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.history = ckpt['history']
        return ckpt['epoch'], ckpt['best_acc']
        
    def train(self, train_loader, val_loader, start_epoch=0, best_acc=0):
        #Complete Traing loopwith fine tunning included

        checkpoint_path = self.config.model_dir / f'{self.model_manager.model_name}_finetune_checkpoint.pth'
        best_model_path = self.config.model_dir / f"{self.model_manager.model_name}_finetune_best.pth"

        print(f"\nStarting fine-tunning for {self.model_manager.model_name}")
        print(f"Checkpoint will be saved to: {checkpoint_path}")

        for epoch in range(start_epoch, self.config.num_epochs):
            # Adjust learning rate if using warmup
            if epoch < 5: # Warmup phase
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.config.learning_rate * (epoch + 1) / 5
            
            # --- CORRECTION: Training logic moved OUT of the warmup/param loop ---

            # Train
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss, val_acc, val_preds, val_labels = self.validate(val_loader)

            # Update the learning rate scheduler
            self.scheduler.step(epoch + train_loss)

            # storing learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rates'].append(current_lr)

            # Save checkpoint
            self.save_checkpoint(epoch, best_acc, checkpoint_path)

            #Reporting intermidiate results to Optuna
            if self.trial:
                self.trial.report(val_acc, epoch)
                #Handle pruning(stop this trial if it's not promising)
                if self.trial.should_prune():
                    raise optuna.TrialPruned()

            # Save the best model
            if val_acc > best_acc:
                best_acc = val_acc
                #Only save to disk if we are Not tuning(to save time), or if it's a really good model
                #If self.trial is None, we are in normal training model -> always save.
                if self.trial is None:
                    self.model_manager.save_model(best_model_path)
                    print(f"New best model saved with accuracy: {best_acc:.2f}%")
                else:
                    #In tuning mode, we might skip saving to disk to be faster, 
                    # unless you explicitly want to keep the best tuned models.
                    # For now, I'll keep saving it so you don't lose the result.
                    self.model_manager.save_model(best_model_path)
                    print(f"New best model saved with accuracy: {best_acc:.2f}%")

            print(f'\nEpoch {epoch+1}/{self.config.num_epochs}:')
            print(f'Train loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

            # Print learning rates for each parameter group
            for i, param_group in enumerate(self.optimizer.param_groups):
                if i == 0:
                    print(f"  Fine-tune LR: {param_group['lr']:.6f}")
                else:
                    print(f"  New layers LR: {param_group['lr']:.6f}")
            
            print("-" * 60)

            #Stoping if the model reaches 95% accuracy in either training or validation
            if train_acc > 95.0 or val_acc > 95.0:
                print(f"\n{'='*40}")
                print("Traget of 95+ plus reached.")
                print(f"Train:{train_acc:.2f}% | Val: {val_acc:.2f}%")
                print(f"Stopping training early.")
                print(f"{'='*40}")

                #Ensuring if the best model is saved if this final run was the best
                if val_acc >= best_acc:
                    self.model_manager.save_model(best_model_path)
                    break
        
        
        # --- CORRECTION: Final loading moved OUT of the epoch loop ---
        # Load best model for final evaluation
        self.model_manager.load_model(best_model_path)
        print(f"\n✓ Fine-tuning completed for {self.model_manager.model_name}")
        print(f"✓ Best validation accuracy: {best_acc:.2f}%")
        
        return self.history


# In[90]:


class FeatureExtractor:
    # Passes images through the trained CNN models (minus the final classification layer)
    # to extract high-level feature vectors (embeddings). These vectors are then used 
    # as the input data to train the XGBoost classifier.
    def __init__(self, config):
        self.config = config
        self.device = config.device

    def extract_feature(self,model_manager, data_loader):
        feature_extractor = model_manager.get_feature_extractor()
        feature_extractor.eval()
        all_features, all_labels = [], []

        with torch.no_grad():
            for inputs, labels in tqdm(data_loader, desc='Extracting features'):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                features = feature_extractor(inputs)
                features = features.view(features.size(0), -1)
                all_features.append(features.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        if len(all_features) == 0:
            raise ValueError("No features extracted! Check your data loader.")
            
        return np.vstack(all_features), np.concatenate(all_labels)        
            


# In[96]:


class XGBoostTrainer:
    def __init__(self, config):
        self.config = config

    def train_single_model(self, X_train, y_train, X_val, y_val):
        # FIX: Pass early_stopping_rounds to the constructor, NOT .fit()
        model = xgb.XGBClassifier(
            **self.config.xgb_params,
            early_stopping_rounds=10, 
            eval_metric="mlogloss"  # Required for multi-class early stopping
        )
        
        # Train (verbose=False to keep output clean)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        return model

    def train_ensemble(self, feature_list, y_train, features_val_list, y_val):
        X_train_combined = np.hstack(feature_list)
        X_val_combined = np.hstack(features_val_list)
        
        # FIX: Pass early_stopping_rounds to the constructor here as well
        model = xgb.XGBClassifier(
            **self.config.xgb_params,
            early_stopping_rounds=10,
            eval_metric="mlogloss"
        )
        
        model.fit(X_train_combined, y_train, eval_set=[(X_val_combined, y_val)], verbose=False)
        return model
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)

        metrics = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'cohen_kappa': cohen_kappa_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted'),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        return metrics, y_pred
    
    def save_model(self, model, model_name):
        with open(self.config.model_dir/f'{model_name}_xgb.pkl', 'wb') as f:
            pickle.dump(model, f)

    def load_model(self, model_name):
        with open(self.config.model_dir/ f"{model_name}_xgb.pkl", 'rb') as f:
            return pickle.load(f)
        


# In[92]:


class ResultsVisualizer:
    def __init__(self, config):
        self.config = config
        plt.style.use('seaborn-v0_8-darkgrid')
    
    def plot_training_history(self, history, model_name):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].plot(history['train_loss'], label='Train')
        axes[0].plot(history['val_loss'], label='Val')
        axes[0].set_title('Loss')
        axes[0].legend()
        
        axes[1].plot(history['train_acc'], label='Train')
        axes[1].plot(history['val_acc'], label='Val')
        axes[1].set_title('Accuracy')
        
        axes[2].plot(history['learning_rates'])
        axes[2].set_title('Learning Rate')
        
        plt.tight_layout()
        plt.savefig(self.config.results_dir / f"{model_name}_history.png")
        plt.close()

    def save_metrics_report(self, metrics_dict):
        with open(self.config.results_dir / "metrics_report.json", 'w') as f:
            json.dump(metrics_dict, f, indent=4)


# In[93]:


class DiabeticRetionpathyPipeline:
    def __init__(self, config):
        self.config = config
        self.model = {}
        self.visualizer = ResultsVisualizer(config)
        torch.manual_seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)
        
    def load_and_prepare_data(self):
        print("Loading APTOS 2019 dataset...")
        
        # 1. Load the CSV
        df = pd.read_csv(self.config.train_csv)
        print(f"Original CSV size: {len(df)}")
        
        # 2. --- NEW STEP: Filter out missing images ---
        valid_rows = []
        # Check extensions
        extensions = ['.png', '.jpg', '.jpeg']
        
        print("Verifying image files...")
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Checking files"):
            img_id = row['id_code']
            found = False
            
            # Check if file exists with any valid extension
            for ext in extensions:
                path = self.config.train_images_dir / f"{img_id}{ext}"
                if path.exists():
                    valid_rows.append(row)
                    found = True
                    break
            
            # Optional: Print the first missing one to debug
            if not found and len(df) - len(valid_rows) == 1:
                print(f"Warning: Could not find image for ID: {img_id}")

        # Create new cleaned dataframe
        df_clean = pd.DataFrame(valid_rows)
        print(f"Cleaned dataset size: {len(df_clean)} (Removed {len(df) - len(df_clean)} missing files)")
        
        if len(df_clean) == 0:
            raise ValueError("No valid images found! Check your paths.")

        # 3. Split data (Using the CLEAN dataframe)
        from sklearn.model_selection import train_test_split
        train_df, val_df = train_test_split(
            df_clean, # Use clean df
            test_size=0.2, 
            random_state=self.config.random_seed,
            stratify=df_clean['diagnosis']
        )
        
        # 4. Create DataLoaders (Rest of your code is same)
        train_loader = DataLoader(
            CustomAptos(train_df, self.config.train_images_dir, DataAugmentation.get_train_transform()),
            batch_size=self.config.batch_size, 
            shuffle=True, 
            num_workers=0, # Keep this 0 for Windows!
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            CustomAptos(val_df, self.config.train_images_dir, DataAugmentation.get_val_transform()),
            batch_size=self.config.batch_size, 
            shuffle=False, 
            num_workers=0, # Keep this 0 for Windows!
            pin_memory=True,
            drop_last=True #We drop the last batch if it has less images than the batch size
        )
        
        return train_loader, val_loader
    
    def run_pipeline(self):
        print("\n===STARTING FINE-TUNED DR DETECTION PIPELINE===")
        
        #1. Train Fine-Tunned CNNs
        train_loader, val_loader = self.load_and_prepare_data()

        for model_name in self.config.pretrained_models.keys():
            #Check if model already exists to skip trainig
            check_path = self.config.model_dir / f'{model_name}_finetune_best.pth'

            manager = DRModelManager(self.config, model_name)

            if check_path.exists():
                print(f"\nFound existing wights for {model_name}. Loading the model.")
                manager.load_model(check_path)

            else:
                print(f"\nFine-Tuning {model_name}...")
                trainer = DRTrainer(self.config, manager)
                histroy = trainer.train(train_loader, val_loader)
                self.visualizer.plot_training_history(histroy, model_name)

                #Free up memory from trainer
                del trainer

            #Load the best model we just saved to ensure we have the best state
            manager.load_model(check_path)
            self.model[model_name] = manager

            torch.cuda.empty_cache() # Clear VRAM to avoid out of memory chrash for GPU(6GB)
        
        #2. Extract features
        print("/nExtracting Feature for XGBoost...")
        feature_extractor = FeatureExtractor(self.config)
        all_feature = {}

        for model_name, manager in self.model.items():
            print(f"Extracting Features form {model_name}")
            
            feats, labels = feature_extractor.extract_feature(manager, val_loader)
            all_feature[model_name] = {'features': feats, 'labels': labels}

        # 3. Train XGBoost Ensemble
        print("\nTraining XGBoost Ensemble...")
        xgb_trainer = XGBoostTrainer(self.config)
        all_metrics = {}
        
        from sklearn.model_selection import train_test_split
        
        # Train Ensemble
        X_combined_list, X_val_combined_list = [], []
        y_train_all, y_test_all = None, None
        
        for model_name in self.model.keys():
            feats, labels = all_feature[model_name]['features'], all_feature[model_name]['labels']
            X_tr, X_te, y_tr, y_te = train_test_split(feats, labels, test_size=0.2, stratify=labels, random_state=42)
            
            # Train individual XGBoost for reporting
            xgb_model = xgb_trainer.train_single_model(X_tr, y_tr, X_te, y_te)
            metrics, _ = xgb_trainer.evaluate_model(xgb_model, X_te, y_te, f"{model_name}_xgb")
            all_metrics[f"{model_name}_xgb"] = metrics
            xgb_trainer.save_model(xgb_model, model_name)

            X_combined_list.append(X_tr)
            X_val_combined_list.append(X_te)
            if y_train_all is None: y_train_all, y_test_all = y_tr, y_te

        if not X_combined_list:
            raise ValueError("No features were extracted. Pipeline cannot continue to Ensemble training.")

        # Train Ensemble XGBoost
        ensemble_model = xgb_trainer.train_ensemble(X_combined_list, y_train_all, X_val_combined_list, y_test_all)
        X_test_combined = np.hstack(X_val_combined_list)
        metrics, _ = xgb_trainer.evaluate_model(ensemble_model, X_test_combined, y_test_all, 'ensemble_xgb')
        all_metrics['ensemble_xgb'] = metrics
        xgb_trainer.save_model(ensemble_model, 'ensemble')

        #Final Report
        self.visualizer.save_metrics_report(all_metrics)
        print("\n Pipeline Complete. Final Metrics:")
        print(json.dumps(all_metrics, indent=2))
        return True


# In[94]:


config_instance = Config()
Pipeline = DiabeticRetionpathyPipeline(config_instance)

Pipeline.load_and_prepare_data()


# In[97]:


Pipeline.run_pipeline()


# In[ ]:


import os
from pathlib import Path

# 1. Initialize your config
config = Config()

# 2. Print where the code THINKS the images are
print(f"Code is looking in: {config.train_images_dir.resolve()}")

# 3. Check if that folder actually exists
if not config.train_images_dir.exists():
    print("❌ ERROR: The directory does not exist!")
else:
    print("✅ Directory exists.")
    # 4. List the first 5 files to see what they look like
    print("First 5 files found in folder:")
    print(os.listdir(config.train_images_dir)[:5])


# In[1]:


import nbformat
from nbconvert import PythonExporter

# Load notebook
with open('DRpipline2.ipynb', 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

# Export to Python
exporter = PythonExporter()
python_code, _ = exporter.from_notebook_node(nb)

# Save to file
with open('DRpipline.py', 'w', encoding='utf-8') as f:
    f.write(python_code)

print("Converted successfully!")

