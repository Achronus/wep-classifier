import os
import numpy as np

from functions.model import Classifier
from functions.utils import Utilities

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets
import torchvision.models as models

class Tuner:
    """
    Contains utility functions that are used within the hyperparameter tuning Notebook. Combines multiple components from the initial Notebook to condense the hyperparameter Notebook and focus on the tuning.
    """
    def __init__(self):
        self.utils = Utilities()
        self.device = self.utils.set_device()
    
    def set_data(self, filepath):
        """
        Sets the dataset using pre-defined transformations.
        
        Parameters:
            filepath (string) - filepath to the dataset
        """
        # Set transformations for batch data
        transform = transforms.Compose([
            transforms.Resize(224), # Resize images to 224
            transforms.CenterCrop(224), # Make images 224x224
            transforms.RandomHorizontalFlip(), # Randomly flip some samples (50% chance)
            transforms.RandomRotation(20), # Randomly rotate some samples
            transforms.ToTensor(), # Convert image to a tensor
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) # Normalize image values
        ])
        
        # Set dataset and labels
        dataset = torchvision.datasets.ImageFolder(filepath, transform=transform)
        self.labels = np.array(list(dataset.class_to_idx), dtype=object)
        return dataset
    
    def _set_initial_models(self, n_classes, h_layers):
        """
        Helper function used to set the three models (GoogLeNet, MobileNet v2, and ResNet-34) with new classifiers.
        
        Parameters:
            n_classes (int) - number of classes for to output
            h_layers (list) - integers that represent each layers node count 
        """
        # Create instances of pretrained CNN architectures
        googlenet = models.googlenet(pretrained=True)
        mobilenetv2 = models.mobilenet_v2(pretrained=True)
        resnet34 = models.resnet34(pretrained=True)
        
        # Initialize new classifiers
        gnet_classifier = Classifier(in_features=googlenet.fc.in_features, out_features=n_classes, 
                                     hidden_layers=h_layers)
        mobilenet_classifier = Classifier(in_features=mobilenetv2.classifier[1].in_features, 
                                          out_features=n_classes, hidden_layers=h_layers)
        resnet_classifier = Classifier(in_features=resnet34.fc.in_features, out_features=n_classes, 
                                       hidden_layers=h_layers)
        
        cnn_models = [googlenet, mobilenetv2, resnet34]
        # Freeze architecture parameters to avoid backpropagating them
        # Avoiding replacing pretrained weights
        for model in cnn_models:
            for param in model.parameters():
                param.requires_grad = False
        
        # Replace last FC/classifier with new classifier
        googlenet.fc = gnet_classifier
        mobilenetv2.classifier = mobilenet_classifier
        resnet34.fc = resnet_classifier

        return cnn_models
    
    def calc_params(self, model_names, n_classes, h_layers):
        """
        Used to calculate the amount of trainable parameters vs total parameters for each model.
        
        Parameters:
            model_names (list) - a list of the model names
            n_classes (int) - number of output classes
            h_layers (list) - hidden node integers, one per layer
        """
        models = self._set_initial_models(n_classes, h_layers)
        
        # Total params for each model
        for idx, model in enumerate(models):
            print(f"{model_names[idx]}:")
            model.total_params = sum(p.numel() for p in model.parameters())
            print(f'{model.total_params:,} total parameters')
            model.trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f'{model.trainable_params:,} training parameters\n')
    
    def _set_filename(self, model_name, batch_size, h_layers):
        """
        Helper function used to set the saved models filename. Returns the filename.
        
        Parameters:
            model_name (string) - name of the model
            batch_size (int) - batch size of data loader
            h_layers (list) - hidden node integers, one per layer
        
        Format: [model_name]_[batch_size]_[hidden_sizes]
        """
        filename = f"{model_name}_{str(batch_size)}_"
        for layer in h_layers:
            filename += f"{str(layer)}_"
        return filename[:-1]
        
    
    def tune_model(self, model_names, batch_size, train_loader, valid_loader, 
                   n_classes, h_layers, lr, epochs=1000, iterations=2, patience=5):
        """
        Used to tune a model on the given training loader, evaluated against the validation loader. Iterates over a list of hidden layers, saving multiple model versions.
        
        Parameters:
            model_names (list) - a list of the model names
            batch_size (int) - batch size of the train and validation loader
            train_loader (torch.DataLoader) - torch training dataset loader
            valid_loader (torch.DataLoader) - torch validation dataset loader
            n_classes (int) - number of output classes
            h_layers (list) - lists of a variety of hidden node sizes
            lr (float) - learning rate for training the model 
            epochs (int) - number of epochs for training (default: 1000)
            iterations (int) - iterations per number of epochs (default: 2)
            patience (int) - number of epochs to wait before early stopping (default: 5)
        """
        # Iterate over hidden layers
        for l in range(len(h_layers)):
            # Create instances of pretrained CNN architectures
            models = self._set_initial_models(n_classes, h_layers[l])

            # Iterate over models
            for m in range(len(models)):
                filename = self._set_filename(model_names[m], batch_size, h_layers[l])
                filepath = "saved_models/" + filename + ".pt"
                
                # Skip model training if already has been trained
                if os.path.isfile(filepath):
                    print(f"{filename} already trained.")
                else:
                    print(f"\nTraining: {filename}")
                    criterion = nn.NLLLoss() # Negative Log Likelihood Loss

                    # Set optimizer
                    if m == 1: # MobileNetV2 specific
                        optimizer = torch.optim.Adam(models[m].classifier.parameters(), 
                                                 lr=lr)
                    else:
                        optimizer = torch.optim.Adam(models[m].fc.parameters(), 
                                                 lr=lr)

                    models[m].to(self.device) # move to GPU

                    # Train model
                    self.utils.train(models[m], train_loader, valid_loader, criterion, 
                                     optimizer, filepath, epochs, iterations, patience)

    def get_model_names(self, model_paths):
        """
        Used to obtain the model names from a list of saved models filenames.
        
        Parameters:
            model_paths (list) - saved model names as strings
        """
        model_names = []
        
        # Set model names
        for item in model_paths:
            model_names.extend(item.split('_')) # Split into individual components
        
        model_names = list(set(model_names)) # Create a unique list
        model_names.sort() # Sort into order (numbers first)
        model_names = model_names[-3:] # Get only model names
        return model_names
                    
    def set_model(self, model_paths, model, model_name, batch_size, h_layers):
        """
        Used to check what type of model needs to be set for testing. Returns the model and its name.
        
        Name format: [model]_[batch_size]_[hidden_size]_[hidden_size]
        
        Parameters:
            model_paths (list) - list of filepaths of saved models
            model (torchvision.models) - initial pretrained model
            model_name (string) - name of the model
            batch_size (int) - number of dataset batches
            hidden_size (list) - hidden node sizes as integers
        """
        # Set initial variables
        load_name = ""
        compare_parts = [model_name, batch_size]
        compare_parts.extend(h_layers)
        
        # Iterate over each model
        for filepath in model_paths:
            compare_name = filepath.split('/')[-1].rstrip('.pt').split('_')

            if compare_name[0] == compare_parts[0] \
            and compare_name[1] == str(compare_parts[1]) \
            and compare_name[-2] == str(compare_parts[-2]) \
            and compare_name[-1] == str(compare_parts[-1]):
                # Return saved model
                load_name = filepath.split('/')[-1].rstrip('.pt')
                self.utils.load_model(model, 'saved_models/' + filepath)
                break
        
        return model, load_name
    
    def save_best_models(self, model_stats, model_names, n_classes):
        """
        Used to save the three best performing models based on the statistics of all model variations.
        
        Parameters:
            model_stats (pandas.DataFrame) - table of best model statistics
            model_names (list) - name of the models as strings
            n_classes (int) - number of output classes
        """
        best_models = []
        
        # Iterate over each model
        for idx, item in enumerate(model_stats['Name']):
            name, batch, h1, h2 = item.split('_')
            h_layers = [int(h1), int(h2)]
            filepath = f'saved_models/{item}.pt'
            cnn_models = self._set_initial_models(n_classes, h_layers)
            
            # Check names match
            for cnn_name in model_names:
                if name == cnn_name:
                    # Load model and store it
                    model = cnn_models[idx]
                    self.utils.load_model(model, filepath)
                    best_models.append(model)
                    new_name = cnn_name.replace('-', '').lower()
                    
                    # Set statistics
                    stats = list(model_stats.iloc[idx, 1:])
                    stats = {'accuracy': stats[0], 'top-1 error': stats[1],
                             'top-5 error': stats[2], 'precision': stats[3],
                             'recall': stats[4], 'f1-score': stats[5]}
                    
                    # Set additional model parameters
                    model.batch_size = batch
                    model.h_layers = h_layers
                    model.stats = stats
                    
                    # Save as best model
                    self._save_model(model)
        
        return best_models
    
    def _save_model(self, model):
        """
        Helper function used to save the best models.
        
        Parameters:
            model (torchvision.models) - model to save
        """
        torch.save({'parameters': model.state_dict(),
                    'train_losses': model.train_losses,
                    'valid_losses': model.valid_losses,
                    'batch_size': model.batch_size,
                    'h_layers': model.h_layers,
                    'stats': model.stats, 
                    }, f'saved_models/best_{new_name}.pt')
        
    def load_best_models(self, models, filenames):
        """
        Used to load the three best models.
        
        Parameters:
            model (list) - torchvision.models to load
            filenames (list) - filenames of saved models to load within saved_models folder
        """
        # Set a checkpoint
        for idx, model in enumerate(models):
            checkpoint = torch.load(f"saved_models/{filenames[i]}.pt")

            # Store utility variables
            model.train_losses = checkpoint['train_losses']
            model.valid_losses = checkpoint['valid_losses']
            model.batch_size = checkpoint['batch_size']
            model.h_layers = checkpoint['h_layers']
            model.stats = checkpoint['stats']

            # load model parameters
            model.load_state_dict(checkpoint['parameters'])
        
        print("Models loaded. Utility variables available:")
        print("\ttrain_losses, valid_losses, batch_size, h_layers, stats, parameters.")