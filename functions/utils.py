import time
import numpy as np

import torch
from torch.utils.data.sampler import SubsetRandomSampler

class Utilities():
    """
    Contains utility functions for handling data, training and testing the models.
    """
    def set_device(self):
        """
        Sets CUDA device to GPU is available.
        """
        if torch.cuda.is_available():
            device = "cuda"
            print("CUDA is available. Device has been set to GPU.")
        else:
            device = "cpu"
            print("CUDA unavailable. Device has been set to CPU.")
        
        # Set object device and return it
        self.device = device
        return device
    
    def augment_data(self, filepath, transform, batch_size, split_size, seed=1):
        """
        Used to augment and split the data into dataloaders.
        
        Parameters:
            filepath (string) - File location of the image dataset
            transform (torchvision.transforms) - list of transforms
            batch_size (int) - number of images per batch
            split_size (float) - size of split for both the test and validation sets
            seed (int) - number for recreating previous instances
        """
        # Store and transform data
        dataset = torchvision.datasets.ImageFolder(filepath, transform=transform)
        
        # Set random seed
        np.random.seed(seed)
        
        # Split and return the data
        train_loader, valid_loader, test_loader = self.split_data(dataset, batch_size,
                                                                  split_size)
        return train_loader, valid_loader, test_loader
    
    def split_data(self, dataset, batch_size, split_size=0.15):
        """
        Splits the specified dataset into dataloaders and returns them.
        
        Parameters:
            dataset (np_array) - dataset of images
            batch_size (int) - number of images per batch
            split_size (float) - size of split for both the test and validation sets
        """
        # Obtain dataset indices
        num_train = len(dataset) # 16,526
        idxs = list(range(num_train))
        np.random.shuffle(idxs) # Shuffle the idxs
        split = int(np.floor(split_size * num_train)) # same for test and valid

        # Set dataset indices
        valid_idx = idxs[:split] # first 15% (2,478)
        test_idx = idxs[split:split*2] # second 15% (2,478)
        train_idx = idxs[split*2:] # last 70% (11,570)

        # Define samplers for obtaining data batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        test_sampler = SubsetRandomSampler(test_idx)
        
        # Set class specific variables for other functions
        self.batch_size = batch_size
        self.trainset_size = len(train_sampler)

        # Prepare data loaders (combine dataset and sampler)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                   sampler=train_sampler)
        valid_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                   sampler=valid_sampler)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                  sampler=test_sampler)
        return train_loader, valid_loader, test_loader
    
    def train(self, model, train_loader, valid_loader, criterion, 
              optimizer, filepath, epochs=1000, iterations=2, patience=5):
        """
        Used to train a model.
        
        Parameters:
            model (torchvision.models) - model to train
            train_loader (torch.DataLoader) - torch training dataset loader
            valid_loader (torch.DataLoader) - torch validation dataset loader
            criterion (torch.loss) - loss function
            optimizer (torch.optim) - optimizer function
            filepath (string) - filepath for saving the model
            epochs (int) - number of epochs for training (default: 1000)
            iterations (int) - iterations per number of epochs (default: 2)
            patience (int) - number of epochs to wait before early stopping (default: 5)
        """
        # Initialize variables
        start_time = time.time()
        train_loss, valid_loss = 0, 0
        self.counter = 0
        stop = False
        valid_loss_min = np.Inf # initial min
        steps_total = self.trainset_size // self.batch_size
        print_every = steps_total // iterations
        valid_losses, train_losses = [], []
        
        # Begin training loop
        for e in range(epochs):
            steps = 0
            # Set to train mode
            model.train()

            # Check for early stopping
            if stop:
                break

            # Continue training
            for images, labels in train_loader:
                steps += 1

                # Set images and labels to GPU
                images, labels = images.to(self.device), labels.to(self.device)

                # Train model
                optimizer.zero_grad() # zero gradients each epoch
                output = model(images) # forward pass
                loss = criterion(output, labels) # calculate loss
                loss.backward() # backpropagate
                optimizer.step() # update weights (parameters)

                train_loss += loss.item() # update loss

                # After step amount of steps
                if steps % print_every == 0:
                    model.eval() # Set model to evaluation mode

                    # Disable gradients to speed up inference
                    with torch.no_grad():
                        valid_loss, accuracy = self.validate(model, valid_loader, criterion)

                    # Update parameters for correct calculations
                    train_loss = train_loss / print_every
                    valid_loss = valid_loss / len(valid_loader)
                    accuracy = accuracy / len(valid_loader)

                    # Add loss to list for plotting
                    train_losses.append(train_loss)
                    valid_losses.append(valid_loss)

                    # Output information
                    print(f"Epoch: {e+1}/{epochs}",
                          f"Step: {steps}/{steps_total}",
                          f"Training Loss: {train_loss:.3f}",
                          f"Validation Loss: {valid_loss:.3f}",
                          f"Accuracy: {accuracy:.3f}")

                    # Save model with best validation loss
                    if valid_loss <= valid_loss_min:
                        print(f"Validation loss decreased ({valid_loss_min:.3f}",
                              f"-> {valid_loss:.3f}). Saving model...")
                        valid_loss_min = valid_loss
                        self.save_model(model, filepath, valid_loss_min,
                                        train_loss, accuracy,
                                        train_losses, valid_losses)
                        self.counter = 0 # Reset early stop counter

                    # Early stop if patience reached before epochs end
                    elif self.early_stopping(valid_loss, patience):
                        stop = True
                        break

                    # Reset running loss and set back to training mode for next iteration
                    train_loss = 0
                    model.train()

        # Calculate training time
        train_time = (time.time() - start_time)
        self.time_taken(train_time)
    
    def validate(self, model, valid_loader, criterion):
        """
        Used to validate the performance of a model. Perfomed inside training loop.
        
        Parameters:
            model (torchvision.models) - model to train
            valid_loader (torch.DataLoader) - torch validation dataset loader
            criterion (torch.loss) - loss function
        """
        # Set initial variables
        accuracy = 0
        valid_loss = 0
        
        # Begin validation loop
        for images, labels in valid_loader:
            # Set images and labels to GPU
            images, labels = images.to(self.device), labels.to(self.device)
                          
            # Validate model
            output = model.forward(images) # forward pass
            loss = criterion(output, labels) # calculate loss
            valid_loss += loss.item()
            
            # Calculate accuracy
            # Log-softmax requires numbers to be exponential to get probabilities 
            probabilities = torch.exp(output)
            
            # Get class with highest probability
            score = (labels.data == probabilities.max(dim=1)[1])
            
            # Calculate accuracy
            accuracy += score.type_as(torch.FloatTensor()).mean()

        return valid_loss, accuracy
    
    @torch.no_grad()
    def predict(self, model, x_test):
        """
        Make predictions on the selected DataLoader. Returns the predictions (y_pred) and
        image labels (y_test).
        
        Parameters:
            model (torchvision.models) - model to make predictions with
            x_test (torch.DataLoader) - torch validation or test loader
        """
        accuracy = 0
        model.to(self.device)
        predictions, img_labels = torch.tensor([]), torch.tensor([])
        
        for images, labels in x_test:
            images, labels = images.to(self.device), labels.to(self.device)

            # Make predictions
            output = model.forward(images)

            # Log-softmax requires numbers to be exponential to get probabilities 
            probabilities = torch.exp(output)

            # Get class with highest probability
            batch_pred = probabilities.max(dim=1)[1]

            # Store predictions and labels
            predictions = torch.cat((predictions,
                                     batch_pred.cpu()), dim=0)
            img_labels = torch.cat((img_labels, 
                                    labels.cpu()), dim=0)
        
        return predictions, img_labels
    
    def early_stopping(self, valid_loss, patience):
        """
        Used to stop training early if the validation loss hasn't improved after 
        a given amount of patience (epochs).
        
        Parameters:
            valid_loss (float) - epochs current validation loss
            patience (int) - number of updates to wait for improvement before termination
        """
        early_stop = False
        self.counter += 1 # Increment counter
        print(f"Early stopping counter: {self.counter}/{patience}.")
        
        # Stop loop if patience is reached
        if self.counter >= patience:
            early_stop = True
            print("Early stopping limit reached. Training terminated.")
                
        return early_stop
    
    def save_model(self, model, filepath, valid_loss, train_loss,
                   accuracy, train_losses, valid_losses):
        """
        Used to save a trained model with utility information.
        
        Parameters:
            model (torchvision.models) - model to save
            filepath (string) - filepath and name of model to save
            valid_loss (float) - best validation loss
            train_loss (float) - best training loss
            accuracy (float) - best validation accuracy
            train_losses (list) - train losses during training 
            valid_losses (list) - validation losses during training
        """
        # Save the model parameters
        torch.save({'parameters': model.state_dict(),
                    'valid_loss': valid_loss,
                    'train_loss': train_loss,
                    'accuracy': accuracy,
                    'train_losses': train_losses,
                    'valid_losses': valid_losses,
                    }, filepath)
    
    def load_model(self, model, filepath):
        """
        Used to load a pre-trained model with its utility information.
        
        Parameters:
            model (torchvision.models) - model to load
            filepath (string) - filepath and name of saved model
        """
        # Set a checkpoint and load the model parameters
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['parameters'])
        
        # Store utility variables
        model.valid_loss = checkpoint['valid_loss']
        model.train_loss = checkpoint['train_loss']
        model.accuracy = checkpoint['accuracy']
        model.train_losses = checkpoint['train_losses']
        model.valid_losses = checkpoint['valid_losses']
        
    def time_taken(self, train_time):
        """
        Calculates the training time taken in hours, minutes and seconds.
        """
        min_secs, secs = divmod(train_time, 60)
        hours, mins = divmod(min_secs, 60)
        print(f"Total training time: {hours:.2f}hrs {mins:.2f}mins {secs:.2f}secs")
    
    def indices_to_labels(self, y_pred, y_true, class_labels):
        """
        Converts predictions and labels from indices to labels.
        
        Parameters:
            y_pred (torch.Tensor) - test or validation loader predictions
            y_true (torch.Tensor) - dataloader labels
            class_labels (list) - image class labels
        """
        class_labels_idx = range(len(class_labels))
        
        # Change from tensors to numpy arrays
        y_pred = np.array(y_pred, dtype=object)
        y_true = np.array(y_true, dtype=object)

        # Convert indices to strings
        for label in range(len(class_labels_idx)):
            for i in range(len(y_pred)):
                # Predicted labels
                if y_pred[i] == class_labels_idx[label]:
                    y_pred[i] = class_labels[label]
                # True labels
                if y_true[i] == class_labels_idx[label]:
                    y_true[i] = class_labels[label]
            
        return y_pred, y_true
        