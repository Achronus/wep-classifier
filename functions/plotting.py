import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from scikitplot.metrics import plot_confusion_matrix, plot_roc

class Plotter():
    """
    Used to create plots and visualise the given dataset.
    
    Parameters:
        labels (np_array) - list of plant class labels as strings
    """
    def __init__(self, labels):
        self.class_labels = labels
        self.class_labels_idx = np.array(range(len(labels)))
    
    def imshow(self, img):
        """
        Used to un-normalize and display an image.
        """
        img = img / 2 + 0.5 # Un-normalize
        plt.imshow(np.transpose(img, (1, 2, 0))) # Convert from tensor image

    def visualize_imgs(self, imgs, labels, figsize=(25, 15), num_rows=5, num_cols=35):
        """
        Creates subplots of a sample of images.
        
        Parameters:
            imgs (np_array) - batch of images
            labels (np_array) - batch of image labels
            figsize (tuple) - subplot figure size
            num_rows (int) - number of rows in subplots
            num_cols (int) - number of columns in subplots
        """
        fig = plt.figure(figsize=figsize)
        
        for idx in np.arange(num_cols):
            ax = fig.add_subplot(num_rows, int(np.ceil(num_cols/num_rows)), idx+1, 
                                 xticks=[], yticks=[])
            self.imshow(imgs[idx])
            ax.set_title(self.class_labels[labels[idx]])
    
    def create_plots(self, models, model_names, figsize, plot_func, 
                     plot_name=None, save=False):
        """
        Dynamically creates the correct amount of plots depending on number of models
        passed in.
        
        Parameters:
            models (list) - one or more models
            model_names (list) - model names in string format
            figsize (tuple) - size of each subplot figure
            plot_func (function) - type of plot to create
            plot_name (string) - plot name for saving
            save (boolean) - when true saves plot to plot folder
        """
        # Create subplots
        if isinstance(models, list):
            fig = plt.figure(figsize=figsize)
            fig.subplots_adjust(wspace=0.25)
            num_cols = len(models)
            # Create individual plot
            for idx in np.arange(num_cols):
                fig.add_subplot(1, num_cols, idx+1)
                plot_func(models[idx], model_names[idx])
            plt.show()
            
        # Create single plot
        else:
            fig = plt.figure()
            plot_func(models, model_names)
            plt.show()
        
        # Save plot
        if plot_name is not None and save:
            fig.savefig(f"plots/{plot_name}.png")
    
    def plot_losses(self, model, model_name):
        """
        Creates a plot of the models training loss and validation loss against
        the amount of epoch iterations. Takes in a model as input.
        
        Parameters:
            model (torchvision.models) - model for plotting
            model_name (string) - name of model
        """        
        # Create plot
        epochs = range(len(model.train_losses))
        line1 = plt.plot(epochs, model.train_losses, label='training loss')
        line2 = plt.plot(epochs, model.valid_losses, label='validation loss')
        plt.xlabel("Iterations")
        plt.ylabel("Losses")
        plt.legend(loc="upper right")
        plt.title(f"{model_name} Loss Comparison")
    
    def plot_cm(self, model, model_name, y_pred, y_true):
        """
        Creates a confusion matrix for the given model.
        
        Parameters:
            model (torchvision.models) - models to evaluate
            model_name (string) - name of model
            y_pred (torch.Tensor) - test or validation loader predictions
            y_true (torch.Tensor) - dataloader labels
        """
        # Plot confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=self.class_labels)
        disp = plot_confusion_matrix(y_true, y_pred, figsize=(25, 15),
                                     labels=self.class_labels,
                                     x_tick_rotation=90,
                                     title=f"{model_name} Confusion Matrix")

    def plot_roc(self, model, model_name, y_probas, y_true, figsize=(25, 15)):
        """
        Creates multiple subplots of the ROC curve for each classes using the given model.
        
        Parameters:
            model (torchvision.models) - models to evaluate
            model_name (string) - name of model
            y_probas (torch.Tensor) - test or validation loader probabilities
            y_true (torch.Tensor) - dataloader labels
            figsize (tuple) - subplot figure size
        """
        plot_roc(y_true, y_probas, figsize=figsize, title=f"{model_name} ROC Plots")
    
    def plot_stats(self, model_stats):
        """
        Displays a table of the given models statistics.
        
        Parameters:
            model_stats (list/dictionary) - a list or single dict of statistics of trained models
        """
        # Set initial variables
        headers = ['Name', 'Accuracy', 'Top-1 Error', 'Top-5 Error', 
                   'Precision', 'Recall', 'F1-Score']
        table = pd.DataFrame(columns=headers)
        
        # Add rows if list
        if isinstance(model_stats, list):
            for i in range(len(model_stats)):
                table = table.append(model_stats[i], ignore_index=True)
        # Add row if single dict
        else:
            table = table.append(model_stats, ignore_index=True)
        
        return table