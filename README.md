# Wild Edible Plant Classifier

This repository focuses on a Wild Edible Plant Classifier that compares the performance of three state-of-the-art CNN architectures: MobileNet v2, GoogLeNet, and ResNet-34. The artefact created is part of my BSc dissertation and is aimed at classifying 35 classes of wild edible plants using Transfer Learning.

## Dataset

![Plant Classes](https://github.com/Achronus/wep-classifier/blob/main/imgs/plant-classes.png "Wild Edible Plant Classes")

The 35 classes of wild edible plants are listed in the table below, accompanied by the number of images (per class) within the dataset. The dataset created is comprised of Flickr images, obtained through their API using the rudimentary scripts within the `\data_gathering` folder. The dataset can be found on Kaggle [here](https://www.kaggle.com/ryanpartridge01/wild-edible-plants/) and contains 16,535 images, where the quantity of images per class varies from 400 to 500.

<table>
<tr><td>

|Class|Quantity|
|-----|--------|
|Alfalfa|470|
|Allium|481|
|Borage|500|
|Burdock|460|
|Calendula|500|
|Cattail|466|
|Chickweed|488|
|Chicory|500|
|Chive blossoms|455|
|Coltsfoot|500|
|Common mallow|439|
|Common milkweed|469|

</td><td>

|Class|Quantity|
|-----|--------|
|Common vetch|451|
|Common yarrow|474|
|Coneflower|500|
|Cow parsley|500|
|Cowslip|442|
|Crimson clover|400|
|Crithmum maritimum|433|
|Daisy|490|
|Dandelion|500|
|Fennel|452|
|Fireweed|500|
|Gardenia|500|

</td><td>

|Class|Quantity|
|-----|--------|
|Garlic mustard|409|
|Geranium|500|
|Ground ivy|408|
|Harebell|500|
|Henbit|500|
|Knapweed|500|
|Meadowsweet|456|
|Mullein|500|
|Pickerelweed|454|
|Ramsons|489|
|Red clover|449|
|_Total_|_16,535_|

</td></tr>
</table>

## File Structure

![ML Pipeline](https://github.com/Achronus/wep-classifier/blob/main/imgs/ml-pipeline.png "Machine Learning Pipeline")

The file structure used for the artefact is outlined below and has helped to achieve the Machine Learning (ML) pipeline illustrated above.

``` ANSI
.
+-- data_gathering
|   +-- 1. get_urls.py
|   +-- 2. get_images.py
|   +-- get_filenames.py
|   +-- img_rename.py
|   +-- resize_images.py
+-- functions
|   +-- model.py
|   +-- plotting.py
|   +-- utils.py
+-- saved_models
|   +-- googlnet_model.pt
|   +-- mobilenetv2_model.pt
|   +-- resnet34_model.pt
+-- 1. wep_classifier_initial.ipynb
+-- 2. wep_classifier_tuning.ipynb
+-- requirements.txt
```

As mentioned earlier, the `\data_gathering` folder outlines scripts that were used to gather and prepare the Flickr image data (2, 3 in ML pipeline). The `\functions` folder holds the classes and functions used to perform all the functionality of the artefact. This covers all the remaining parts of the pipeline (4 -> 7).

The artefact code is run within the two Jupyter Notebooks - `1. wep_classifier_initial.ipynb` (steps 4 -> 6) and `2. wep_classifier_tuning.ipynb` (step 7).

## Dependencies

This project requires a Python 3 environment, which can be created by following the instructions below.

1. Create (and activate) a new environment.

   - Linux or Mac

    ```bash
    conda create --name wep
    source activate wep
    ```

   - Windows

   ```bash
   conda create --name wep
   activate wep
   ```

2. Clone the repository, navigate to the `wep-classifier/` folder and install the required dependencies.

    _(Note)_ a requirements.txt file is accessible within this folder which details a list of the required dependencies.

    ```bash
    git clone https://github.com/Achronus/wep-classifier.git
    cd wep-classifier
    conda install -c conda-forge jupyterlab
    conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
    pip install -r requirements.txt
    ```

3. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `wep` environment.

    ```bash
    python -m ipykernel install --user --name wep --display-name "wep"
    ```

4. Run the `jupyter-lab` command to start JupyterLab and access the Jupyter Notebooks.

A list of the used packages and versions are highlighted in the table below.

<table>
<tr><td>

</td><td>

|Package|Version|
|-------|-------|
|Python|3.9.2|
|Matplotlib|3.3.4|
|Numpy|1.20.1|
|Torch|1.7.0|
|Torchvision|0.8.1|

</td><td>

|Package|Version|
|-------|-------|
|Scipy|1.6.0|
|Pandas|1.2.2|
|Scikit-learn|0.2.4|
|Scikit-plot|0.3.7|
|_Total packages_|_9_|

</td></tr>
</table>

## References

This section highlights useful documentation links relevant to the project.

- [PyTorch Models](https://pytorch.org/vision/0.8/models.html)
- [PyTorch Datasets](https://pytorch.org/vision/0.8/datasets.html)
- [PyTorch - Load and Save Models](https://pytorch.org/tutorials/beginner/saving_loading_models.html)
- [Scikitplot Metrics](https://scikit-plot.readthedocs.io/en/stable/metrics.html)
