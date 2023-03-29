# SBTNet
Our solution in competition NTIRE 2023 Bokeh Effect Transformation: https://codalab.lisn.upsaclay.fr/competitions/10229.

## News
- *2023/3/29:* Update the test results and the pretrained model (**disable AlphaNet while testing real-world images**).

## Test Results
Download the test results from [Google Drive](https://drive.google.com/drive/folders/1ZTwTKC-NOEPne38cWRrrzweHBbZNItFB?usp=share_link).

## Usage
Download the pretrained model from [Google Drive](https://drive.google.com/drive/folders/1ZTwTKC-NOEPne38cWRrrzweHBbZNItFB?usp=share_link), and place it in the folder `checkpoints`. 
Run the following code to generate test results.
```
python evaluation.py --root_folder 'TEST_ROOT_FOLDER' --save_folder 'SAVE_FOLDER'
```
- `root_folder`:  root folder of the test dataset.
- `save_folder`: folder to save the results.
