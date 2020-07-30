# Pose-Guided Person Image Synthesis in the Non-iconic Views

This is the code for TIP paper 'Pose-Guided Person Image Synthesis in the Non-iconic Views'. To run this code, you need to do as follows,

## Prepare the environment
This code needs pytorch 1.4.0.

## Download Datasets
We provide the processed frames, poses and segmentations for **Penn Action** and **BBC-Pose**. Market-1501 is the same as the one used in [PATN](https://github.com/tengteng95/Pose-Transfer).

## Train
```{python}
python pose_to_image.py -n debug -R -nl -dp -d {Penn_Action/bbcpose/market1501} -dr {your_data_root}
```

## Generate
```{python}
python inference.py --dr {your_data_root} -n {name of model} -dp -nl
```
Then you will have the generated images in ```./result_images/{dataset}/{name of model}```.

## Evaluate
We provide several metrics in ```./metrics``` including Inception Score, FID, SSIM and M-SSIM.

You need to modify the variable ```dataset``` and ```root``` to your own setting, and run
```{python}
python inception_score.py
python compute_ssim.py
python fid_score.py -c {gpu}
```
The scripts will evaluate the generated images from all models in the chosen dataset.
