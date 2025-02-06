## Installation
It is recommended that the user install Anaconda or Miniconda to create a virtual environment and use this package in the virtual environment. To install Miniconda, see https://docs.anaconda.com/miniconda/install/.

Create an environment called *r_peak_detection*. In Anaconda Powershell Prompt, type
```bash
conda create -n r_peak_detection python=3.10
```

After the environment is created, activate it by typing
```bash
conda activate r_peak_detection
```

To install the package, download the code to your computer. Navigate to the folder r_peak_detection, then type
```bash
pip install -e .
```
Don't omit the "." at the end in the line above!

## Usage
### Preparation
Activate your environment first.
```bash
conda activate r_peak_detection
```
### Running The Command
To label the R peaks in a ECG mat file, use the command `detect-r-peaks` (which will be automatically set up after you followed the [Installation](###Installation) step) then give it the path to the mat file. for example, run
```bash
detect-r-peaks C:\Users\yzhao\python_projects\r_peak_detection\data\F26C_07112023_signals.mat
```
After running it, you will see the output
```bash
Finding peaks...
Validating peaks...
100%|█████████████████████████████████████████████████████████████████████████████| 1130/1130 [00:02<00:00, 415.43it/s]
R-peak indices saved to C:\Users\yzhao\python_projects\r_peak_detection\data\F26C_07112023_signals.mat.
```
The results will be saved to the same mat file. The indices of R-peaks can be found in the field called `good_r_peaks`.
#### Optional Arguments
1. `-threshold`

`-threshold` sets how confident the model needs to be to consider a peak as a good R-peak. The higher the value, the fewer good R-peaks you may see. The default decision threshold is *0.5*. If you want to use a different threshold, change it to the number (between *0* and *1*) you like, for example, *0.6*. Your threshold value will be remembered, so the next time you don't have to supply it again if you will be using the same number. Example command:
```bash
detect-r-peaks C:\Users\yzhao\python_projects\r_peak_detection\data\F26C_07112023_signals.mat -threshold 0.6
```

2. `-checkpoint_path`

`-checkpoint_path` sets the model to use. You can drag and drop the checkpoint file, ie., the *.pt* file into the commandline as a shortcut to get the path to the checkpoint file. Your model choice will also be remembered, so the next time you don't have to supply it again if you will be using the same model. Example command:
 ```bash
 detect-r-peaks C:\Users\yzhao\python_projects\r_peak_detection\data\F26C_07112023_signals.mat -checkpoint_path C:\Users\yzhao\python_projects\r_peak_detection\checkpoints\r_peak_classifier_large_out1_32_out2_64_bs_32_dt_0.5.pth
 ```