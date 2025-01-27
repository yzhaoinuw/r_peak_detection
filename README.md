### Installation
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

### Usage
Activate your environment first.
```bash
conda activate r_peak_detection
```

To label the R peaks in a ECG mat file, use the command `detect-r-peaks` (which will be automatically set up after you followed the [Installation](###Installation) step) then give it the path to the mat file. for example, run
```bash
detect-r-peaks C:\Users\yzhao\python_projects\r_peak_detection\data\F26C_07112023_signals.mat
```
You will see the output
```bash
Finding peaks...
Validating peaks...
R-peak indices saved to C:\Users\yzhao\python_projects\r_peak_detection\data\F26C_07112023_signals.mat.
```
The results will be saved to the same mat file. The indices of R-peaks can be found in the field called `good_r_peaks`.  