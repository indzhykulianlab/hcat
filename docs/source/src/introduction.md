# HCAT - Hair Cell Analysis Toolbox

Hcat is a suite of machine learning enabled algorithms for performing common image analyses in the hearing field. 


---
## Quickstart Guide
#### Installation
1) Install [Anaconda](https://www.anaconda.com/)
2) Perform the installation by copying and pasting the following comands into your *Anaconda Prompt* (Windows) or *Terminal* (Max/Linux)
3) Create a new anaconda environment: `conda create -yn hcat python=3.9`
4) Activate the anaconda environment: `conda activate hcat`
```{warning}
You will need to avtivate your conda environment every time you restart your prompt!
```

5) Install pytorch for CPU ONLY: `conda install pytorch torchvision torchaudio cpuonly -c pytorch`
6) Install hcat and dependencies: `pip install hcat --upgrade` 
7) Run hcat: `hcat`

```{note}
Follow the detailed installation guide for instructions on how to enable GPU acceleration 
```

#### Analysis
Detection Gui:
* Run in terminal: `hcat`

CLI Hair Cell Detection Analysis:
* Run in terminal: `hcat detect "path/to/file.tif"`

---