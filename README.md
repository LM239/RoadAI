# RoadAI

Team NoDig's code for the RoadAI competition.

[GitHub repository](https://github.com/LM239/RoadAI)

# Prerequisite
- Python >= 3.11.1


## (Optional) Devcontainer requirements
- VSCode
- VSCode Dev Containers extension
- Docker

# General Setup
1) Clone project
2) Download GPS data from [RoadAI Google Disk](https://drive.google.com/drive/folders/1_NEoph7pBfK36pVU16cwOh8r6PpkBvwV) and extract it to the folder `data/GPSData`

## Standard setup
Either
- Run `pip install -r requirements.txt` to install all dependencies
  
Or
- Install jupyter separately, and run the notebooks which will install the remaining dependencies themselves.

## Devcontainer setup
1) Start Docker
2) Open VSCode
3) Open in devcontainer by clicking `View` -> `Command Palette` -> `Dev Containers: Rebuild and Reopen in container`
4) Open a terminal in the project
5) Run the command `jupyter notebook --allow-root`
6) Click on link to open jupyter notebook in the browser
7) Visit and run notebooks

# Article submitted to Nordic Machine Intelligence (NMI)

- [PDF](https://nrk.no)
- [Tex file](https://vg.no)

# Visual presentation

- [HTML](https://lm239.github.io/RoadAI/visual_presentation/)
# Notebooks
- Automated load and dump detection with LightGBM
  - [PDF](https://lm239.github.io/RoadAI/load_dump_lightgbm_demo.pdf)
  - [HTML](https://lm239.github.io/RoadAI/load_dump_lightgbm_demo)
  - [Code](https://github.com/LM239/RoadAI/blob/main/load_dump_lightgbm_demo.ipynb)
- Daily report
  - [PDF](https://lm239.github.io/RoadAI/daily_report_demo.pdf)
  - [HTML](https://lm239.github.io/RoadAI/daily_report_demo)
  - [Code](https://github.com/LM239/RoadAI/blob/main/daily_report_demo.ipynb)

