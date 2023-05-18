# UNet

This project is a Python implementation of UNet model for image segmentation. 

## Getting Started

### Prerequisites
- Python 3.6+
- PyTorch 1.8+
- tqdm
- numpy
- matplotlib
- albumentations

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/Eponeshnikov/UNet.git
   ```
2. Install required packages using pip
   ```sh
   pip install -r requirements.txt
   ```

## Usage
1. Download data from https://www.kaggle.com/datasets/kumaresanmanickavelu/lyft-udacity-challenge
2. Add your data to `data` directory in the following structure: 
    ```
    data/
    ├── Cars/
        ├── dataA/
            ├── dataA/
                ├── images/
                    ├── img1.jpg
                    └── ...
                ├── masks/
                    ├── mask1.png
                    └── ...
            ├── valA/
                ├── images/
                    ├── img1.jpg
                    └── ...
                ├── masks/
                    ├── mask1.png
                    └── ...
        ├── dataB/
            ├── ...
        ├── dataC/
            ├── ...
        ├── dataD/
            ├── ...
        ├── dataE/
            ├── ...
    ```
2. Modify the `data_dir` variable in the code to match your data directory path.
3. Run ```main.ipynb``` using jupyter-notebook.
