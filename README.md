
# Scene Classification Challenge

This repository contains the source code for a machine learning project that addresses the Scene Classification Challenge. The project aims to classify images of natural scenes around the world into categories such as buildings, forests, glaciers, mountains, sea, and streets.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Before running the project, you need to install the required Python packages:

```bash
pip install -r requirements.txt
```

### Downloading the Data

The dataset used in this project is not included in the repository due to its large size. You can download the dataset from the following links:

- Train data: [Download train data](https://datahack.analyticsvidhya.com/contest/assignment-scene-classification-challenge/download/train-file)
- Test data: [Download test data](https://datahack.analyticsvidhya.com/contest/assignment-scene-classification-challenge/download/test-file)

After downloading, please place the datasets in the `data` directory of this project.

### Project Structure

The project has the following directory structure:

- `data/`: Directory containing the dataset (you need to download the dataset and place it here).
- `myenv/`: Python virtual environment directory (if used).
- `notebook/`: Jupyter notebooks with exploratory code and experiments.
- `source/`: Source code of the project.

### Running the Code

To run the main script, navigate to the `source` directory and execute the Python script:

```bash
python scene_classification.py
```

Please note that the script uses local file paths to access the data. You will need to replace these with the paths where you have stored your data.

## Authors

Paramanand Bhat.

## License

This project is licensed under the MIT License.
