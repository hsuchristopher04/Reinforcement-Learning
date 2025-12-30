1. Clone or download the project folder
Make sure all the following files are in the same directory:
 - decision_tree.py
 - decision_tree_classification.py
 - weights.txt 
 - MNIST dataset (expected in ./datasets/MNIST/raw)
2. Install requirements.txt using:
 - pip install -r requirements.txt
3. Download the MNIST dataset
 - Place the files in: ./datasets/MNIST/raw
4. Run the Decision Tree classification script
 - python decision_tree_classification.py
This script will:
 - Load the MNIST dataset
 - Load the pre-trained linear weights (weights.txt)
 - Generate 45-dimensional features for each image
 - Train decision trees with different numbers of leaves
 - Evaluate train and test accuracy
 - Plot accuracy vs number of leaves
 5. View results
