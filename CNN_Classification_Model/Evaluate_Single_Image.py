import cv2, os
import numpy as np
from keras.models import load_model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
target_size = (56, 56)
pieceTypeList = ['b_jiang','b_ju', 'b_ma', 'b_pao', 'b_shi', 'b_xiang', 'b_zu',
		'r_bing', 'r_ju', 'r_ma', 'r_pao', 'r_shi', 'r_shuai', 'r_xiang']
pieceTypeList_with_grid = ['b_jiang','b_ju', 'b_ma', 'b_pao', 'b_shi', 'b_xiang', 'b_zu', 'grid',
                'r_bing', 'r_ju', 'r_ma', 'r_pao', 'r_shi', 'r_shuai', 'r_xiang']
label_type = pieceTypeList_with_grid
# The things you need to change is in here
weights = '/Users/jartus/Chinese-Chess/Temporary_Model/cnn_mini.h5'
#weights = '/Users/jartus/Chinese-Chess/h5_file/new_model_v2.h5'
file_path = '/Users/jartus/Chinese-Chess/Dataset/finetune/'
file_path = '/Users/jartus/Chinese-Chess/Dataset/train/b_ju/b_ju_285.png'

def evaluate_one(weights, file_path):
    """
    Function to evaluate a single image file.
    This function takes the path to a single image file and uses a pre-trained model to
    predict the type of chess piece in the image.
    Args:
        weights (str): The path to the pre-trained model weights.
        file_path (str): The path to the image file.
    """
    model = load_model(weights)
    x = cv2.imread(file_path)
    x = cv2.resize(x, target_size)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = x / 255
    x = np.expand_dims(x, axis=0)
    preds = model.predict_classes(x)
    print(label_type[int(preds)])
def evaluate(weights, file_path):    # Input is a directory, Output is the total number and error number
    """
    Function to evaluate all image files in a directory.
    This function takes a directory path and iterates over all the image files in the
    directory, using a pre-trained model to predict the type of chess piece in each image.
    Args:
        weights (str): The path to the pre-trained model weights.
        file_path (str): The path to the directory containing the image files.
    """
    model = load_model(weights)
    for i in os.listdir(file_path):
        if i == '.DS_Store':
            continue
        x = cv2.imread(file_path + i)
        x = cv2.resize(x, target_size)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        x = np.expand_dims(x, axis=0)
        preds = model.predict(x)
        preds = np.around(preds)
        print(label_type[np.where(preds[0] == 1)[0][0]])

evaluate(weights, file_path)
