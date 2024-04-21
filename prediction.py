from PIL import Image
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import os
import re

mean = None
std = None
model = None
device = None


class ConvNet1(nn.Module):
    def __init__(self):
        super(ConvNet1, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.fc1 = nn.Linear(16 * 16 * 64, 128) 
        self.fc2 = nn.Linear(128,64) 
        self.fc3 = nn.Linear(64,32) 
        self.fc4 = nn.Linear(32, 5) 

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def load_model(model_path, model_class, *args, **kwargs):
    model = model_class(*args, **kwargs)
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    return model

def clean_id(input_string):
    cleaned_string = re.sub(r'[^\d]+$', '', input_string)
    return cleaned_string

def get_time_left(stage,size):
    multiplier = 0
    quarters = 0

    match stage:
        case 0:
            multiplier = 0
        case 1:
            multiplier = 0.25
        case 2:
            multiplier = 0.45
        case 3:
            multiplier = 0.85
        case 4:
            multiplier = 1
    
    if size <= 99999:
        quarters = 2
    elif size <= 299999:
        quarters = 3
    elif size <= 599999:
        quarters = 4
    elif size <= 999999:
        quarters = 5
    else:
        quarters = 6

    return (1-multiplier) * quarters

def process_image(image_path):
    image = Image.open(image_path)
    image = image.convert('RGB').resize((128, 128))
    image = (np.array(image) - mean) / std
    image_tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0)
    return image_tensor

def predict(image_tensor):
    with torch.no_grad(): 
        output = model(image_tensor.to(device))
        _, predicted = torch.max(output, 1) 
    return predicted.item()

def finish_q_prediction(img_id,time_left_prediction,df_info):
    ground_broken = df_info.loc[df_info['PropertyID'] == img_id, 'YearQuarterGroundBroken'].values[0]
    w = [4,1]
    ground_broken = sum([(int(val)*w[i]) for i,val in enumerate(str(ground_broken).split('.'))])
    est_finish_q = ground_broken + time_left_prediction
    if est_finish_q % 4 == 0:
        est_finish_q = str(est_finish_q//4 - 1) + '.4'
    else:
        est_finish_q = str(est_finish_q//4) + '.' + str(est_finish_q%4)
    return est_finish_q

def run_prediction(image_path,df_info):
    image_tensor = process_image(image_path)
    stage_prediction = predict(image_tensor)
    img_id = int(clean_id(''.join(list(image_path.split('\\')[-1])[:10])))
    time_left_prediction = get_time_left(stage_prediction,df_info.loc[df_info['PropertyID'] == img_id, 'Size_sf'].values[0])
    # round down since our thresholds are upper bounds
    time_left_prediction = int(time_left_prediction)
    return finish_q_prediction(img_id,time_left_prediction,df_info)


def start_prediction(image_path,df_info):
    global mean
    global std
    global model
    mean = np.load('mean.npy')
    std = np.load('std.npy')
    model = load_model('cnn_model.pth', ConvNet1)
    return run_prediction(image_path,df_info)

def main():
    # Example of how to run the prediction

    # image path MUST contain property ID and must be in the form */{propertyID}*.jpg
    # df is the dataframe containing the information about the properties, we need this to get the size of the property
    img_path = 'labeled_data/0/1652880652_bing.jpg'
    df = pd.read_csv('https://www.dropbox.com/scl/fi/xl6leojssqiz12a6g6e35/Atlanta_supply_dat.xlsx-UC_buildings.csv?rlkey=9t4h432b0d5160kivwut4wwyy&dl=1')

    print(start_prediction(img_path,df))
    
# use this to iterate over multiple images
# please contact GOTWIC on discord or sr6474@nyu.edu if you need help with this
def main2():
    df = pd.read_csv('https://www.dropbox.com/scl/fi/xl6leojssqiz12a6g6e35/Atlanta_supply_dat.xlsx-UC_buildings.csv?rlkey=9t4h432b0d5160kivwut4wwyy&dl=1')\
    # create a new column called 'EstimatedFinishQuarter' and fill it with the estimated finish quarter
    df['CompletionYearQuarter'] = None
    for folder in range(0, 5):
        folder_path = os.path.join('labeled_data', str(folder))
        for filename in os.listdir(folder_path):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                file_path = os.path.join(folder_path, filename)
                est_finish_q = start_prediction(file_path,df)
                img_id = int(clean_id(''.join(list(file_path.split('\\')[-1])[:10])))
                df.loc[df['PropertyID'] == img_id, 'CompletionYearQuarter'] = est_finish_q
    
    df.to_csv('EstimatedFinishQuarter.csv', index=False)


if __name__ == '__main__':
    main()