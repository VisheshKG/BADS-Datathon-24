from PIL import Image
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

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

def run_prediction(image_path,df_info):
    image_tensor = process_image(image_path)
    stage_prediction = predict(image_tensor)
    img_id = int(''.join(list(image_path.split('/')[-1])[:10]))
    time_left_prediction = get_time_left(stage_prediction,df_info.loc[df_info['PropertyID'] == img_id, 'Size_sf'].values[0])
    print(f'Stage Prediction: {stage_prediction} | Time Left Prediction: {time_left_prediction} quarters')


def main():
    global mean
    global std
    global model
    mean = np.load('mean.npy')
    std = np.load('std.npy')
    model = load_model('cnn_model.pth', ConvNet1)

    # Example of how to run the prediction
    img_path = 'labeled_data/0/1652880652_bing.jpg'
    df = pd.read_csv('https://www.dropbox.com/scl/fi/xl6leojssqiz12a6g6e35/Atlanta_supply_dat.xlsx-UC_buildings.csv?rlkey=9t4h432b0d5160kivwut4wwyy&dl=1')

    # image path MUST contain property ID and must be in the form */{propertyID}*.jpg
    # df is the dataframe containing the information about the properties, we need this to get the size of the property
    run_prediction('labeled_data/0/1652880652_bing.jpg',df)


if __name__ == '__main__':
    main()