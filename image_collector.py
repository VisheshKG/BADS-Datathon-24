import webbrowser
import pandas as pd
import dxcam
import numpy as np
import time
import cv2
import pyautogui


url = 'https://www.dropbox.com/scl/fi/xl6leojssqiz12a6g6e35/Atlanta_supply_dat.xlsx-UC_buildings.csv?rlkey=9t4h432b0d5160kivwut4wwyy&dl=1'
df = pd.read_csv(url)

zoom = 16
meters = 70000

main_cam = dxcam.create()

for index, warehouse in df.iterrows():
    print(warehouse)
    
    # get lat, long
    lat = warehouse['Latitude']
    lng = warehouse['Longitude']
    id = warehouse['PropertyID']

    # get urls
    google_maps_url = f"https://earth.google.com/web/@{lat},{lng},{meters}d"
    bing_maps_url = f"https://www.bing.com/maps?cp={lat}~{lng}&lvl={zoom}&style=h"    

    # screenshot

    webbrowser.open(google_maps_url)
    time.sleep(12)
    frame = main_cam.grab()
    cv2.imwrite(f'unlabeled_data/{id}_goog.jpg', frame)
    pyautogui.hotkey('ctrl', 'w')
    time.sleep(0.5)

    webbrowser.open(bing_maps_url)
    time.sleep(3)
    frame = main_cam.grab()
    cv2.imwrite(f'unlabeled_data/{id}_bing.jpg', frame)
    pyautogui.hotkey('ctrl', 'w')
    time.sleep(0.5)





