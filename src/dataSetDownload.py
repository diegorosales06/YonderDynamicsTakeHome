from roboflow import Roboflow
from dotenv import load_dotenv
import os
load_dotenv()

# This script will download the processed dataset I created on Roboflow. https://app.roboflow.com/gimbletrainingset/yonderdynamicstakehome/2

rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
project = rf.workspace("gimbletrainingset").project("yonderdynamicstakehome")
version = project.version(2)
dataset = version.download("yolov8")
