# Methodology

Replace this template with your own. Keep it short: write what a teammate would need to run your work and trust it.

## 1. How to run it

**0) Install Python**
```bash
brew install python
```


**1) Clone the repo and set up a virtual env**
```bash
git clone https://github.com/diegorosales06/YonderDynamicsTakeHome.git
cd YonderDynamicsTakeHome
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```


**2) Live webcam inference**

To run the live camera feed and yolo model on the weights provided. Run one of the commands:
```bash
python src/webcam_infer.py                  # default: 8nBest.pt, camera 0, conf 0.05
python src/webcam_infer.py --conf 0.25      # raise the confidence threshold
python src/webcam_infer.py --camera 1       # pick a different camera
```
Needs camera permission for the terminal (macOS: System Settings → Privacy & Security → Camera). Press `q` to quit, `s` to save a snapshot. The UI shows FPS and inference latency.


**3) Download the dataset from Roboflow**

The dataset isn't in the repo (it's git-ignored). You need a free Roboflow API key. Make a `.env` file in the repo root:
```
ROBOFLOW_API_KEY=your_key_here
```
To get the dataset from Roboflow, run the following file:
```bash
python src/dataSetDownload.py
```
This drops the dataset into `Sampled-YD-Object-Detection-2/`.

**4) (Optional) retrain the model**

Training was done in Google Colab on a free T4 GPU. Open `src/trainingNotebook.ipynb` in Colab and run top-to-bottom. The weights come out in `runs/detect/trainN/weights/best.pt`, which we copied into `src/modelWeights/`.

If you just want to look at the results, skip this step. The trained weights are already in `src/modelWeights/`. We are shipping `8nBest.pt` for the reasons discussed in Section 2.

**5) Run evaluation on the validation set**
```bash
python src/metrics.py
```
Writes per-class precision, recall, F1, mAP50, and mAP50-95 to `src/metrics.csv` and prints the same table to the console. Ultralytics also drops `confusion_matrix.png` and the PR / F1 curves into `runs/detect/val*/`.

**6) Find the model's mistakes on the val set**
```bash
python src/findFaults.py
```
Writes `src/errors/errors.txt` listing every image with a false positive, false negative, or misclassification, and saves the annotated prediction JPG for each of those images into `src/errors/`. This is what we used for the error analysis in Section 2.

## 2. Thought process

- Imagess within the datset are augmented variants of the same photos. Because of this, our validation numbers may be inflated, since we are essentialy testing the model on images it was already seen and "memorized". If this is true, our IRL test of the model will have low validation rates. 
- Don't want to apply too much preproccessing, since that preproccessing would have to be applied in realtime to the camera feed on the rover. Wan't to keep precocessing as limited as possible.

- Trained on a T4 GPU within colab

### Preprocessing (runs on every image, including on the rover)
- **Auto-Orient**: strips EXIF rotation metadata so images render consistently
- **Resize**: stretch image to 512×512. I chose 512 over YOLO's default 640 to cut inference cost on the onboard NPU (a 512 image has about ~40% fewer input pixels, so fast infererce) at a small accuracy cost

### Augmentation:
Applied via Roboflow with 3 augmented outputs per source image, pushing the 800 original training images to 2400 training images. Augmentations were chosen to simulate conditions the rover’s camera will realistically see, not to invent conditions it won’t.

**What we know our rover and camera are gonna experience in actual deployment:**
- Will see objects in direct sunlight or in shade possiblity even at night
- Debris/dust will get on the camera, potentially alter blocking the image and causing blur
- Our camera can experience motion blur at extreme speeds or bumpy terrain
- Camera static from a poor physical/network connection

**Augmentation to account for actual depoylment:**
- Brightness (Between -15% and +15%): gives the dataset exposure to objects being in different lighting enviroments
- Saturation (Between -25% and +25%): adds images that mimic harsh sunlight, overcast skies, fog, or artificial indoor light. Since lighting can dramatically change the vividness of the colors on an image
- Exposure (Between -12% and +12%): adds data on dramtic lighting shifts like direct sunlight or clouds
- Blur (Up to 1px): gives the data set images that mimic motion blur
- Noise (Up to 3.15% of pixels): account for camera static

### Data Split: 

- Kept the original dataset split: 800 train / 199 valid originally, which then became 2400 train / 199 valid after the 3× augmentation outputs got applied into the training side only
- Why keep the provided split: I kept it since it was already labeled and partitioned, and the dataset is small enough that rebuilding a split from scratch wouldn't meaningfully change some of the more important metrics. So, I spent that time on augmentation and error analysis instead, which I thought was a better trade.
- Augmentation is applied to the training set only. The 199 val images stay unmodified (besides preprocessing). This is standard practice, validation has to mirror what the model sees at real deployment, not changes in training data.
### Evaluation:

**How we want our model to behave:**
- In an ideal world, we would want our model to have a high recall and high precision. Our model would see every object and get their inference right every time. 
- In a real model, there is often a tradeoff between these 2 values. If you have a high recall, then the model has a lower senstivity to the classifcation. Having a high precision means that the model was more cuatious and predicted less images overall. So, finding everything often leads to more mistakes, while being ultra-cautious means missing things. 
- We would want a high mAP@50:90. This metric shows us that our bounding boxes we accurate at a range between .50 and .95 IoU (Intersection over Union) and had a high amount of overlap. This is critical in deployment with tasks like gripping or pushing, which will be disscused later. 

```
|               | precision | recall | mAP50 | mAP50-95 |
|---------------|-----------|--------|-------|----------|
| **8s all**    | 0.949     | 0.881  | 0.924 | 0.619    |
| **8n all**    | 0.949     | 0.896  | 0.926 | 0.630    |
| 8s bottle     | 0.946     | 0.808  | 0.880 | 0.611    |
| 8n bottle     | 0.926     | 0.829  | 0.866 | 0.591    |
| 8s mallet     | 0.952     | 0.953  | 0.968 | 0.626    |
| 8n mallet     | 0.972     | 0.962  | 0.986 | 0.669    |
```

**From these metrics:**
- The larger models did not win according to our metrics. The 8n model ties or beats 8s. For a rover that has limited compute on the NPU, having a model with less parameters (~3M) and faster inference time from this make it a great option for delopyment in compeition. **see notes on # of epochs**
- Both models are precise, but not confident at finding everything. Our precision for the models sits at round 0.95 and our recall at 0.88 - .90. So when they call a box they are right most of the time (from our precision metric), but they miss quite a bit of objects (from our recall metric). If we were to deploy these models in compeition, we would see a high amount of false negative and a low amount of false positives. 
- The mallet is the easier class the detect. Our 8s precison values on the mallet class are ~.03 higher, but this gap is massive when we take a look at recall. Our recall values on mallet are ~.15 higher than our bottle classes. This recall metric shows that our models are completely missing a significant amount of bottles and not registering them at all. This is likely because bottles vary more in shape, transparency, label, and look more like background clutter than mallets do.
- Across both models and classes, mAP@0.5-0.95 is signifcantly lower than mAP50. So the models can find the objects, but their bounding boxes are not tight around it (IoU). The mAP@0.5 and respective 0.5 IoU threshold shows that these models are making the proper detections, but this metric allows their bounding boxes and overlap to be sloppy. When we tighten up the threshold and restriction in mAP@0.5-0.95, we can see our sloppy bounding boxes start to get filtered out and create a much lower mAP. This matters for deployment in compeition for a grasping task, the rover has to know where to grab and exactly where an object sits. Having a low mAP@50:95 means that we can't accurately know where the object is within the fram, just that it exists.

**Things to note:**
-  The larger 8s model was trained on 71% the amount of epochs. 8s was trained on 50, and 8n was trained on 70 epochs. 
## 3. Known limitation


### Error analysis on the validation set: 
Run src/findFaults.py to get this validation set, will be under src/errors
The failures group into a few repeating patterns:
- Mallets had signicfalty more false positives than bottles. Taking a look at the images, essentialy every orange item was market as a mallet. 
- Transparent / clear bottles don't have a strong distinguishing silhouette against the background.
- Bottles that are tipped on their side or partially behind another object confuse the model

### Errors on IRL testing:
As seen in the demo video (https://youtu.be/VK30G-J6Fv8?si=llHJ_kCOxXkI9puL), the model starts to fail in backgrounds that are not rocky/tan. Green backgrounds or indoors enviroments cause the model to completely miss 

### What I would do next: 
1. Collect more bottle data, specifically the hard cases. Transparent bottles, bottles on their side, bottles in cluttered scenes. Our recall ceiling is bottle recall, and augmentation can't fix a lack of variety in source photos.
3. Add indoor + green backgrounds training data.The IRL test shows the model falls apart indoors and agasint backgrouds that are not tan/gray. If the rover will ever operate in a lab, hangar, or at dusk, this is a issue with deployment
4. Tighten the boxes. The mAP50 to mAP50-95 drop says the model knows roughly where objects are but not precisely. I could try a higher input resolution during training (though it would cost rover compute), or a tighter IoU threshold

### Things that currently don't work
- Model is unreliable indoors — training data is almost entirely outdoor scenes.
- Transparent / mostly-empty bottles get missed often.
- Bounding boxes aren't tight enough for a grasping task without additional refinement.