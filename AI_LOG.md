# AI Usage Log

Replace this template with your own entries. Add one entry per significant use of an AI tool.

## 1. <short title of what you used it for>

**What I asked:**
I asked AI what type of augmentations I should apply to the dataset.  
**What I kept vs. rewrote, and why:**
The AI model gave me a few suggestions but didn't give me an explanation as to why those augmentations were needed. The model didn't touch on what conditions the rover would experience in real deployment and how to account for it. 

**What the AI got wrong that I had to catch:**
The model suggested adding a rotation augmentation to the dataset, this was not required and not helpful to the dataset.


## 2. <next use>
**What I asked:**
Asked it to right an inference script using the trained models
**What I kept vs. rewrote, and why:**
I kept most of the script. There was quite a bit of fluff within the script itself. There was a lot of unnecessary parameters and functions for the script I ended up cutting to make it easier for the team to read through. 



## 3. <next use>
**What I asked:**
Asked how I could get the neccessary metrics from the trained model 

**What I kept vs. rewrote, and why:**
I kept the general @metrics.py script. But swapped around where some of the data being saved and printed was coming from. Specifcally with the 2 seperate classes. 

**What the AI got wrong that I had to catch:**
The AI combined both mallet and bottle metrics into a single value (recall, precison, mPA). We want these 2 values split for proper analysis and to see how the model responds to each class. 


## 4. <next use>
**What I asked:**
Gave AI a series of vidoes to get the average FPS and imference time. 
**What I kept vs. rewrote, and why:**
Kept the metrics given outputted after validation. 

**How I verified it (for example, checked the metric by hand, ran it on data it hadn't seen):**
Ran a script that measured the averge inference and FPS for ~10 second run. Took a video recording of that same run and had AI do the FPS and inference time calculations. The metrics from the script and AI were about the same. 


## 5. <next use>
**What I asked:**
Asked AI the write me a script that tests images within our /valid folder of our dataset to see which images are not accruately classified. 
**What I kept vs. rewrote, and why:**
I kept most of the file. I altered the final path of the photos found to make it more intuitive for users to find and see the photos. 
**How I verified it (for example, checked the metric by hand, ran it on data it hadn't seen):**
I verified the outputed script by going through about 1/2 of the photos within the src/errors