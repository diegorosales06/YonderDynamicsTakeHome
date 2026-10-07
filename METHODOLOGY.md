# Methodology

Replace this template with your own. Keep it short: write what a teammate would need to run your work and trust it.

## 1. How to run it

A reviewer should be able to follow this from a fresh clone without asking you anything. Test it yourself in a clean checkout before you submit.

## 2. Thought process


- Imagess within the datset are augmented variants of the same photos. Because of this, our validation numbers may be inflated, since we are essentialy testing the model on images it was already seen and "memorized". If this is true, our IRL test of the model will have low validation rates. 


- Don't want to apply too much preproccessing, since that preproccessing would have to be applied in realtime to the camera feed on the rover. Wan't to keep precocessing as limited as possible. 
# Preprocesing/Augmentation: 
- Image dataset is already in 640x640
- Rover may see objects in direct sunligh or in shade. Changing HSV values can expand the dataset to help combat the different lighting 
- Motion blur: A rovers camera can have motion blue, I clean training set does not. 
- Salt and Pepper Noise: Camera static can opccur 
- Dust on camera


1.5x px blur
noise 3%

Rover would accumlate dust on the camera, camera static could occur, rover cameras have motion blur, objects can be sunlight or shade. 


- Look at mAP
- Does not work on close objects?
- Model fails indoors? WORKS OUTDOORS!

What metrics to evaulate this by: 
- Is it in the proper conditions?
- Does it fail on enviroments that it WILL see in deployment?

## 3. Known limitations

What doesn't work, and what you'd do next. Being upfront counts in your favor.
