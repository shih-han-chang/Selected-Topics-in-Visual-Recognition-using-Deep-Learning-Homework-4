# Selected-Topics-in-Visual-Recognition-using-Deep-Learning-Homework-4

## Hardware   
  Use Linux with PyTorch to train this model  
  - PyTorch >= 1.1.  
  - Python >= 3.6.
  - Numpy 1.15.4
  - Pillow 5.4.1
  - h5py 2.8.0
  - tqdm 4.30.0
  - Other common packages.  

## Dataset 
  * The training data have 291 high resolution images.
  * The test data have 14 low resolution images. 
  
## Training model
  * Plese run the 
  * To train YOLACT using the train script simply specify the parameters listed in train.py as a flag or manually change them.  
    - python train.py --config=res101_custom_config
## Evalution
  * To evaluate a trained network:  
    - python eval.py --trained_model=res101_custom_100000.pth
  * Then, it will generate a.json file with test result
