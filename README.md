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
  * Plese use the data_prepare.py to create a .h file, which includes training data set.  
  * To train RDN using the train script simply specify the parameters listed in train.py as a flag or manually change them.  
    - python train.py --train-file trainPath --outputs-dir storePath
## Evalution
  * To evaluate a trained network:  
    - python test.py --weights-file weight --image-files imgPath
   * Then, it will generate the predicted high resolution images
