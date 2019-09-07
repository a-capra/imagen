import numpy as np 
import matplotlib.pyplot as plt
import keras
from keras.models import model_from_json
import tensorflow as tf

################################################################################
##### Evaluate the performance  ######
################################################################################

if __name__=='__main__':

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    keras.backend.tensorflow_backend.set_session(tf.Session(config=config))

     #Loading the CIFAR10 data 
    (X, y), (_, _) = keras.datasets.cifar10.load_data() 
    #Selecting a single class images 
    #The number was randomly chosen and any number 
    #between 1 to 10 can be chosen
    category=3 # cats
    print('choose a category',category)
    X = X[y.flatten() == category]
    #Normalize the input 
    X = (X / 127.5) - 1.
    latent_dimensions = 100

    
    #Plotting some of the original images 
    s=X[:40] 
    s = 0.5 * s + 0.5
    f, ax = plt.subplots(5,8, figsize=(16,10)) 
    for i, image in enumerate(s): 
        ax[i//8, i%8].imshow(image,interpolation='lanczos') 
        ax[i//8, i%8].axis('off')
    f.tight_layout()
    #plt.show()
    

    ##### LOAD the GENERATOR #####
    with open("generator_geekGAN_model.json", "r") as jsonmodel:
        json_string=jsonmodel.readline()
    generator = model_from_json(json_string)
    generator.load_weights('generator_geekGAN.h5')
    
    #Plotting some of the last batch of generated images 
    noise = np.random.normal(size=(40, latent_dimensions)) 
    generated_images = generator.predict(noise) 
    generated_images = 0.5 * generated_images + 0.5
    f, ax = plt.subplots(5,8, figsize=(16,10)) 
    for i, image in enumerate(generated_images): 
        ax[i//8, i%8].imshow(image,interpolation='lanczos') 
        ax[i//8, i%8].axis('off')
    f.tight_layout()
    plt.show() 
