'''
Original code:
https://www.geeksforgeeks.org/building-a-generative-adversarial-network-using-keras/
'''
import numpy as np 
import matplotlib.pyplot as plt 
import keras 
from keras.layers import Input, Dense, Reshape, Flatten, Dropout 
from keras.layers import BatchNormalization, Activation, ZeroPadding2D 
from keras.layers.advanced_activations import LeakyReLU 
from keras.layers.convolutional import UpSampling2D, Conv2D 
from keras.models import Sequential, Model 
from keras.optimizers import Adam,SGD 
from keras.utils.vis_utils import plot_model

def build_generator(): 
    model = Sequential() 
    #Building the input layer 
    model.add(Dense(128 * 8 * 8, activation="relu", input_dim=latent_dimensions)) 
    model.add(Reshape((8, 8, 128))) 
    model.add(UpSampling2D()) 
    model.add(Conv2D(128, kernel_size=3, padding="same")) 
    model.add(BatchNormalization(momentum=0.78)) 
    model.add(Activation("relu")) 
    model.add(UpSampling2D()) 
    model.add(Conv2D(64, kernel_size=3, padding="same")) 
    model.add(BatchNormalization(momentum=0.78)) 
    model.add(Activation("relu")) 
    model.add(Conv2D(3, kernel_size=3, padding="same")) 
    model.add(Activation("tanh")) 
    #Generating the output image 
    noise = Input(shape=(latent_dimensions,)) 
    image = model(noise) 
    return Model(noise, image) 


def build_discriminator():
    #Building the convolutional layers to classify whether an image is real or fake
    model = Sequential() 
    model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=image_shape, padding="same")) 
    model.add(LeakyReLU(alpha=0.2)) 
    model.add(Dropout(0.25)) 
    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same")) 
    model.add(ZeroPadding2D(padding=((0,1),(0,1)))) 
    model.add(BatchNormalization(momentum=0.82)) 
    model.add(LeakyReLU(alpha=0.25)) 
    model.add(Dropout(0.25)) 
    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same")) 
    model.add(BatchNormalization(momentum=0.82)) 
    model.add(LeakyReLU(alpha=0.2)) 
    model.add(Dropout(0.25)) 
    model.add(Conv2D(256, kernel_size=3, strides=1, padding="same")) 
    model.add(BatchNormalization(momentum=0.8)) 
    model.add(LeakyReLU(alpha=0.25)) 
    model.add(Dropout(0.25)) 
    #Building the output layer 
    model.add(Flatten()) 
    model.add(Dense(1, activation='sigmoid')) 
    image = Input(shape=image_shape) 
    validity = model(image) 
    return Model(image, validity) 


def display_images(epoch,generated_images): 
    r, c = 4,4
    #Scaling the generated images 
    generated_images = 0.5 * generated_images + 0.5
    fig, axs = plt.subplots(r, c) 
    count = 0
    for i in range(r): 
        for j in range(c): 
            axs[i,j].imshow(generated_images[count, :,:,],interpolation='lanczos') 
            axs[i,j].axis('off')
            count += 1
    fig.tight_layout()
    #plt.show()
    fig.savefig('results_%d.png'%epoch)
    #plt.close()




if __name__=='__main__':
    #Loading the CIFAR10 data 
    (X, y), (_, _) = keras.datasets.cifar10.load_data() 
    #Selecting a single class images 
    #The number was randomly chosen and any number 
    #between 1 to 10 can be chosen
    category=8
    print('choose a category')
    X = X[y.flatten() == category] 
    #Defining the Input shape
    image_shape = (32, 32, 3) 
    latent_dimensions = 100

    ################################################################################
    ##### Build the Generative Adversarial Network ######
    ################################################################################
    # Build the discriminator 
    discriminator = build_discriminator()  
    #Making the Discriminator untrainable so that the generator can learn from fixed gradient 
    discriminator.trainable = False
 
    # Build the generator 
    generator = build_generator() 

    # Define the input for the generator and generating the images 
    z = Input(shape=(latent_dimensions,)) 
    image = generator(z)
    # Check the validity of the generated image 
    valid = discriminator(image)
    
    # Define the combined model of the Generator and the Discriminator 
    combined_network = Model(z, valid) 
    combined_network.compile(loss='binary_crossentropy', optimizer=Adam(0.0002,0.5))

    discriminator.trainable = True
    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002,0.5), metrics=['accuracy'])
  
    ##### Outputs #####
    # png
    plot_model(discriminator, to_file='geekGANdisc_model_plot.png', show_shapes=True, show_layer_names=True)
    plot_model(generator, to_file='geekGANgen_model_plot.png', show_shapes=True, show_layer_names=True)
    plot_model(combined_network, to_file='geekGAN_model_plot.png', show_shapes=True, show_layer_names=True)
    # json
    model_json = discriminator.to_json()
    with open("discriminator_geekGAN_model.json", "w") as json_file:
        json_file.write(model_json)
    model_json = generator.to_json()
    with open("generator_geekGAN_model.json", "w") as json_file:
        json_file.write(model_json)
    model_json = combined_network.to_json()
    with open("generator_containing_discriminator_geekGAN_model.json", "w") as json_file:
        json_file.write(model_json)
    ####################
    
    ################################################################################
    ##### Train the network ######
    ################################################################################
    num_epochs=15000
    batch_size=32
    display_interval=1000
    dlosses=[]
    glosses=[]
    #Normalize the input 
    X = (X / 127.5) - 1.
    #Define the Adversarial ground truths 
    valid = np.ones((batch_size, 1)) 
    #Add some noise 
    valid += 0.05 * np.random.random(valid.shape) 
    fake = np.zeros((batch_size, 1)) 
    fake += 0.05 * np.random.random(fake.shape)
    for epoch in range(num_epochs):

        #print('Epoch:',epoch)
        
        ### Train the Discriminator ###
        # Sample a random half of images 
        index = np.random.randint(0, X.shape[0], batch_size) 
        images = X[index] 
        # Sample noise and generating a batch of new images 
        noise = np.random.normal(0, 1, (batch_size, latent_dimensions))
        generated_images = generator.predict(noise) 
        # Train the discriminator to detect whether a generated image is real or fake 
        discm_loss_real = discriminator.train_on_batch(images, valid) 
        discm_loss_fake = discriminator.train_on_batch(generated_images, fake) 
        discm_loss = 0.5 * np.add(discm_loss_real, discm_loss_fake)
        #print('DiscriminatorLoss:',discm_loss[0])
        dlosses.append(discm_loss[0])
        
        ### Train the Generator ###
        # Train the generator to generate images which pass the authenticity test
        discriminator.trainable = False
        genr_loss = combined_network.train_on_batch(noise, valid)
        discriminator.trainable = True
        #print('GeneratorLoss',genr_loss)
        glosses.append(genr_loss)

        if epoch%100==0:
            print('Epoch:',epoch,'DiscriminatorLoss:',discm_loss[0],'GeneratorLoss:',genr_loss)

        # racking the progress				 
        if epoch % display_interval == 0:
            display_images(epoch,generated_images)
            
    display_images(num_epochs,generated_images)
    #combined_network.save_weights('combined_geekGAN.h5', True)
    #discriminator.save_weights('discriminator_geekGAN.h5', True)
    combined_network.save_weights('combined_geekGAN.h5')
    discriminator.save_weights('discriminator_geekGAN.h5')
    np.savetxt('dlosses.txt', dlosses, fmt='%f')
    np.savetxt('glosses.txt', glosses, fmt='%f')

    ################################################################################
    ##### Evaluate the performance  ######
    ################################################################################
    #Plotting some of the original images 
    s=X[:40] 
    s = 0.5 * s + 0.5
    f, ax = plt.subplots(5,8, figsize=(16,10)) 
    for i, image in enumerate(s): 
        ax[i//8, i%8].imshow(image,interpolation='lanczos') 
        ax[i//8, i%8].axis('off')
    f.tight_layout()
    #plt.show()
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
