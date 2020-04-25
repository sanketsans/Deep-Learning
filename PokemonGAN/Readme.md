## Generating Pokemons using DCGAN

Dataset used can be found here : ![Pokemon Images](https://www.kaggle.com/kvpratama/pokemon-images-dataset)

Paper refered for the algorithm and hyper-parameter tuning - ![Paper](https://arxiv.org/pdf/1511.06434.pdf)

## Model 
A GAN is comprised of two adversarial networks, a discriminator and a generator.

## Discriminator 
It is responsible for labeling the images as real or fake. 
For the Discriminator model, 
  - The input images are 32x32x3.
  - Convolutional Hidden Layers with leaky realu activation with 0.2 leak.
  - A fully connected layer with no activation function, since we are using **BCEWithLogitLoss** loss function - it already 
  includes sigmoid and cross-entropy method.
  - I also used batch normalization with nn.BatchNorm2d on each layer except the first convolutional layer and final, linear output layer.
  - Make sure not to use Maxpool layers in convolution - instead use a **kernel_size of 4** and a **stride of 2** for strided convolutions.
  
  ![Architecture](https://github.com/sanketsans/Deep-Learning/blob/master/PokemonGAN/Images/conv_discriminator.png)
  
  
## Generator
It is responsible for generating the fake images. 
For the generator model, 
  - Use a noise vector z of size 100. 
  - The first layer is a fully connected layer which is reshaped into a deep and narrow layer, something like 4x4x512.
  - Then, we use batch normalization and a leaky ReLU activation. 
  - Next is a series of transpose convolutional layers, where we typically half the depth and double the width and height of the previous layer. 
  - And, we'll apply batch normalization and ReLU to all but the last of these hidden layers. Where we will just apply a tanh activation.
  - Used a **kernel size of 4** and **stride of 2**
  
  ![Architecture](https://github.com/sanketsans/Deep-Learning/blob/master/PokemonGAN/Images/conv_generator.png)
  
  
## Discriminator and Generator Losses
  
### Discriminator Losses

> * For the discriminator, the total loss is the sum of the losses for real and fake images, `d_loss = d_real_loss + d_fake_loss`. 
> * We want the discriminator to output 1 for real images and 0 for fake images.

The losses will by binary cross entropy loss with logits, which we can get with [BCEWithLogitsLoss](https://pytorch.org/docs/stable/nn.html#bcewithlogitsloss). This combines a `sigmoid` activation function **and** and binary cross entropy loss in one function.

### Generator Loss

The generator loss will look similar only with flipped labels.In this case, the labels are **flipped** to represent that the generator is trying to fool the discriminator into thinking that the images it generates (fakes) are real! or you can use real loss for generated images. 

---
## Training

Training will involve alternating between training the discriminator and the generator.

### Discriminator training
1. Compute the discriminator loss on real, training images        
2. Generate fake images
3. Compute the discriminator loss on fake, generated images     
4. Add up real and fake loss
5. Perform backpropagation + an optimization step to update the discriminator's weights

### Generator training
1. Generate fake images
2. Compute the discriminator loss on fake images, using **flipped** labels!
3. Perform backpropagation + an optimization step to update the generator's weights

