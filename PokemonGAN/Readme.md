## Generating Pokemons using DCGAN

Dataset used can be found here : ![Pokemon Images](https://www.kaggle.com/djilax/pkmn-image-dataset).
I converted all the images to 128x128 which the final generated image size as well. 

**To directly upload a dataset from kaggle to drive and use in colab** 

Go to https://www.kaggle.com/ **-> My Account -> Create New API token** - It will automatically download a json file. 
In colab, 
  - **Upload the token file**
  ```
  from google.colab import files
  files.upload()
  ```
 
  - **Connect the pathway**
  ```
  !pip install -q kaggle
  !mkdir -p ~/.kaggle
  !cp kaggle.json ~/.kaggle/
  !ls ~/.kaggle
  !chmod 600 /root/.kaggle/kaggle.json\
  ```
  
  - **List the datasets**
  ```
  !kaggle datasets list -s 'simpsons-faces'
  ```
  
  - **Mount your drive**
  ```
  from google.colab import drive
  import os
  drive.mount('/content/gdrive')
  ```
  
  - **Download dataset to a folder in your drive**
  ```
  !kaggle datasets download 'kostastokis/simpsons-faces' -p /content/gdrive/My\ Drive/Kaggle/simpson/
  ```


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
  
**Noise vector $z$**

The noise vector $z$ has the important role of making sure the images generated from the same class $y$ don't all look the same—think of it as a random seed. You generate it randomly, usually by sampling random numbers either between 0 and 1 uniformly, or from the normal distribution, which you can denote $z$ ~ $N(0, 1)$. The zero means the normal distribution has a mean of zero, and the 1 means that the normal distribution has a variance of 1. 

In reality, $z$ is usually larger than just 1 value to allow for more combinations of what $z$ could be. There's no special number that determines what works, but 100 is standard. Some researchers might use a power of 2, like 128 or 512, but again, nothing special about the number itself, just that it's large enough to contain a lot of possibilities. As a result, you would sample $z$ from that many different dimensions (constituting multiple normal distributions).

*Fun Fact: this is also called a spherical normal and denoted $z$ ~ $N(0, I)$ where the $I$ represents the identity matrix and means the variance is 1 in all dimensions.*

**Truncation trick**

So now that you're a bit familiar with noise vectors, here's another cool concept that people use to tune their outputs. It's called the truncation trick. I like to think of the truncation trick as a way of trading off fidelity (quality) and diversity in the samples. It works like this: when you randomly sample your noise vector $z$, you can choose to keep that random $z$ or you can sample another one. 

Why would you want to sample another one? 

Well, since I'm sampling $z$ from a normal distribution, my model will see more of those $z$ values within a standard deviation from the mean than those at the tails of the distribution—and this happens during training. This means that while the model is training, it's likely to be familiar with certain noise vectors and as a result model those areas coming from familiar noise vector regions. In these areas, my model will likely have much more realistic results, but nothing too funky, it's not taking as many risks in those regions mapped from those familiar noise vectors. This is the trade-off between fidelity (realistic, high quality images) and diversity (variety in images). 

<img src="https://build.openmodelica.org/Documentation/Modelica%203.2.3/Resources/Images/Math/Distributions/TruncatedNormal.density.png" alt="truncated normal distribution" width="400"/>

> *Image Credit: Modelica*


What the truncation trick does is resamples the noise vector $z$ until it falls within some bounds of the normal distribution. In fact, it samples $z$ from a truncated normal distribution where the tails are cut off at different values (red line in graph is truncated normal, blue is original). You can tune these values and thus tune fidelity/diversity. Recall that having a lot of fidelity is not always the goal—one failure mode of that is that you get one really real image but nothing else (no diversity), and that's not very interesting or successful from a model that's supposed to model the realm of all possible human faces or that of all possible coconuts—including that of a cat pouncing after a flying coconut (but with extremely low probability).


## My architecture for updated GAN:
```
Discriminator(
  (conv1): Sequential(
    (0): Conv2d(3, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
  )
  (conv2): Sequential(
    (0): Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (conv3): Sequential(
    (0): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (conv4): Sequential(
    (0): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (fc): Linear(in_features=4096, out_features=1, bias=True)
)

Generator(
  (fc): Linear(in_features=100, out_features=8192, bias=True)
  (deconv1): Sequential(
    (0): ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (deconv2): Sequential(
    (0): ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (deconv3): Sequential(
    (0): ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (deconv4): Sequential(
    (0): ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (deconv5): Sequential(
    (0): ConvTranspose2d(32, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  )
)
```
  
  
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

I trained the GAN for 500 epochs and some of the images generated are: 

![Image1](https://github.com/sanketsans/Deep-Learning/blob/master/PokemonGAN/generated%20images/image_123.jpg)
![Image2](https://github.com/sanketsans/Deep-Learning/blob/master/PokemonGAN/generated%20images/image_148.jpg)
![Image3](https://github.com/sanketsans/Deep-Learning/blob/master/PokemonGAN/generated%20images/image_166.jpg)
![Image4](https://github.com/sanketsans/Deep-Learning/blob/master/PokemonGAN/generated%20images/image_168.jpg)
![Image5](https://github.com/sanketsans/Deep-Learning/blob/master/PokemonGAN/generated%20images/image_169.jpg)
![Image6](https://github.com/sanketsans/Deep-Learning/blob/master/PokemonGAN/generated%20images/image_53.jpg)
![Image7](https://github.com/sanketsans/Deep-Learning/blob/master/PokemonGAN/generated%20images/image_94.jpg)

**Generation process:**

[![https://github.com/sanketsans/Deep-Learning/blob/master/PokemonGAN/generated%20images/image_169.jpg](http://img.youtube.com/vi/HWHVh9orOQk/0.jpg)](http://www.youtube.com/watch?v=HWHVh9orOQk "Generating Fake pokemon ")
