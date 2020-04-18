### Style Transfer with Deep Neural Networks

In this project, style transfer uses the features found in the 19-layer VGG Network, which is comprised of a series 
of convolutional and pooling layers, and a few fully-connected layers. In the image below, the convolutional layers are 
named by stack and their order in the stack. Conv_1_1 is the first convolutional layer that an image is passed through, in the
first stack. Conv_2_1 is the first convolutional layer in the second stack. The deepest convolutional layer in the network is 
conv_5_4.

<img src='https://github.com/sanketsans/Deep-Learning/blob/master/style_transfer/notebook_ims/vgg19_convlayers.png' width=80% />
Separating Style and Content

Style transfer relies on separating the content and style of an image. Given one content image and one style image, we aim to create a new, target image which should contain our desired content and style components:

    objects and their arrangement are similar to that of the content image
    style, colors, and textures are similar to that of the style image

An example is shown below, where the content image is of a cat, and the style image is of Hokusai's Great Wave. The generated target image still contains the cat but is stylized with the waves, blue and beige colors, and block print textures of the style image!

<img src='https://github.com/sanketsans/Deep-Learning/blob/master/style_transfer/notebook_ims/style_tx_cat.png' width=80% />

For transfer learning, we need to preserve the weights of the layers that were trained for long. So, need to freeze the weights
of all feature layers and only modify the end fully connected layers to our needs. 

for param in vgg.parameters():
  param.requires_grad_(False)

According to paper, ![Image Style Transfer Using Convolutional Neural Networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf),
by Gatys in PyTorch, the style content(color, pattern, shapes etc.) of an image is represented by layers : 
- conv1_1
- conv2_1
- conv3_1
- conv4_1
- conv5_1

And the content of an image is represented by :
- conv4_2

So, we will extract the information from all these layers from VGG16 network and use them to calculate our error function. 

## Gram Matrix
The matrix which will represent the style information of an image. To calculate gram image for each feature layer of an image : 
-   Get the depth, height, and width of a tensor using batch_size, d, h, w = tensor.size()
-   Reshape that tensor so that the spatial dimensions are flattened, since each feature layer has a number of feature maps. Its
dimensions will be depth(num of feature maps)xlengthxwidth(image). On flattening -> depthx(length*width)
-   Calculate the gram matrix by multiplying the reshaped tensor by it's transpose. 

## Style weights & Content weights
Now to include more of style representation in the output image, we need to enchance the style loss(beta) as compared to 
content loss(alpha). Normally, the ration is alpha:beta :: 1:100. 
You can also specify weights to each layer in the style feature. 

## Updating Loss 
To calculate the loss function for content, we only to compare those layers from which we are extracting the content 
information i.e; conv4_2
- content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)

Similarly , we only need to compare the gram matrix of output image to input image for each feature layer we are focusing on. 
- layer_style_loss = torch.mean((target_gram - style_gram)**2) * style_weights[layer] ## here i am multiplying the loss for each
layer as well. The enchancement should be in between 0 & 1.

In the end, we need to enchance the overall content & style loss by multiplying the alpha and beta. 
- total_loss = (content_weight * content_loss) + (style_loss * style_weight)

We repeat the same experiement for a number of epoch. We might see a huge loss at the beginning , if your beta is too high(1000), 
but it slowly reduces. Try to compare results on the target after each epoch. 
