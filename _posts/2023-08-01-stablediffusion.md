---
layout: post
title: Using Diffusers for Image Generation with Language Prompts using Stable Diffusion XL
date: 2023-08-01 16:00
summary: This tutorial demonstrates how to use the "Diffusers" library and a pre-trained DiffusionPipeline to generate images from text prompts. By leveraging the power of Stable Diffusion Dreambooth XL model, users can create images based on textual input.
categories: General
---

<img src="https://i.ibb.co/WPhmFcp/diffusion.jpg" alt="diffusion" border="0">


In this tutorial, we will learn how to use the "Diffusers" library, a part of the Hugging Face's Transformers ecosystem, for generating images from text prompts. We will use a pre-trained DiffusionPipeline to accomplish this. The DiffusionPipeline combines the power of language models and image generation to create images based on textual input.

## Prerequisites

Before we get started, please make sure you have the following installed:

- Python (3.6 or higher)

You can install the required packages using the following command:

```bash
pip install transformers accelerate invisible-watermark>=0.2.0 diffusers>=0.19.0
```

## Step 1: Import Necessary Libraries

First, let's import the required libraries for our tutorial:

```python
from diffusers import DiffusionPipeline
import torch
```

## Step 2: Load the Pre-trained Model

We will use a pre-trained DiffusionPipeline for our image generation task. For this tutorial, we will use the "stable-diffusion-xl-base-1.0" model.

```python
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16")
```

The `torch_dtype=torch.float16` and `variant="fp16"` arguments are optional but recommended for faster execution on GPUs. The model will be loaded onto the GPU (if available) for faster image generation.

## Step 3: Generate an Image from Text Prompt

Now, let's use our loaded model to generate an image from a given text prompt.

```python
# Provide the text prompt for the image generation
prompt = "Fallen angel man, D&D, dark fantasy, highly detailed, digital painting, artstation, concept art, sharp focus, illustration, cinematic lighting, art by artgerm and greg rutkowski and alphonse mucha"

# Generate the image using the DiffusionPipeline
image = pipe(prompt=prompt).images[0]
```

The `pipe(prompt=prompt)` call takes the text prompt as input and returns an image corresponding to that prompt. The generated image will be stored in the `image` variable.

## Step 4: Display the Generated Image

Finally, let's display the generated image using Python's matplotlib library.

```python
import matplotlib.pyplot as plt

# Display the generated image
plt.imshow(image)
plt.axis('off')
plt.show()
```


## Full Code

Here's the full code combining all the steps together:

```python
from diffusers import DiffusionPipeline
import torch
import matplotlib.pyplot as plt

# Load the pre-trained model
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16")

# Set the device to GPU if available
pipe.to("cuda")

# Provide the text prompt for the image generation
prompt = "Oppenheimer thinking about atomic structure with Heisenberg, Einstein and Niels Bohr looking over, dreamy, iPhone wallpaper,"

# Generate the image using the DiffusionPipeline
image = pipe(prompt=prompt).images[0]

# Display the generated image
plt.imshow(image)
plt.axis('off')
plt.show()
```


In the above tutorial, we have learned how to use the Diffusers library to generate images from text prompts using a pre-trained DiffusionPipeline model easily. You can experiment with different text prompts and models to generate various images. Have fun exploring the possibilities of image generation through text prompts!

You can find a set of about 80,000 prompts filtered and extracted [here](https://huggingface.co/datasets/Gustavosta/Stable-Diffusion-Prompts)