# spring_2025_dl_audio_project - VocalCycleGAN
Erdos deep learning bootcamp final project

## Introduction
The goal of this project was to create a model that, when given plain speech and some instrumental music, generates a vocal track from the speech that both suits the music and sounds like the speeker was singing. We used a CycleGAN with [Wave-U-Net-Pytorch](https://github.com/f90/Wave-U-Net-Pytorch/tree/master) by Daniel Stoller acting as the generative models and [MINIROCKETPlus](https://timeseriesai.github.io/tsai/models.minirocketplus_pytorch.html) as the discriminator. For the speech, we used [LibriSpeech](https://www.openslr.org/12) and for the music (vocals and accompaniment) we used [MUSDB18](https://sigsep.github.io/datasets/musdb.html#musdb18-compressed-stems). We applied Librosa to transform the audio to a mel spectrogram to reduce the size of the inputs to the models. Because this is a lossy transformation, we needed to recover the phase information using the classical Griffin-Lim algorithm in Librosa. 

## CycleGAN - Generators, Discriminators, and Losses
The model class VocalCycleGAN defined in vcgan.py is trained by a modified cycleGAN training loop. CycleGAN was introduced in [2017 by Zhu-Park-Isola-Efros](https://junyanz.github.io/CycleGAN/) to transform between images in two different domains (for example turning images of horses to images of zebras and vice versa). In our cycleGAN framework, the two domains are singing and speech. Let us describe our cycleGAN loop in detail.

We have four neural networks in our training loop:
- generator_vocal : An instance of Wave-U-Net that accepts human speech and a clip of an instrumental track and attempts to generate a vocal performance.
- generator_speech : An instance of Wave-U-Net that accepts a vocal performance and attempts to generate human speech.
- discriminator_vocal : An instance of the MiniRocket discriminator that attempts to determine whether a given clip of audio is real singing or machine generated. 
- discriminator_speech : An instance of the MiniRocket discriminator that attempts to determine whether a given clip of audio is real speech or machine generated.

Both discriminators output a number between zero and one representing the probability that a sample is real. So the discriminator outputs smaller values when it thinks the sample is fake.

To train the discriminators, we minimize two types of loss function:
- Binary cross entropy of the predictions of each generator
- Adversarial loss $\sum_x D(G(x))^2$ where $x$ is a piece of sample data, $D$ is the discriminator, and $G$ is the generator. Minimizing this encourages the generator

To train the generators, we minimize three loss functions:
- L1_loss

## Data

## Training
We might have graphs of our losses as we trained here.

## Files

## Dependencies

## 
