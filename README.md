# spring_2025_dl_audio_project
Erdos deep learning bootcamp final project

## Introduction
The goal of this project was to create a model that, when given plain speech and some instrumental music, generates a vocal track from the speech that both suits the music and sounds like the speeker was singing. We used a CycleGAN with [Wave-U-Net-Pytorch](https://github.com/f90/Wave-U-Net-Pytorch/tree/master) by Daniel Stoller acting as the generative models and [MINIROCKETPlus](https://timeseriesai.github.io/tsai/models.minirocketplus_pytorch.html) as the discriminator. For the speech, we used [LibriSpeech](https://www.openslr.org/12) and for the music (vocals and accompaniment) we used [MUSDB18](https://sigsep.github.io/datasets/musdb.html#musdb18-compressed-stems). We applied Librosa to transform the audio to a mel spectrogram to reduce the size of the inputs to the models. Because this is a lossy transformation, we needed to recover the phase information using the classical Griffin-Lim algorithm in Libtrosa. 

## Training
We might have graphs of our losses as we trained here.

## 
