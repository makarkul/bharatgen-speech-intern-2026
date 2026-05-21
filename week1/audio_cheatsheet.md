Week 1 Audio Cheatsheet

# Sampling Rate

Sampling rate is the number of samples we take in one sec. Main two types are 16kHz and 44.1kHz
Human speech is bw 80Hz and 8kHz. So we take 16kHz sampling rate for speech recognition.
Other stuff like musical instruments fall under 20kHz range. So we take 44.1kHz when we want to sample them. 

## ASR Models
automatic speech recognition models need 16kHz freq. they do not need any higher frequency because it simply does not need any more. our speech all lies withing the 8kHz range. the model was also trained on data with 16kHz so we need that frequency.

## What happens if we feed a diff frequency to a diff model
It will give a bad result. For eg, if our model supports 16kHz frequency, then if we feed in 44.1kHz, it will accept it. But it will still divide it into chunks of 16. So our audio clip would be approx 2.5 times longer


# Stereo and Mono

In audio, there are channels of audio. 
In mono audio, theres only one dimension of sound, in one channel.
But in stereo, there can be multiple. 
Mono is like listening with one ear.
Stereo eg: in movies theatres, theres surround sound. thats a multi channel audio, it is coming from various sources.


## Why we need mono 
For our model, we are working on speech. 
Speech is from one single source, so we only need mono audio. 

If there are multiple channels for audio, it can cause ambiguity. So we need the array we have to be of a single dimension. 

the single dimension mono also requires less storage and processing.


# Resampling

Audio files we have are usually of different frequencies. For example Voice recordings from my phone were 48kHz. But for ASR, we need a sampling rate of 16kHz. The ASR models have been trained on 16kHz data, which is all we need for speech recognition. If we feed wrong frequency to it, it will give inconsistent results which will be wrong. 
So we need to resample it to 16kHz before we can proceed further. 


# Mel Spectrogram

When we get an audio file, we divide it with the help of sample rates. We can see this using a waveform diagram. But the waveform diagram only gives us the amplitude of the wave. It only tells us how loud the sound is. It does not tell us anyting about the pitch or the frequency of the sound wave. 

This is where we need a spectrogram.

A spectrogram is a figure which compares time vs frequency of wave. It shows us the loudness of the wave through colors.

## Why Mel Scale
A normal 16kHz sample rate divides the sample into 16000 sections, this is too much data to process. And because it has been divided into so many sections, it is hard to make sense of.

So, we use the mel scale. 

Humans hear sound differently from 80Hz to 8khz. 
we can differentiate the difference in frequencies in the lower end of the easier than in the higher end.
Like we can differentiate bw 200hz and 300hz easily, but not between 5500 and 5600.
The freq on the higher end are pretty same for us. 

So the mel scale is useful for us. 

The mel scale puts more focous on sounds with lower frequencies and stretches the width of a section. And it will compress the frequencies which are higher into a certain range.

So in the sections, the lower frequencies are in smaller sections, the larger frequencies have been put into sections which expand more.