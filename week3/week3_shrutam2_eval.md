# Week 3 — Shrutam-2 ASR Evaluation

- **Models:** `bharatgenai/Shrutam-2` vs `openai/whisper-large-v3`
- **Hardware:** RunPod H100 80GB (SXM) · batch size 1
- **Test set:** 10 Hindi + 10 Tamil (AI4Bharat Shrutilipi) + 5 code-mixed Hindi-English (own recordings) = 25 clips, 16 kHz mono
- **Scoring:** WER & CER via `jiwer` after light normalization (lowercase, strip punctuation, collapse whitespace), aggregated at corpus level per group.

> **Caveat:** the Hindi/Tamil clips are public Shrutilipi data and may have been seen in training, so those WERs are an *optimistic* floor. Only the 5 code-mixed clips are guaranteed unseen.

---

## 1. WER, CER by language  (lower is better)

| Language | Shrutam-2 WER | Shrutam-2 CER | Whisper large-v3 WER | Whisper large-v3 CER |
|---|---|---|---|---|
| Hindi | 0.04 | 0.01 | 0.42 | 0.22 |
| Tamil | 0.19 | 0.03 | 0.64 | 0.14 |
| Code-mixed | 0.48 | 0.56 | 0.43 | 0.49 |
| **Overall** | 0.15 | 0.08 | 0.50 | 0.21 |

**My reading:**

WER: Word Error Rate
CER: Character Error Rate

These two are used to calculate the error rate from the string generated.
CER is usually lower, especially for Indic languages, because words usually contain many sub words and add-ons. 

A single character being different will trigger the WER, but it is not completely wrong, here the CER is more useful.

Overe here, we can clearly see CER performs much better than WER.

It is even better for Tamil, because tamil words are usually much longer than hindi. Tamil connects multiple parts and forms it into one word. Hindi user more shorter words comparatively. 

Another part is that for code mix clips, the error rate is much higher. This is mainly due to the fact that it does not transcribe the word in english, and rather writes the word in hindi with english pronunciation. 

Instead of writing: इस startup का valuation
Shrutam writes: इस स्टार्टअप का वैल्यूएशन

Which is not technically wrong, but it is not in english.

---

## 2. Profiling

| Model | Avg latency/clip (s) | Peak GPU mem (GB) | Throughput (audio-s / inference-s) |
|---|---|---|---|
| Shrutam-2 | 0.43 | 7.9 | 26.2 |
| Whisper large-v3 | 2.89 | 4.6 | 3.9 |

*Latency = wall-clock per clip · Throughput = audio-seconds per inference-second.*

#### Latency by language (Whisper ÷ Shrutam-2)

| Language | Shrutam-2 (s) | Whisper large-v3 (s) | Speed-up (Whisper / Shrutam-2) |
|---|---|---|---|
| Hindi | 0.44 | 3.17 | 7.3x |
| Tamil | 0.51 | 3.23 | 6.4x |
| Code-mixed | 0.29 | 1.67 | 5.7x |
| **Overall** | 0.43 | 2.89 | 6.6x |

**My reading:**

Shrutam: fp32  
Whisper: fp16


Even thought Shrutam used 2x the memory size, it was significantly faster than Whisper over here. There were mainly two reasons behind it.

1. Whisper was mainly trained on english centric vocabulary, so it mainly takes in english tokens. For languages which are not English, it requires much more tokens. It divides the phrases and words into smaller and smaller tokens which it can understand. 

Here, the code mixed clips have some english words in them. Whisper performed slightly better in the mixed clips because it didnt spend extra tokens on the english words. Only the hindi words required extra tokens from it.

2. Whisper takes in audio in chunks of 30 seconds. So even a 10s audio gets converted to 30s with padding.
Shrutam processes only the actual length of the clip.

---

## 3. Transcription failures (Shrutam-2)


I was not able to debug the Tamil transcriptions. 
Here are a few predictions I was able to make for mixed audios. 

### 1 — codemixed/clip_002.wav  (WER 0.65)
- **Reference:**  मैंने weekend पर Netflix की वो new series binge-watch की, last episode का twist तो mind-blowing था।
- **Prediction:** मैंने वीकेंड पर नेटफ्लिक्स की वो न्यू सिरीज़ बिंज वॉच की लास्ट एपिसोड का ट्विस्ट तो माइंड ब्लोइंग था
- **What went wrong:** The model correctly transscribed the audio, but it did not write the words in english. Shrutam does not have a specfic Hinglish model. It only has Hindi, so it just cannot write the words in english. 

### 2 — codemixed/clip_005.wav  (WER 0.57)
- **Reference:**  इस phone का camera तो amazing है but battery backup थोड़ा disappointing लगा मुझे।
- **Prediction:** इस फोन का कैमरा तो अमेज़िंग है बट बैटरी बैकअप थोड़ा डिस અપॉइंटिंग लगा मुझे
- **What went wrong:** Instead of writing disappointing, the model got confused and switched over to Gujarati. 


### 3 — codemixed/clip_001.wav  (WER 0.38)
- **Reference:**  Honestly यार, इस startup का valuation इतना high है कि मुझे थोड़ा doubt हो रहा है।
- **Prediction:** और अब यार इस स्टार्टअप का वैल्यूएशन इतना हाई है कि मुझे थोड़ा डाउट हो रहा है
- **What went wrong:** First word was not recognised properly, probably due to different pronunciation/accent in the audio clip. 




