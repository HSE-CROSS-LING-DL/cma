# Cross Lingual Morphological Analysis
### Deep Learning Final Project

---

## Team

- Mikhailov Vlad
- Reshetnikova Arina
- Serikov Oleg

## Task

Multilingual morphological analysis for a low-resource language ([VarDial 2019 Competition](https://github.com/ftyers/vardial-shared-task)).

Upgrade [last year result](https://www.aclweb.org/anthology/W19-1415.pdf) using deep learning techniques.

[Detailed description](https://docs.google.com/document/d/1iVaGEvkJm2wbELNv74AJYCofToSNaSOc2fWlrMY8xfw/edit#heading=h.p7fj7q5ek1cq)

## Data

- [Train](https://github.com/ftyers/vardial-shared-task/blob/master/train/trk-uncovered)
- [Dev](https://github.com/ftyers/vardial-shared-task/blob/master/dev/trk-uncovered)
- [Test](https://github.com/ftyers/vardial-shared-task/blob/master/test/trk-uncovered)

## Code

- [transliteration](https://github.com/HSE-CROSS-LING-DL/transliteration-tur)
- [POStagger.ipynb](POStagger.ipynb) - POS tag prediction
---
#### Other trials:
* [multilingual_fasttext.ipynb](multilingual_fasttext.ipynb) - learning fasstext embeddings for all languages
* [wikidump_preprocessing.ipynb](wikidump_preprocessing) - preprocessing data for further language model training
* [rnn_language_model.ipynb](rnn_language_model.ipynb) - language model
* [lemma_prediction.ipynb](lemma_prediction.ipynb) – char-level encoder-decoder for lemma prediction

[Presentation](https://docs.google.com/presentation/d/1BsMvcf_Irg1cm1ITF-c1RklAlF0tzvXRw8NpJbARCrU/edit#slide=id.g6cc047a59b_0_60)
