# OOD4NLU

This repository contains the code for the work [Out-of-domain detection for natural language understanding in dialog systems](https://arxiv.org/pdf/1909.03862.pdf).

The `POG` folder contains the pseudo OOD sample generation model code. The `CNN_KL` folder contains the code for a CNN-based text classifier with KL regularization.

All these codes are developed and tested using TensorFlow 1.12.0 and python 3.6.5.

## Usage of POG

1. Install dependency

```bash
pip install -r requirements.txt
```

2. Make a new project folder (for example, `project`). Copy the `config.json` file to this folder, and make a new `data` folder.

```bash
mkdir project
cp ${code-folder-of-POG}/config.json project/
cd project
mkdir data
```

3. Put the following files in the `data` folder: `ind_train`, `ind_dev`, `ood_dev`, `ind_test`, `ood_test`.
Each file is a tab-separated file with two columns. The first column is the intent name, and the second column is the sentence.
We have prepared the [CLINC150 dataset](https://github.com/clinc/oos-eval) for you in `POG/data`.
Note that the POG model does not need to use OOD data for training.

```text
translate	in spanish, meet me tomorrow is said how
translate	in french, how do i say, see you later
translate	how do you say hello in japanese
```

3. Change the `project/config.json` file to specify which data file to use.

- train_file: the training file of IND data, which is `ind_train`
- ind_valid_file: the validation file of IND data, which is `ind_dev`
- ood_valid_file: the validation file of OOD data, which is `ood_dev`
- ind_test_file": the test file of IND data, which is `ind_test`
- ood_test_file": the test file of OOD data, which is `ood_test`

Note that our code will automatically look into the `data` folder for these files.

You can also specify some important hyperparameters:
- word_vocab_size: maximum number of words in the vocabulary (important!)
- pretrained_embed: whether to use the pretrained embedding (You can use the GloVe embedding for example)
- max_decode_len: the maximum length of the decoded sentence
- max_epoch: the maximum number of epochs
- max_utter_len: the maximum length of the input utterance (used in the preprocessing step)

4. Change to the code folder of our POG implementation, and use the following command to train the POG model:

```bash
cd ${code-folder-of-POG}
python main.py --config ${path-to-project/config.json} --gpu {gpu}
```

5. After the POG model is trained, use the following command to sample utterances from the trained model:

```bash
python generate.py --config ${path-to-project/config.json} --gpu {gpu} --outfile {outfile} --count {50000} --is_sample True --sample_t 1.0
```

## Usage of CNN_KL

To train the CNN text classifier, you need to prepare a set of OOD data.
For example, you can use the above POG model to generate pseudo OOD data, or sample from whatever text corpus for OOD data.
After obtaining the OOD data, you can use the following steps to train the CNN text classifier with the KL regularization.

1. Install dependency

```bash
pip install -r requirements.txt
```

2. Make a new project folder (for example, `project`). Copy the `config.json` file to this folder, and make a new `data` folder.

```bash
mkdir project
cp config.json project/
cd project
mkdir data
```

3. Put the following files in the `data` folder: `ind_train`, `ind_dev`, `ood_dev`, `ind_test`, `ood_test`, `fake_ood`.
Each file is a tab-separated file with two columns. The first column is the intent name, and the second column is the sentence.
We have prepared the [CLINC150 dataset](https://github.com/clinc/oos-eval) for you in `CNN_KL/data`,
along with a set of pseudo OOD data (i.e., `fake_ood`) generated using the POG model.
The classifier will use these pseudo OOD samples to calculate the KL regularization loss.

```text
translate	in spanish, meet me tomorrow is said how
translate	in french, how do i say, see you later
translate	how do you say hello in japanese
```

3. Change the `project/config.json` file to specify which data file to use.

- ind_train_data": the training file of IND data, which is `data/ind_train`
- ood_train_data": the training file of OOD data, which is `data/fake_ood`
- ind_valid_data": the validation file of IND data, which is `data/ind_dev`
- ood_valid_data": the validation file of OOD data, which is `data/ood_dev`
- ind_test_data": the validation file of IND data, which is `data/ind_test`
- ood_test_data": the validation file of OOD data, which is `data/ood_test`

Note that you need to retrain the `data` prefix in the path for each data file.

Important hyperparameters:
- pretrained_embed: whether to use the pretrained embedding
- max_decode_len: the maximum length of the decoded sentence
- max_epoch: the maximum number of epochs
- max_utter_len: the maximum length of the utterance (used in the preprocessing step)

4. Change to the code folder of our CNN_KL implementation, and use the following command to train the text classifier with different random seeds:

```bash
cd ${code-folder-of-CNN_KL}
python multi-seed.py --config ${path-to-project/config.json} --gpu {gpu} --shuffle_data True --seeds 10,20,30,40,50
```

The above command will train five text classifiers with five random seeds 10, 20, 30, 40, and 50, respectively.

5. Use the following command to test the trained text classifiers:

```bash
python multi-seed.py --config ${path-to-project/config.json} --gpu {gpu} --seeds 10,20,30,40,50 --is_train False
```

Note that the code for POG and CNN_KL released here is re-implemented based on our paper.
The original codes used for our study are part of Samsung's internal codebase and thus cannot be retrieved.

Please kindly cite our paper if you find this repository useful.

```bibtex
@article{zheng2020out,
  title={Out-of-domain detection for natural language understanding in dialog systems},
  author={Zheng, Yinhe and Chen, Guanyi and Huang, Minlie},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  volume={28},
  pages={1198--1209},
  year={2020},
  publisher={IEEE}
}
```

# Response to the re-implemented results reported by Marek et al., 2021

We have noticed a work published on NAACL2021 Industrial Track: 
[Marek et al., OodGAN: Generative Adversarial Network for Out-of-Domain Data Generation](https://aclanthology.org/2021.naacl-industry.30/) 
that try to re-implement our model.
However, the re-implemented results reported in their paper are much lower than ours.
We have tried to contact the authors of Marek et al. to see if they can provide us with their code to reproduce the results reported in their paper.
However, we are informed by the authors that their code is not publicly available. 

We suspect the primary reason for such a performance gap is that Marek et al. did not train a good classifier for IND samples, let alone utilize the generated pseudo OOD data.
In fact, the OOD detection performance largely depends on the quality of the IND classifier. 
If the classifier can not perform well in classifying IND samples, then the classifier's performance in detecting OOD samples will likely be very low.

In Marek et al., the maximum classification accuracy of IND samples reported on the CLINC150 dataset is 90.11% (see Table 4 in Marek et al.).
However, in our implementation, a simple CNN-based text classifier can push this accuracy to 93.0+%.
The accuracy score could be much higher if we used a pretrained model (about 97.00% if we use BERT).
We suspect such a degenerated classifier is the main reason for their low OOD detection performance.

Moreover, the results on the CLINC150 dataset reported by Marek et al. are suspicious because their model underperforms the simplest baseline: using a naive text classifier trained on IND samples without KL regularization (i.e., the MSP baseline reported in our paper.).
Similar results for this simple baseline are reported in various papers, including

- [Revisiting Mahalanobis Distance for Transformer-Based Out-of-Domain Detection, AAAI2021](https://arxiv.org/pdf/2101.03778.pdf), see Table2
- [D2U: Distance-to-Uniform Learning for Out-of-Scope Detection](https://openreview.net/pdf?id=BUXecToWr-5), see Table2
- [Energy-based Unknown Intent Detection with Data Manipulation, ACL2021](https://arxiv.org/pdf/2107.12542.pdf), see Table2
- [Practical and Efficient Out-of-Domain Detection with Adversarial Learning, SAC2022](https://dl.acm.org/doi/pdf/10.1145/3477314.3507089), see Table2
- [Evaluating the Practical Utility of Confidence-score based Techniques for Unsupervised Open-world Classification, ACL2022](https://aclanthology.org/2022.insights-1.3/), see Table2

The performance of this simplest baseline reported in most above papers can reach an AUROC score of about 93.0, much higher than the results reported by Marek et al.
