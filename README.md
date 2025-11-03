# Cross-View Code Alignment for Image Hashing in the Age of Foundation Models

[![arXiv](https://img.shields.io/badge/arXiv-2510.27584-b31b1b.svg)](https://arxiv.org/abs/2510.27584)

## 👩‍💻 Authors

<sup>1</sup> Ilyass Moummad, <sup>1,2</sup> Kawtar Zaher, <sup>3</sup> Hervé Goëau, <sup>1</sup> Alexis Joly

<sup>1</sup> INRIA, LIRMM, Université de Montpellier, France <br>
<sup>2</sup> Institut National de l’Audiovisuel, France <br>
<sup>3</sup> CIRAD, UMR AMAP, Montpellier, Occitanie, France

---

**Keywords:** Image Hashing · Image Retrieval · Cross-View Alignment · Coding-Rate Maximization · Foundation Models

---

## ⚡ TL;DR

We propose *Cross-View Code Alignment (CroVCA)*, a simple and universal principle for hashing foundation model embeddings using Binary Cross-Entropy and Coding-Rate Maximization. It unifies unsupervised and supervised hashing under the same framework.

---

## 📝 Abstract

> Efficient large-scale retrieval requires representations that are both compact and discriminative. Foundation models provide powerful visual and multimodal embeddings, but nearest neighbor search in these high-dimensional spaces is computationally expensive. Hashing offers an efficient alternative by enabling fast Hamming distance search with binary codes, yet existing approaches often rely on complex pipelines, multi-term objectives, designs specialized for a single learning paradigm, and long training times. We introduce **CroVCA (Cross-View Code Alignment)**, a simple and unified principle for learning binary codes that remain consistent across semantically aligned views. A single binary cross-entropy loss enforces alignment, while coding-rate maximization serves as an anti-collapse regularizer to promote balanced and diverse codes. To implement this, we design **HashCoder**, a lightweight MLP hashing network with a final batch normalization layer to enforce balanced codes. HashCoder can be used as a probing head on frozen embeddings or to adapt encoders efficiently via LoRA fine-tuning. Across benchmarks, CroVCA achieves state-of-the-art results in just **5 training epochs**. At 16 bits, it performs particularly well—for instance, unsupervised hashing on COCO in under 2 minutes and supervised hashing on ImageNet100 in about 3 minutes on a single GPU. These results highlight CroVCA’s efficiency, adaptability, and broad applicability.

---

## 🚀 Features

- Unified training for **unsupervised + supervised** hashing
- Lightweight **HashCoder** with BatchNorm promoting balanced binary codes
- Optional **LoRA fine-tuning** for efficient adaptation of foundation models
- **Fast training** with just 5 epochs on a single GPU

---

## 📂 Data Preparation

Follow the step in [hashing-baseline](https://github.com/ilyassmoummad/hashing-baseline/tree/main/image) and [deephash](https://github.com/swuxyj/DeepHash-pytorch/tree/master) to download the data.

---

## 💻 Installation

```bash
git clone https://github.com/ilyassmoummad/CroVCA.git
```

Install required packages (see `requirements.txt`) to ensure dependencies like `open_clip`, `transformers`, `torchvision`, etc., are available:

```bash
pip install -r requirements.txt
```

---

## 📊 Example Usage

Unsupervised hashing on CIFAR-10:

```bash 
python main.py --encoder dinov3 --hashcoder small --dataset cifar10 --data_dir path_to_data --bitdim 16 --epochs 5 --device cuda:0
```

---

## 📝 To cite this work:

```
@misc{crovca,
      title={Image Hashing via Cross-View Code Alignment in the Age of Foundation Models}, 
      author={Ilyass Moummad and Kawtar Zaher and Hervé Goëau and Alexis Joly},
      year={2025},
      eprint={2510.27584},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2510.27584}, 
}
```
