# Model Card for Cats vs Dogs image classification

This model uses a Fine-Tunned version of [google/vit-base-patch16-224-in21k](https://huggingface.co/google/vit-base-patch16-224-in21k) Vision Transformer to classify images from the Cats vs. Dogs dataset. It takes input images and predicts whether they contain a cat or a dog.

## Model Details

### Model Description

The Vision Transformer (ViT) is a transformer encoder model (BERT-like) pretrained on a large collection of images in a supervised fashion, namely ImageNet-21k, at a resolution of 224x224 pixels. 

Images are presented to the model as a sequence of fixed-size patches (resolution 16x16), which are linearly embedded. One also adds a [CLS] token to the beginning of a sequence to use it for classification tasks. One also adds absolute position embeddings before feeding the sequence to the layers of the Transformer encoder.

Note that this model does not provide any fine-tuned heads, as these were zero'd by Google researchers. However, the model does include the pre-trained pooler, which can be used for downstream tasks (such as image classification).

By pre-training the model, it learns an inner representation of images that can then be used to extract features useful for downstream tasks: if you have a dataset of labeled images for instance, you can train a standard classifier by placing a linear layer on top of the pre-trained encoder. One typically places a linear layer on top of the [CLS] token, as the last hidden state of this token can be seen as a representation of an entire image.

By fine-tunning the model, it learns threats of cat and dog images that allow the model to extract relevant features of the two types of images, allowing the model to learn how to classify properly different images than the ones used for training.

- **Developed by:** [Model Developers](#model-developers)
- **Model type:** Vision Transformer
- **Language(s) (NLP):** Not applicable (vision task)
- **Finetuned from model:** ImageNet-21k

### Model Developers

- **Pablo Gete**: pablo.gete@estudiantat.upc.edu

- **Oriol López**: oriol.lopez.petit@estudiantat.upc.edu

- **Xavier Pacheco**: xavier.pacheco.bach@estudiantat.upc.edu

- **Xiao Li Segarra**: xiao.li.segarra@estudiantat.upc.edu

- **David Torrecilla**: david.torrecilla.tolosa@estudiantat.upc.edu

### Model Sources

- **Repository:** [google/vit-base-patch16-224-in21k](https://huggingface.co/google/vit-base-patch16-224-in21k)

## Uses

The uses can be found following:

### Direct Use

The model is intended for direct image classification tasks, where an image is fed to the model, and it outputs the prediction—either "cat" or "dog." It can be used for quick and reliable inference in scenarios where images contain only one of the two animals.

### Downstream Use [optional]

The model can be adapted for use in various real-world applications, such as sorting or labeling images in pet-related services, or as part of an automated system in animal shelters. Further fine-tuning might be necessary for tasks that involve a broader range of animals or specific use cases like image captioning.


## Bias, Risks, and Limitations

The model is limited by its training data and may perform poorly on images outside the Cats vs. Dogs dataset. It might not generalize well to other types of animal images without further fine-tuning.

This model does not present significant ethical concerns but should be used responsibly, ensuring datasets used for further training or deployment are appropriately labeled and free of bias.


### Recommendations

Users (both direct and downstream) should be made aware of the risks, biases and limitations of the model. More information needed for further recommendations.

## How to Get Started with the Model

Read carefully the follwoing to start with the model

## Training Details

### Training Data

The information on the data used for training this model can be found in the following Dataset Card: (https://github.com/taed2-2425q1-gced-upc/TAED2_scooby_doo_team/tree/master/docs/dataset_card.md).

It consists on two .parquet files with images of dogs or cats and its labels.

### Training Procedure

#### Preprocessing 

The preprocessing step can be found in `prepare.py` file. It consists on loading data from Parquet files, cropping all the images to 228x228 resolution splitting it into training, validation, and test sets, and saving the data and corresponding images to specified directories. Configuration parameters are loaded from a YAML file to control the data splitting process. Finally, images are saved in separate folders for each dataset split (train, valid, test).

#### Training overview

The training step can be found in `train.py` file. It performs a fine tunning on the Vision Transformer model, doing a hyperparameter search on various parameters. When the training has finished the hyperparameter search, the model is saved as a pickle file for future use.

#### Training Hyperparameters

The training hyperparameters tested can be found in the following file (https://github.com/taed2-2425q1-gced-upc/TAED2_scooby_doo_team/blob/master/params.yaml).

The following values were the best hyperparameters found during the training phase:
      - `batch size = 16`
      - `learning rate = 5e-05`
      - `num epochs = 8`
      - `weight decay = 0.0`
      - `optimizer = adam`

## Evaluation

In order to find the best hyperparameter, a validation set is used for getting the accuracy values for each combination, on images that the model has not seen during the training phase. 

### Testing Data, Factors & Metrics

#### Testing Data

The testing data consists of a separate set of images not used during training or validation, obtained from the Cats vs. Dogs dataset. The dataset is split into a small subset of test images, with an equal distribution of cat and dog images to ensure balanced evaluation. The images are preprocessed to match the model's input requirements (224x224 resolution).

#### Factors

The evaluation is disaggregated by:

- Class Labels: Cats and Dogs.
- Image Quality: Variations in lighting, background, and image resolution to assess how these factors affect classification accuracy.
- Model Variants: Comparisons between different fine-tuned versions of the Vision Transformer based on hyperparameter settings.

#### Metrics

As there is no specific requirements on any class higher recognition. `accuracy` is used as the primary metric to evaluate the model

### Results

The accuracy score achieved during the evaluation was `-`, reflecting the model's performance in classifying images of dogs and cats.

#### Summary

As the preprocessing and training steps were correctly done with balanced data, the accuracy achieved reflects the correctness of the model on classifying images of dogs or cats.

## Environmental Impact

To be measured and completed in the future.

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** {{ hardware_type | default("[More Information Needed]", true)}}
- **Hours used:** {{ hours_used | default("[More Information Needed]", true)}}
- **Cloud Provider:** {{ cloud_provider | default("[More Information Needed]", true)}}
- **Compute Region:** {{ cloud_region | default("[More Information Needed]", true)}}
- **Carbon Emitted:** {{ co2_emitted | default("[More Information Needed]", true)}}

## Technical Specifications

### Model Architecture and Objective

Vision Transformer fine-tunned.

### Compute Infrastructure

#### Hardware

- **GPU or CPU**: Can be trained using any CPU or GPU
- **RAM**: 4GB or more recommended for faster training
- **Storage**: Sufficient local or cloud storage required to handle the dataset and model files

#### Software

Using poetry the correct Python installation can be found (https://github.com/taed2-2425q1-gced-upc/TAED2_scooby_doo_team/blob/master/pyproject.toml).

## Citation [optional]

**BibTeX:**

```bibtex
@misc{wu2020visual,
      title={Visual Transformers: Token-based Image Representation and Processing for Computer Vision}, 
      author={Bichen Wu and Chenfeng Xu and Xiaoliang Dai and Alvin Wan and Peizhao Zhang and Zhicheng Yan and Masayoshi Tomizuka and Joseph Gonzalez and Kurt Keutzer and Peter Vajda},
      year={2020},
      eprint={2006.03677},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

```bibtex
@inproceedings{deng2009imagenet,
  title={Imagenet: A large-scale hierarchical image database},
  author={Deng, Jia and Dong, Wei and Socher, Richard and Li, Li-Jia and Li, Kai and Fei-Fei, Li},
  booktitle={2009 IEEE conference on computer vision and pattern recognition},
  pages={248--255},
  year={2009},
  organization={Ieee}
}
```
