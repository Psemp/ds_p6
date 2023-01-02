# <u>Classification of products based on their description and pictures</u>

###**Project's requirements :**

[Direct link to OpenClassrooms project](https://openclassrooms.com/fr/paths/164/projects/631/assignment)


<i>The notebooks have been separated to keep a rather light enviro. and to execute the more demanding ones (BERT and CNN/VGG16) on Kaggle. Several annexes as well.</i>

- requirements.txt for dependencies (tensorflow / tensorflow-macos line must be commented out and adapted.)

- dataset downloadable [here](https://s3-eu-west-1.amazonaws.com/static.oc-static.com/prod/courses/files/Parcours_data_scientist/Projet+-+Textimage+DAS+V2/Dataset+projet+pre%CC%81traitement+textes+images.zip). It is pretty heavy and thus not included in the repo.

*For text :*

- Preprocessing, dimensional reduction (`nb_01_cleaning_txt`)
- Bag Of Words & TF-IDF (`nb_02_text_classification_1`)
- Word embedding using Word2Vec & FastTest (`nb_02_text_classification_1`)
- Sentence embedding using BERT (`nb_02_text_classification_2_BERT`) and USE (`nb_02_text_classification_3_USE`)

*For images :*

- Feature extraction using SIFT (`nb_03_sift_classification`)
- CNN transfer learning (pretrained VGG16 with fine tuning : `nb_04_image_classification_cnn`)

*Annexes :*

- `image_preprocessing_mp.py` : used to square, fill empty pixels with black, correct histograms and contrast, convert to black (SIFT), resize (124 for SIFT, 224 for VGG16). Done using `concurrent.futures` to keep treatment time low.
- `nb_anx_0_gensim_downloader` : Downloads the two embedding models and checks for correct saving
- `nb_anx_1_tf_export` : Specific export of the dataset as a csv (problems with pickle format on Kaggle)
- `nb_anx_2_images_to_h5` : Loads the dataset, splits it into train/test with balanced classes, then proceeds to load and write the images and their classes into a h5 file more suited to NN tasks.