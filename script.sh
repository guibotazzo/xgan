#!/bin/bash

###############################
# DCGAN (inputxgrad) on CIFAR10
###############################

#GAN='DCGAN'
#DATASET='cifar10'
#FOLDER='weights/'$GAN'/'$DATASET'/'
#
#for i in 1 2 3 4 5
#do
#python train.py --gan $GAN -d $DATASET
#mkdir $FOLDER'run'$i'/'
#mv $FOLDER'disc_epoch_100.pth' $FOLDER'run'$i'/'
#mv $FOLDER'gen_epoch_100.pth' $FOLDER'run'$i'/'
#mv runs/* $FOLDER'run'$i'/'
#rm $FOLDER'*.pth'
#done

#GAN='RaSGAN'
#DATASET='mnist'
#XAI='saliency'
#FOLDER='weights/'$GAN'/'$DATASET'/'$XAI'/'
#
#for i in 3 4 5
#do
#python train.py --gan $GAN -d $DATASET -s 32 -c 1 --xai $XAI
#mkdir $FOLDER'run'$i'/'
#mv $FOLDER'disc_epoch_100.pth' $FOLDER'run'$i'/'
#mv $FOLDER'gen_epoch_100.pth' $FOLDER'run'$i'/'
#mv runs/* $FOLDER'run'$i'/'
#rm $FOLDER'*.pth'
#done





## -- CR - DEEPLIFT --------------
## -------- Benign --------
#unzip -q datasets/CR64_original.zip
#mv CR64 datasets
#rm -r datasets/CR64/Malignant
#
#python train.py --gan WGAN-GP --xai deeplift -d cr -s 64 -c 3
#
#mkdir weights/WGAN-GP/cr/deeplift/Benign
#mv runs/* weights/WGAN-GP/cr/deeplift/Benign
#mv weights/WGAN-GP/cr/deeplift/gen_epoch_100.pth weights/WGAN-GP/cr/deeplift/Benign
#mv weights/WGAN-GP/cr/deeplift/disc_epoch_100.pth weights/WGAN-GP/cr/deeplift/Benign
#
## -------- Malignant --------
#rm -r datasets/CR64
#unzip -q datasets/CR64_original.zip
#mv CR64 datasets
#rm -r datasets/CR64/Benign
#
#python train.py --gan WGAN-GP --xai deeplift -d cr -s 64 -c 3
#
#mkdir weights/WGAN-GP/cr/deeplift/Malignant
#mv runs/* weights/WGAN-GP/cr/deeplift/Malignant
#mv weights/WGAN-GP/cr/deeplift/gen_epoch_100.pth weights/WGAN-GP/cr/deeplift/Malignant
#mv weights/WGAN-GP/cr/deeplift/disc_epoch_100.pth weights/WGAN-GP/cr/deeplift/Malignant
#
#
## -- CR - INPUTXGRAD --------------
## -------- Benign --------
#unzip -q datasets/CR64_original.zip
#mv CR64 datasets
#rm -r datasets/CR64/Malignant
#
#python train.py --gan WGAN-GP --xai inputxgrad -d cr -s 64 -c 3
#
#mkdir weights/WGAN-GP/cr/inputxgrad/Benign
#mv runs/* weights/WGAN-GP/cr/inputxgrad/Benign
#mv weights/WGAN-GP/cr/inputxgrad/gen_epoch_100.pth weights/WGAN-GP/cr/inputxgrad/Benign
#mv weights/WGAN-GP/cr/inputxgrad/disc_epoch_100.pth weights/WGAN-GP/cr/inputxgrad/Benign
#
## -------- Malignant --------
#rm -r datasets/CR64
#unzip -q datasets/CR64_original.zip
#mv CR64 datasets
#rm -r datasets/CR64/Benign
#
#python train.py --gan WGAN-GP --xai inputxgrad -d cr -s 64 -c 3
#
#mkdir weights/WGAN-GP/cr/inputxgrad/Malignant
#mv runs/* weights/WGAN-GP/cr/inputxgrad/Malignant
#mv weights/WGAN-GP/cr/inputxgrad/gen_epoch_100.pth weights/WGAN-GP/cr/inputxgrad/Malignant
#mv weights/WGAN-GP/cr/inputxgrad/disc_epoch_100.pth weights/WGAN-GP/cr/inputxgrad/Malignant








## -- UCSB64 - SALIENCY --------------
## -------- Benign --------
#unzip -q datasets/UCSB64_original.zip
#mv UCSB64 datasets
#rm -r datasets/UCSB64/Malignant
#
#python train.py --gan WGAN-GP --xai saliency -d ucsb -s 64 -c 3
#
#mkdir weights/WGAN-GP/ucsb/saliency/Benign
#mv runs/* weights/WGAN-GP/ucsb/saliency/Benign
#mv weights/WGAN-GP/ucsb/saliency/gen_epoch_100.pth weights/WGAN-GP/ucsb/saliency/Benign
#mv weights/WGAN-GP/ucsb/saliency/disc_epoch_100.pth weights/WGAN-GP/ucsb/saliency/Benign
#
## -------- Malignant --------
#rm -r datasets/UCSB64
#unzip -q datasets/UCSB64_original.zip
#mv UCSB64 datasets
#rm -r datasets/UCSB64/Benign
#
#python train.py --gan WGAN-GP --xai saliency -d ucsb -s 64 -c 3
#
#mkdir weights/WGAN-GP/ucsb/saliency/Malignant
#mv runs/* weights/WGAN-GP/ucsb/saliency/Malignant
#mv weights/WGAN-GP/ucsb/saliency/gen_epoch_100.pth weights/WGAN-GP/ucsb/saliency/Malignant
#mv weights/WGAN-GP/ucsb/saliency/disc_epoch_100.pth weights/WGAN-GP/ucsb/saliency/Malignant
#
#
# -- UCSB64 - DEEPLIFT --------------
# -------- Benign --------
rm -r datasets/UCSB64
unzip -q datasets/UCSB64_original.zip
mv UCSB64 datasets
rm -r datasets/UCSB64/Malignant

python train.py --gan WGAN-GP --xai deeplift -d ucsb -s 64 -c 3

mkdir weights/WGAN-GP/ucsb/deeplift/Benign
mv runs/* weights/WGAN-GP/ucsb/deeplift/Benign
mv weights/WGAN-GP/ucsb/deeplift/gen_epoch_100.pth weights/WGAN-GP/ucsb/deeplift/Benign
mv weights/WGAN-GP/ucsb/deeplift/disc_epoch_100.pth weights/WGAN-GP/ucsb/deeplift/Benign

# -------- Malignant --------
rm -r datasets/UCSB64
unzip -q datasets/UCSB64_original.zip
mv UCSB64 datasets
rm -r datasets/UCSB64/Benign

python train.py --gan WGAN-GP --xai deeplift -d ucsb -s 64 -c 3

mkdir weights/WGAN-GP/ucsb/deeplift/Malignant
mv runs/* weights/WGAN-GP/ucsb/deeplift/Malignant
mv weights/WGAN-GP/ucsb/deeplift/gen_epoch_100.pth weights/WGAN-GP/ucsb/deeplift/Malignant
mv weights/WGAN-GP/ucsb/deeplift/disc_epoch_100.pth weights/WGAN-GP/ucsb/deeplift/Malignant

# -- UCSB64 - INPUTXGRAD --------------
# -------- Benign --------
rm -r datasets/UCSB64
unzip -q datasets/UCSB64_original.zip
mv UCSB64 datasets
rm -r datasets/UCSB64/Malignant

python train.py --gan WGAN-GP --xai inputxgrad -d ucsb -s 64 -c 3

mkdir weights/WGAN-GP/ucsb/inputxgrad/Benign
mv runs/* weights/WGAN-GP/ucsb/inputxgrad/Benign
mv weights/WGAN-GP/ucsb/inputxgrad/gen_epoch_100.pth weights/WGAN-GP/ucsb/inputxgrad/Benign
mv weights/WGAN-GP/ucsb/inputxgrad/disc_epoch_100.pth weights/WGAN-GP/ucsb/inputxgrad/Benign

# -------- Malignant --------
rm -r datasets/UCSB64
unzip -q datasets/UCSB64_original.zip
mv UCSB64 datasets
rm -r datasets/UCSB64/Benign

python train.py --gan WGAN-GP --xai inputxgrad -d ucsb -s 64 -c 3

mkdir weights/WGAN-GP/ucsb/inputxgrad/Malignant
mv runs/* weights/WGAN-GP/ucsb/inputxgrad/Malignant
mv weights/WGAN-GP/ucsb/inputxgrad/gen_epoch_100.pth weights/WGAN-GP/ucsb/inputxgrad/Malignant
mv weights/WGAN-GP/ucsb/inputxgrad/disc_epoch_100.pth weights/WGAN-GP/ucsb/inputxgrad/Malignant