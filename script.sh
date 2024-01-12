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

# -- CR - SALIENCY --------------
# -------- Benign --------
unzip -q datasets/CR64_original.zip
mv CR64 datasets
rm -r datasets/CR64/Malignant

python train.py --gan WGAN-GP --xai saliency -d cr -s 64 -c 3

mkdir weights/WGAN-GP/cr/saliency/Benign
mv runs/* weights/WGAN-GP/cr/saliency/Benign
mv weights/WGAN-GP/cr/saliency/gen_epoch_100.pth weights/WGAN-GP/cr/saliency/Benign
mv weights/WGAN-GP/cr/saliency/disc_epoch_100.pth weights/WGAN-GP/cr/saliency/Benign

# -------- Malignant --------
rm -r datasets/CR64
unzip -q datasets/CR64_original.zip
mv CR64 datasets
rm -r datasets/CR64/Benign

python train.py --gan WGAN-GP --xai saliency -d cr -s 64 -c 3

mkdir weights/WGAN-GP/cr/saliency/Malignant
mv runs/* weights/WGAN-GP/cr/saliency/Malignant
mv weights/WGAN-GP/cr/saliency/gen_epoch_100.pth weights/WGAN-GP/cr/saliency/Malignant
mv weights/WGAN-GP/cr/saliency/disc_epoch_100.pth weights/WGAN-GP/cr/saliency/Malignant