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

# CR
python train.py --gan WGAN-GP --xai saliency -d cr -s 64 -c 3
python train.py --gan WGAN-GP --xai deeplift -d cr -s 64 -c 3
python train.py --gan WGAN-GP --xai inputxgrad -d cr -s 64 -c 3
python train.py --gan WGAN-GP -d cr -s 64 -c 3

# UCSB
python train.py --gan WGAN-GP --xai saliency -d ucsb -s 64 -c 3
python train.py --gan WGAN-GP --xai deeplift -d ucsb -s 64 -c 3
python train.py --gan WGAN-GP --xai inputxgrad -d ucsb -s 64 -c 3
python train.py --gan WGAN-GP -d ucsb -s 64 -c 3