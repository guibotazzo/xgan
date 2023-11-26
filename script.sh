#!/bin/bash

###############################
# DCGAN (inputxgrad) on CIFAR10
###############################

GAN='DCGAN'
DATASET='cifar10'
FOLDER='weights/'$GAN'/'$DATASET'/'

for i in 1 2 3 4 5
do
python train.py --gan $GAN -d $DATASET
mkdir $FOLDER'run'$i'/'
mv $FOLDER'disc_epoch_100.pth' $FOLDER'run'$i'/'
mv $FOLDER'gen_epoch_100.pth' $FOLDER'run'$i'/'
mv runs/* $FOLDER'run'$i'/'
rm $FOLDER'*.pth'
done

###############################
# DCGAN (inputxgrad) on CIFAR10
###############################

#GAN='DCGAN'
#XAI='saliency'
#DATASET='cifar10'
#FOLDER='weights/'$GAN'/'$DATASET'/'$XAI'/'
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

###############################
# DCGAN (inputxgrad) on CIFAR10
###############################

#GAN='DCGAN'
#XAI='deeplift'
#DATASET='cifar10'
#FOLDER='weights/'$GAN'/'$DATASET'/'$XAI'/'
#
#for i in 1 2 3 4 5
#do
#python train.py --gan $GAN -d $DATASET --xai $XAI
#mkdir $FOLDER'run'$i'/'
#mv $FOLDER'disc_epoch_100.pth' $FOLDER'run'$i'/'
#mv $FOLDER'gen_epoch_100.pth' $FOLDER'run'$i'/'
#mv runs/* $FOLDER'run'$i'/'
#rm $FOLDER'*.pth'
#done

#################################
# WGAN-GP (saliency) on CIFAR10
#################################

#GAN='WGAN-GP'
#XAI='saliency'
#DATASET='cifar10'
#FOLDER='weights/'$GAN'/'$DATASET'/'$XAI'/'
#
#for i in 2 3 4 5
#do
#python train.py --gan $GAN -d $DATASET --xai $XAI
#mkdir $FOLDER'run'$i'/'
#mv $FOLDER'disc_epoch_100.pth' $FOLDER'run'$i'/'
#mv $FOLDER'gen_epoch_100.pth' $FOLDER'run'$i'/'
#mv runs/* $FOLDER'run'$i'/'
#rm $FOLDER'*.pth'
#done

#################################
# WGAN-GP (inputxgrad) on CIFAR10
#################################

#GAN='WGAN-GP'
#XAI='inputxgrad'
#DATASET='cifar10'
#FOLDER='weights/'$GAN'/'$DATASET'/'$XAI'/'
#
#for i in 4 5
#do
#python train.py --gan $GAN -d $DATASET --xai $XAI
#mkdir $FOLDER'run'$i'/'
#mv $FOLDER'disc_epoch_100.pth' $FOLDER'run'$i'/'
#mv $FOLDER'gen_epoch_100.pth' $FOLDER'run'$i'/'
#mv runs/* $FOLDER'run'$i'/'
#rm $FOLDER'*.pth'
#done