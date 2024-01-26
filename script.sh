#!/bin/bash

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

## -- CR - SALIENCY --------------
## -------- 1 --------
#rm -r datasets/LA64
#unzip -q datasets/LA64_original.zip
#mv LA64 datasets
#rm -r datasets/LA64/2
#rm -r datasets/LA64/3
#rm -r datasets/LA64/4
#
#python train.py --gan WGAN-GP --xai saliency -d la -s 64 -c 3
#
#mkdir weights/WGAN-GP/la/saliency/1
#mv runs/* weights/WGAN-GP/la/saliency/1
#mv weights/WGAN-GP/la/saliency/gen_epoch_100.pth weights/WGAN-GP/la/saliency/1
#mv weights/WGAN-GP/la/saliency/disc_epoch_100.pth weights/WGAN-GP/la/saliency/1
#
## -------- 2 --------
#rm -r datasets/LA64
#unzip -q datasets/LA64_original.zip
#mv LA64 datasets
#rm -r datasets/LA64/1
#rm -r datasets/LA64/3
#rm -r datasets/LA64/4
#
#python train.py --gan WGAN-GP --xai saliency -d la -s 64 -c 3
#
#mkdir weights/WGAN-GP/la/saliency/2
#mv runs/* weights/WGAN-GP/la/saliency/2
#mv weights/WGAN-GP/la/saliency/gen_epoch_100.pth weights/WGAN-GP/la/saliency/2
#mv weights/WGAN-GP/la/saliency/disc_epoch_100.pth weights/WGAN-GP/la/saliency/2
#
## -------- 3 --------
#rm -r datasets/LA64
#unzip -q datasets/LA64_original.zip
#mv LA64 datasets
#rm -r datasets/LA64/1
#rm -r datasets/LA64/2
#rm -r datasets/LA64/4
#
#python train.py --gan WGAN-GP --xai saliency -d la -s 64 -c 3
#
#mkdir weights/WGAN-GP/la/saliency/3
#mv runs/* weights/WGAN-GP/la/saliency/3
#mv weights/WGAN-GP/la/saliency/gen_epoch_100.pth weights/WGAN-GP/la/saliency/3
#mv weights/WGAN-GP/la/saliency/disc_epoch_100.pth weights/WGAN-GP/la/saliency/3
#
## -------- 4 --------
#rm -r datasets/LA64
#unzip -q datasets/LA64_original.zip
#mv LA64 datasets
#rm -r datasets/LA64/1
#rm -r datasets/LA64/2
#rm -r datasets/LA64/3
#
#python train.py --gan WGAN-GP --xai saliency -d la -s 64 -c 3
#
#mkdir weights/WGAN-GP/la/saliency/4
#mv runs/* weights/WGAN-GP/la/saliency/4
#mv weights/WGAN-GP/la/saliency/gen_epoch_100.pth weights/WGAN-GP/la/saliency/4
#mv weights/WGAN-GP/la/saliency/disc_epoch_100.pth weights/WGAN-GP/la/saliency/4
#
## -- CR - DEEPLIFT --------------
## -------- 1 --------
#rm -r datasets/LA64
#unzip -q datasets/LA64_original.zip
#mv LA64 datasets
#rm -r datasets/LA64/2
#rm -r datasets/LA64/3
#rm -r datasets/LA64/4
#
#python train.py --gan WGAN-GP --xai deeplift -d la -s 64 -c 3
#
#mkdir weights/WGAN-GP/la/deeplift/1
#mv runs/* weights/WGAN-GP/la/deeplift/1
#mv weights/WGAN-GP/la/deeplift/gen_epoch_100.pth weights/WGAN-GP/la/deeplift/1
#mv weights/WGAN-GP/la/deeplift/disc_epoch_100.pth weights/WGAN-GP/la/deeplift/1
#
## -------- 2 --------
#rm -r datasets/LA64
#unzip -q datasets/LA64_original.zip
#mv LA64 datasets
#rm -r datasets/LA64/1
#rm -r datasets/LA64/3
#rm -r datasets/LA64/4
#
#python train.py --gan WGAN-GP --xai deeplift -d la -s 64 -c 3
#
#mkdir weights/WGAN-GP/la/deeplift/2
#mv runs/* weights/WGAN-GP/la/deeplift/2
#mv weights/WGAN-GP/la/deeplift/gen_epoch_100.pth weights/WGAN-GP/la/deeplift/2
#mv weights/WGAN-GP/la/deeplift/disc_epoch_100.pth weights/WGAN-GP/la/deeplift/2
#
## -------- 3 --------
#rm -r datasets/LA64
#unzip -q datasets/LA64_original.zip
#mv LA64 datasets
#rm -r datasets/LA64/1
#rm -r datasets/LA64/2
#rm -r datasets/LA64/4
#
#python train.py --gan WGAN-GP --xai deeplift -d la -s 64 -c 3
#
#mkdir weights/WGAN-GP/la/deeplift/3
#mv runs/* weights/WGAN-GP/la/deeplift/3
#mv weights/WGAN-GP/la/deeplift/gen_epoch_100.pth weights/WGAN-GP/la/deeplift/3
#mv weights/WGAN-GP/la/deeplift/disc_epoch_100.pth weights/WGAN-GP/la/deeplift/3
#
## -------- 4 --------
#rm -r datasets/LA64
#unzip -q datasets/LA64_original.zip
#mv LA64 datasets
#rm -r datasets/LA64/1
#rm -r datasets/LA64/2
#rm -r datasets/LA64/3
#
#python train.py --gan WGAN-GP --xai deeplift -d la -s 64 -c 3
#
#mkdir weights/WGAN-GP/la/deeplift/4
#mv runs/* weights/WGAN-GP/la/deeplift/4
#mv weights/WGAN-GP/la/deeplift/gen_epoch_100.pth weights/WGAN-GP/la/deeplift/4
#mv weights/WGAN-GP/la/deeplift/disc_epoch_100.pth weights/WGAN-GP/la/deeplift/4
#
## -- CR - INPUTXGRAD --------------
## -------- 1 --------
#rm -r datasets/LA64
#unzip -q datasets/LA64_original.zip
#mv LA64 datasets
#rm -r datasets/LA64/2
#rm -r datasets/LA64/3
#rm -r datasets/LA64/4
#
#python train.py --gan WGAN-GP --xai inputxgrad -d la -s 64 -c 3
#
#mkdir weights/WGAN-GP/la/inputxgrad/1
#mv runs/* weights/WGAN-GP/la/inputxgrad/1
#mv weights/WGAN-GP/la/inputxgrad/gen_epoch_100.pth weights/WGAN-GP/la/inputxgrad/1
#mv weights/WGAN-GP/la/inputxgrad/disc_epoch_100.pth weights/WGAN-GP/la/inputxgrad/1
#
## -------- 2 --------
#rm -r datasets/LA64
#unzip -q datasets/LA64_original.zip
#mv LA64 datasets
#rm -r datasets/LA64/1
#rm -r datasets/LA64/3
#rm -r datasets/LA64/4
#
#python train.py --gan WGAN-GP --xai inputxgrad -d la -s 64 -c 3
#
#mkdir weights/WGAN-GP/la/inputxgrad/2
#mv runs/* weights/WGAN-GP/la/inputxgrad/2
#mv weights/WGAN-GP/la/inputxgrad/gen_epoch_100.pth weights/WGAN-GP/la/inputxgrad/2
#mv weights/WGAN-GP/la/inputxgrad/disc_epoch_100.pth weights/WGAN-GP/la/inputxgrad/2
#
## -------- 3 --------
#rm -r datasets/LA64
#unzip -q datasets/LA64_original.zip
#mv LA64 datasets
#rm -r datasets/LA64/1
#rm -r datasets/LA64/2
#rm -r datasets/LA64/4
#
#python train.py --gan WGAN-GP --xai inputxgrad -d la -s 64 -c 3
#
#mkdir weights/WGAN-GP/la/inputxgrad/3
#mv runs/* weights/WGAN-GP/la/inputxgrad/3
#mv weights/WGAN-GP/la/inputxgrad/gen_epoch_100.pth weights/WGAN-GP/la/inputxgrad/3
#mv weights/WGAN-GP/la/inputxgrad/disc_epoch_100.pth weights/WGAN-GP/la/inputxgrad/3
#
## -------- 4 --------
#rm -r datasets/LA64
#unzip -q datasets/LA64_original.zip
#mv LA64 datasets
#rm -r datasets/LA64/1
#rm -r datasets/LA64/2
#rm -r datasets/LA64/3
#
#python train.py --gan WGAN-GP --xai inputxgrad -d la -s 64 -c 3
#
#mkdir weights/WGAN-GP/la/inputxgrad/4
#mv runs/* weights/WGAN-GP/la/inputxgrad/4
#mv weights/WGAN-GP/la/inputxgrad/gen_epoch_100.pth weights/WGAN-GP/la/inputxgrad/4
#mv weights/WGAN-GP/la/inputxgrad/disc_epoch_100.pth weights/WGAN-GP/la/inputxgrad/4


# ------------------------
# ----------- LG ---------
# ------------------------
# ------------ SALIENCY ------------
# -------- 1 --------
rm -r datasets/LG64
unzip -q datasets/LG64_original.zip
mv LG64 datasets
rm -r datasets/LG64/Class\ 2

python train.py --gan WGAN-GP --xai saliency -d lg -s 64 -c 3

mkdir weights/WGAN-GP/lg/saliency/Class\ 1
mv runs/* weights/WGAN-GP/lg/saliency/Class\ 1
mv weights/WGAN-GP/lg/saliency/gen_epoch_100.pth weights/WGAN-GP/lg/saliency/Class\ 1
mv weights/WGAN-GP/lg/saliency/disc_epoch_100.pth weights/WGAN-GP/lg/saliency/Class\ 1

# -------- 2 --------
rm -r datasets/LG64
unzip -q datasets/LG64_original.zip
mv LG64 datasets
rm -r datasets/LG64/Class\ 1

python train.py --gan WGAN-GP --xai saliency -d lg -s 64 -c 3

mkdir weights/WGAN-GP/lg/saliency/Class\ 2
mv runs/* weights/WGAN-GP/lg/saliency/Class\ 2
mv weights/WGAN-GP/lg/saliency/gen_epoch_100.pth weights/WGAN-GP/lg/saliency/Class\ 2
mv weights/WGAN-GP/lg/saliency/disc_epoch_100.pth weights/WGAN-GP/lg/saliency/Class\ 2

# ------------ DEEPLIFT ------------
# -------- 1 --------
rm -r datasets/LG64
unzip -q datasets/LG64_original.zip
mv LG64 datasets
rm -r datasets/LG64/Class\ 2

python train.py --gan WGAN-GP --xai deeplift -d lg -s 64 -c 3

mkdir weights/WGAN-GP/lg/deeplift/Class\ 1
mv runs/* weights/WGAN-GP/lg/deeplift/Class\ 1
mv weights/WGAN-GP/lg/deeplift/gen_epoch_100.pth weights/WGAN-GP/lg/deeplift/Class\ 1
mv weights/WGAN-GP/lg/deeplift/disc_epoch_100.pth weights/WGAN-GP/lg/deeplift/Class\ 1

# -------- 2 --------
rm -r datasets/LG64
unzip -q datasets/LG64_original.zip
mv LG64 datasets
rm -r datasets/LG64/Class\ 1

python train.py --gan WGAN-GP --xai deeplift -d lg -s 64 -c 3

mkdir weights/WGAN-GP/lg/deeplift/Class\ 2
mv runs/* weights/WGAN-GP/lg/deeplift/Class\ 2
mv weights/WGAN-GP/lg/deeplift/gen_epoch_100.pth weights/WGAN-GP/lg/deeplift/Class\ 2
mv weights/WGAN-GP/lg/deeplift/disc_epoch_100.pth weights/WGAN-GP/lg/deeplift/Class\ 2

# ------------ INPUTXGRAD ------------
# -------- 1 --------
rm -r datasets/LG64
unzip -q datasets/LG64_original.zip
mv LG64 datasets
rm -r datasets/LG64/Class\ 2

python train.py --gan WGAN-GP --xai inputxgrad -d lg -s 64 -c 3

mkdir weights/WGAN-GP/lg/inputxgrad/Class\ 1
mv runs/* weights/WGAN-GP/lg/inputxgrad/Class\ 1
mv weights/WGAN-GP/lg/inputxgrad/gen_epoch_100.pth weights/WGAN-GP/lg/inputxgrad/Class\ 1
mv weights/WGAN-GP/lg/inputxgrad/disc_epoch_100.pth weights/WGAN-GP/lg/inputxgrad/Class\ 1

# -------- 2 --------
rm -r datasets/LG64
unzip -q datasets/LG64_original.zip
mv LG64 datasets
rm -r datasets/LG64/Class\ 1

python train.py --gan WGAN-GP --xai inputxgrad -d lg -s 64 -c 3

mkdir weights/WGAN-GP/lg/inputxgrad/Class\ 2
mv runs/* weights/WGAN-GP/lg/inputxgrad/Class\ 2
mv weights/WGAN-GP/lg/inputxgrad/gen_epoch_100.pth weights/WGAN-GP/lg/inputxgrad/Class\ 2
mv weights/WGAN-GP/lg/inputxgrad/disc_epoch_100.pth weights/WGAN-GP/lg/inputxgrad/Class\ 2


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
## -- UCSB64 - DEEPLIFT --------------
## -------- Benign --------
#rm -r datasets/UCSB64
#unzip -q datasets/UCSB64_original.zip
#mv UCSB64 datasets
#rm -r datasets/UCSB64/Malignant
#
#python train.py --gan WGAN-GP --xai deeplift -d ucsb -s 64 -c 3
#
#mkdir weights/WGAN-GP/ucsb/deeplift/Benign
#mv runs/* weights/WGAN-GP/ucsb/deeplift/Benign
#mv weights/WGAN-GP/ucsb/deeplift/gen_epoch_100.pth weights/WGAN-GP/ucsb/deeplift/Benign
#mv weights/WGAN-GP/ucsb/deeplift/disc_epoch_100.pth weights/WGAN-GP/ucsb/deeplift/Benign
#
## -------- Malignant --------
#rm -r datasets/UCSB64
#unzip -q datasets/UCSB64_original.zip
#mv UCSB64 datasets
#rm -r datasets/UCSB64/Benign
#
#python train.py --gan WGAN-GP --xai deeplift -d ucsb -s 64 -c 3
#
#mkdir weights/WGAN-GP/ucsb/deeplift/Malignant
#mv runs/* weights/WGAN-GP/ucsb/deeplift/Malignant
#mv weights/WGAN-GP/ucsb/deeplift/gen_epoch_100.pth weights/WGAN-GP/ucsb/deeplift/Malignant
#mv weights/WGAN-GP/ucsb/deeplift/disc_epoch_100.pth weights/WGAN-GP/ucsb/deeplift/Malignant

## -- UCSB64 - INPUTXGRAD --------------
## -------- Benign --------
#rm -r datasets/UCSB64
#unzip -q datasets/UCSB64_original.zip
#mv UCSB64 datasets
#rm -r datasets/UCSB64/Malignant
#
#python train.py --gan WGAN-GP --xai inputxgrad -d ucsb -s 64 -c 3
#
#mkdir weights/WGAN-GP/ucsb/inputxgrad/Benign
#mv runs/* weights/WGAN-GP/ucsb/inputxgrad/Benign
#mv weights/WGAN-GP/ucsb/inputxgrad/gen_epoch_100.pth weights/WGAN-GP/ucsb/inputxgrad/Benign
#mv weights/WGAN-GP/ucsb/inputxgrad/disc_epoch_100.pth weights/WGAN-GP/ucsb/inputxgrad/Benign
#
## -------- Malignant --------
#rm -r datasets/UCSB64
#unzip -q datasets/UCSB64_original.zip
#mv UCSB64 datasets
#rm -r datasets/UCSB64/Benign
#
#python train.py --gan WGAN-GP --xai inputxgrad -d ucsb -s 64 -c 3
#
#mkdir weights/WGAN-GP/ucsb/inputxgrad/Malignant
#mv runs/* weights/WGAN-GP/ucsb/inputxgrad/Malignant
#mv weights/WGAN-GP/ucsb/inputxgrad/gen_epoch_100.pth weights/WGAN-GP/ucsb/inputxgrad/Malignant
#mv weights/WGAN-GP/ucsb/inputxgrad/disc_epoch_100.pth weights/WGAN-GP/ucsb/inputxgrad/Malignant