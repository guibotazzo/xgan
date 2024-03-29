#!/bin/bash

DATASET=("cr") #DATASET=("cr" "la" "lg" "ucsb" "nhl")
IMG_SIZE=64
CHANNELS=3
EPOCHS=200

GAN=("WGAN-GP") #GAN=("DCGAN" "WGAN-GP" "RaSGAN")
XAI=("none" "saliency" "deeplift" "inputxgrad")

for dataset in "${DATASET[@]}"
do

  if [ "$dataset" = "cr" ]; then
    CLASSES=("Benign" "Malignant")
  elif [ "$dataset" = "la" ]; then
    CLASSES=("1" "2" "3" "4")
  elif [ "$dataset" = "lg" ]; then
    CLASSES=("Class 1" "Class 2")
  elif [ "$dataset" = "ucsb" ]; then
    CLASSES=("Benign" "Malignant")
  elif [ "$dataset" = "nhl" ]; then
    CLASSES=("CLL" "FL" "MCL")
  fi

  for gan in "${GAN[@]}"
  do
    for xai in "${XAI[@]}"
    do
      for current_class in "${CLASSES[@]}"
      do
        python train.py -d "$dataset" -s "$IMG_SIZE" -c "$CHANNELS" -e "$EPOCHS" --gan "$gan" --xai "$xai" --label "$current_class"

        # Create a folder to save the results of the current class
        CURRENT_DIR=weights/"$gan"/"$dataset"/"$xai"/
        OUT_DIR="$CURRENT_DIR""$current_class"/
        mkdir "$OUT_DIR"
        mv runs/* "$OUT_DIR"
        mv "$CURRENT_DIR"gen_epoch_*.pth "$OUT_DIR"
        mv "$CURRENT_DIR"disc_epoch_*.pth "$OUT_DIR"

      done # End for current_class...
    done # End for xai...
  done # End for gan...
done # End for dataset...
