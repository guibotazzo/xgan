#!/bin/bash

EPOCHS=("10" "20" "30" "40" "50" "60" "70" "80" "90" "100" "110" "120" "130" "140" "150" "160" "170" "180" "190" "200")
DATASET=("cr")  # DATASET=("cr" "la" "lg" "ucsb" "nhl")
GAN=("WGAN-GP") # GAN=("DCGAN" "WGAN-GP" "RaSGAN")
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
        for epoch in "${EPOCHS[@]}"
        do

          python quality_eval.py --dataset "$dataset" --gan "$gan" --xai "$xai" --label "$current_class" --epoch "$epoch"

        done
      done
    done
  done

done
