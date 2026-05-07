#!/bin/bash

CFG=rn50_ep50
CTP=end
NCTX=16
SHOTS1=16
SHOTS2=4
CSC1=False
#CSC2=True
LOSS=custom

# List of datasets (adjust based on what you've downloaded)
DATASETS=(
    caltech101
    dtd
    #eurosat
    fgvc_aircraft
    #food101
    oxford_flowers
    oxford_pets
    #stanford_cars
    #sun397
    ucf101
)

for DATASET in "${DATASETS[@]}"
do
    echo "======================================"
    echo "Running dataset: $DATASET"
    echo "======================================"

    bash scripts/coop/main.sh $DATASET $CFG $CTP $NCTX $SHOTS1 $CSC1 $LOSS
   # bash scripts/coop/main.sh $DATASET $CFG $CTP $NCTX $SHOTS1 $CSC2 $LOSS
    bash scripts/coop/main.sh $DATASET $CFG $CTP $NCTX $SHOTS2 $CSC1 $LOSS
    #bash scripts/coop/main.sh $DATASET $CFG $CTP $NCTX $SHOTS2 $CSC2 $LOSS

    echo "Finished dataset: $DATASET"
done