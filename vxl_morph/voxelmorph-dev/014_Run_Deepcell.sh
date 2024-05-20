export DATA_DIR=/path/to/data/dir # your local folder
export MOUNT_DIR=/data # you don't need to change this
export APPLICATION=mesmer
export NUCLEAR_FILE=example_DAPI_image.tif

docker run -it --gpus 1 \
  -v $DATA_DIR:$MOUNT_DIR \
  vanvalenlab/deepcell-applications:latest-gpu \
  $APPLICATION \
  --nuclear-image $MOUNT_DIR/$NUCLEAR_FILE \
  --output-directory $MOUNT_DIR \
  --output-name mask.tif \
  --compartment nuclear \
  --image-mpp 0.648