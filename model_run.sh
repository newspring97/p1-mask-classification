python train.py --epoch 25 --batch_size 32 --val_batch_size 10 --model ResNet_Model --augmentation CenterCropAugmentation
python train.py --epoch 25 --batch_size 32 --val_batch_size 10 --model EfficientNet_Model --augmentation CenterCropAugmentation
python train.py --epoch 25 --batch_size 32 --val_batch_size 10 --model EfficientNet_B2_Model --augmentation CenterCropAugmentation

echo "Done"
