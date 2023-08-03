python tool/train.py --model unet
python tool/train_revise.py --model unetplusplus --checkpoint_name checkpoints/epoch=74_train_loss=0.158447_train_DiceScore=0.989665
python tool/train_revise.py --model linknet --checkpoint_name checkpoints/epoch=74_train_loss=0.301469_train_DiceScore=0.959865