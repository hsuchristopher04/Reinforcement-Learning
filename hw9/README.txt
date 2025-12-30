

RL HW 9

Authors: Aditya Dhodapkar (dhodaa), Chris Hsu (hsuc3)

----- Models and their locations -----
  the cnn_buildings_best.pth is included in the zip file
  our Cifar and Building models are in the scratch/models folder (dhodaa) on the server
  train and val labels are also on the server (dhodaa) in barn folder


----- How to Run -----
  training buildings model: srun -t 120 --gres=gpu:1 python train_buildings.py

  test buildings model: python test.py [test set dir] [path to labels file]

  training CIFAR-10 Model: srun -t 60 --gres=gpu:1 python cnn_classification_cifar10.py

