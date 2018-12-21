# 3D-VAE-GAN
an tensorflow implemention of 3D-VAE-GAN for single image 3D reconstruction

I use tfrecord as the input style of the tensorflow

first use the code in data/LivelQADataset.py for tfrecord generation
```
    dataset = LiveIQADataset()
    sy_path ='/home/afan/Reconstruction/RenderForTrain/RenderResult/crop_images/03001627'
    binvox_dir = './ShapeNetVox64'
    train_path = './train.txt'
    test_path = './test.txt'
    dataset.generateTrainTestTF(sy_path, binvox_dir, train_path, test_path)
```

then run the scipy in nets/runVGAN.py for the training
