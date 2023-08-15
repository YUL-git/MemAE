import os
import pickle
import numpy as np
from skimage.transform import resize
from torchvision import datasets, transforms

class MNIST_Dataset():
    def __init__(self, opt):
        self.dataset_dir = opt.data_path
        self.class_name = opt.normal_class
        self.image_height = opt.img_height
        self.image_width = opt.img_width
        self.image_channel_size = opt.img_chn_size

        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.ToPILImage(),
                                             transforms.ToTensor(),
                                             transforms.Grayscale(num_output_channels=1),
                                             transforms.Normalize((0.5,), (0.5,)),
                                            ])
        os.makedirs('./Train/',exist_ok=True)
        os.makedirs('./Valid/',exist_ok=True)

        self.load_dataset()
        self.preprocess_dataset()

        class_train_file = os.path.join('./Train/', f'class_{self.class_name}')
        with open(class_train_file, 'rb') as f:
            self.train_dataset = pickle.load(f)

        self.valid_dataset = []
        for i in range(10):
            if i == self.class_name:
                class_valid_thres_file = os.path.join('./Valid/', f'class_{self.class_name}')
                with open(class_valid_thres_file, 'rb') as f:
                    self.valid_thres_dataset = pickle.load(f)
                
                class_valid_file = os.path.join('./Valid/', f'class_{i}')
                with open(class_valid_file, 'rb') as f:
                    v_dataset = pickle.load(f)
            else:
                class_valid_file = os.path.join('./Valid/', f'class_{i}')
                with open(class_valid_file, 'rb') as f:
                    v_dataset = pickle.load(f)
                    v_dataset = v_dataset[:int(len(v_dataset)*0.03)]
                self.valid_dataset.extend(v_dataset)


    def load_dataset(self):
        self.raw_train_dataset = datasets.MNIST(root=self.dataset_dir,
                                                train=True,
                                                download=True,
                                                transform=None)

        self.raw_test_dataset  = datasets.MNIST(root=self.dataset_dir,
                                                train=False,
                                                download=True,
                                                transform=None)

    def preprocess_dataset(self):
        for i in range(10):
            _dataset = []

            for j in range(len(self.raw_train_dataset.data)):
                if self.raw_train_dataset.targets[j] == i:
                    _dataset.append(self.raw_train_dataset[j])

            dataset = []
            for j, (image, label) in enumerate(_dataset):
                img = np.array(image)
                lab = np.array(label)
                img = np.expand_dims(img, axis=2)
                img = resize(img, (28, 28), anti_aliasing=True)
                img = img.astype(np.float32)
                dataset.append((self.transform(img), label))

            split = int(len(dataset)*0.1)
            train_dataset = dataset[split:]
            valid_dataset = dataset[:split]

            train_dir = f'./Train/class_{i}'
            valid_dir = f'./Valid/class_{i}'
            with open(train_dir, 'wb') as f:
                pickle.dump(train_dataset, f)
            with open(valid_dir, 'wb') as f:
                pickle.dump(valid_dataset, f)

        for i in range(10):
            _dataset = []

            for j in range(len(self.raw_test_dataset.data)):
                if self.raw_test_dataset.targets[j] == i:
                    _dataset.append(self.raw_test_dataset[j])

            dataset = []
            for j, (image,label) in enumerate(_dataset):
                img = np.array(image)
                lab = np.array(label)
                img = np.expand_dims(img, axis=2)
                img = resize(img, (28, 28), anti_aliasing=True)
                img = img.astype(np.float32)
                dataset.append((self.transform(img), label))

            test_dir = f'./Train/class_{i}'
            existing_data = None
            with open(test_dir, 'rb') as f:
                existing_data = pickle.load(f)
            existing_data.extend(dataset)

            with open(test_dir, 'wb') as f:
                pickle.dump(existing_data, f)