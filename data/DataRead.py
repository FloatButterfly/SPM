import os

from skimage.io import imread
from torch.utils.data import Dataset
from .base_dataset import *


class TrainPairedData(Dataset):
    def __init__(self, root, opt):
        super().__init__()
        self.opt = opt
        self.images = list()
        self.labels = list()
        if len(self.images) < 1:
            for item in os.listdir(os.path.join(root, "images")):
                img_dir = os.path.join(root, "images")
                img_dir = os.path.join(img_dir, item)
                if self.opt.labels_x8:
                    label_dir = os.path.join(root, "labels_x8")
                    if not os.path.exists(label_dir):
                        label_dir = os.path.join(root, "labels")
                else:
                    label_dir = os.path.join(root, "labels")

                if self.opt.dataset_mode == "cityscapes":
                    label_dir = os.path.join(label_dir, item)
                elif self.opt.dataset_mode == "ade20k":
                    label_dir = os.path.join(label_dir, str(item.split('.')[0]) + ".png")
                else:
                    label_dir = os.path.join(label_dir, str(item.split('_')[-1].split('.')[0]) + ".png")

                if not os.path.exists(label_dir):
                    continue
                else:
                    self.images.append(img_dir)
                    self.labels.append(label_dir)
            # print(self.images, self.labels)

    def load_dataset(self, root, name):
        if not os.path.exists(os.path.join(root, "images" + '_' + name)):
            return
        for item in os.listdir(os.path.join(root, "images" + '_' + name)):
            img_dir = os.path.join(root, "images" + '_' + name)
            img_dir = os.path.join(img_dir, item)
            label_dir = os.path.join(root, "labels" + '_' + name)
            # label_dir = os.path.join(label_dir, str((item.split('.')[0]).split('_')[-1]) + ".png")
            label_dir = os.path.join(label_dir, str(item.split('.')[0]) + ".png")
            if not os.path.exists(label_dir):
                continue
            else:
                self.images.append(img_dir)
                self.labels.append(label_dir)

    def paths_match(self, path1, path2):
        filename1_without_ext = os.path.splitext(os.path.basename(path1))[0]
        filename2_without_ext = os.path.splitext(os.path.basename(path2))[0]
        return filename1_without_ext == filename2_without_ext or 'input_' + filename1_without_ext == filename2_without_ext or filename1_without_ext == 'input_' + filename2_without_ext

    def postprocess(self, input_dict):
        label = input_dict['label']
        label = label - 1
        label[label == -1] = self.opt.label_nc

    def __getitem__(self, index):
        image_path = self.images[index]
        label_path = self.labels[index]
        assert self.paths_match(image_path, label_path), "The label_path %s and image_path %s don't match." % \
                                                         (label_path, image_path)
        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path)

        params = get_params(self.opt, image.size)
        transforms_image = get_transform(self.opt, params, method=Image.LANCZOS)
        transforms_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)

        image_tensor = transforms_image(image)
        label_tensor = transforms_label(label)
        label_tensor = label_tensor * 255

        # label_tensor[label_tensor == 255] = self.opt.label_nc

        input_dict = {'image': image_tensor,
                      'label': label_tensor,
                      'path': image_path,
                      }

        if self.opt.dataset_mode == "ade20k":
            self.postprocess(input_dict)

        return input_dict

    def __len__(self):
        return len(self.images)
