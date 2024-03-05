import os
import cv2
import toml
import argparse
import numpy as np

import torch
from torch.nn import functional as F

import utils
from   utils import CONFIG
import networks
import networks.classifer as classifer
from PIL import Image
import os 
import math

def get_cls_mask(model, image, trimap):
        model.eval()
        with torch.no_grad():
            out = model(torch.cat([image, trimap], dim=1))
            torch.cuda.empty_cache()
        return out
    

def to_one_hot(cls_mask, n_classes):
    _,tensor = torch.max(cls_mask,dim=1)
    n, h, w = tensor.size()
    one_hot = torch.zeros(n, n_classes, h, w).cuda().scatter_(1, tensor.long().view(n, 1, h, w), 1)
    return one_hot

def single_inference(model,classifer, image_dict):

    with torch.no_grad(): 
        image, trimap = image_dict['image'], image_dict['trimap']

        image = image.cuda()
        trimap = trimap.cuda()
        classifer = classifer.cuda()
        # run model
        cls_mask = get_cls_mask(classifer,image,trimap)
        
        pred = model(image, trimap,cls_mask)
        alpha_pred_os1, alpha_pred_os4, alpha_pred_os8 = pred['alpha_os1'], pred['alpha_os4'], pred['alpha_os8']
        
        # refinement
        alpha_pred = alpha_pred_os8.clone().detach()
        weight_os4 = utils.get_unknown_tensor_from_pred(alpha_pred, rand_width=CONFIG.model.self_refine_width1, train_mode=False)
        alpha_pred[weight_os4>0] = alpha_pred_os4[weight_os4>0]
        weight_os1 = utils.get_unknown_tensor_from_pred(alpha_pred, rand_width=CONFIG.model.self_refine_width2, train_mode=False)
        alpha_pred[weight_os1>0] = alpha_pred_os1[weight_os1>0]
        

        h, w = image_dict['alpha_shape']
        alpha_pred = alpha_pred[0, 0, ...].data.cpu().numpy() * 255
        alpha_pred = alpha_pred.astype(np.uint8)

        alpha_pred[np.argmax(trimap.cpu().numpy()[0], axis=0) == 0] = 0.0
        alpha_pred[np.argmax(trimap.cpu().numpy()[0], axis=0) == 2] = 255.

        alpha_pred = alpha_pred[32:h+32, 32:w+32]
        
        return alpha_pred


def generator_tensor_dict(image_path, trimap_path):
    # read images
    image = cv2.imread(image_path)
    trimap = cv2.imread(trimap_path, 0)

    w =  image.shape[1]
    h =  image.shape[0]
    sample = {'image': image, 'trimap':trimap, 'alpha_shape':(image.shape[0], image.shape[1])}
    sample['image1'] = image
    # reshape
    h, w = sample["alpha_shape"]
    
    if h % 32 == 0 and w % 32 == 0:
        padded_image = np.pad(sample['image'], ((32,32), (32, 32), (0,0)), mode="reflect")
        padded_trimap = np.pad(sample['trimap'], ((32,32), (32, 32)), mode="reflect")

        sample['image'] = padded_image
        sample['trimap'] = padded_trimap

    else:
        target_h = 32 * ((h - 1) // 32 + 1)
        target_w = 32 * ((w - 1) // 32 + 1)
        pad_h = target_h - h
        pad_w = target_w - w
        padded_image = np.pad(sample['image'], ((32,pad_h+32), (32, pad_w+32), (0,0)), mode="reflect")
        padded_trimap = np.pad(sample['trimap'], ((32,pad_h+32), (32, pad_w+32)), mode="reflect")
        
        sample['image'] = padded_image
        sample['trimap'] = padded_trimap
    # ImageNet mean & std
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

    # convert GBR images to RGB
    image, trimap = sample['image'][:,:,::-1], sample['trimap']

    # swap color axis
    image = image.transpose((2, 0, 1)).astype(np.float32)

    # trimap configuration
    padded_trimap[padded_trimap < 85] = 0
    padded_trimap[padded_trimap >= 170] = 2
    padded_trimap[padded_trimap >= 85] = 1

    # normalize image
    image /= 255.

    # to tensor
    sample['image'], sample['trimap'] = torch.from_numpy(image), torch.from_numpy(trimap).to(torch.long)
    sample['image'] = sample['image'].sub_(mean).div_(std)

    # trimap to one-hot 3 channel
    sample['trimap'] = F.one_hot(sample['trimap'], num_classes=3).permute(2, 0, 1).float()
    # add first channel
    sample['image'], sample['trimap'] = sample['image'][None, ...], sample['trimap'][None, ...]

    return sample

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model_mse.pth', help="path of checkpoint")

    # local
    parser.add_argument('--image-dir', type=str, default='Composition-1k-testset/merged', help="input image dir")
    parser.add_argument('--trimap-dir', type=str, default='Composition-1k-testset/trimaps', help="input trimap dir")
    parser.add_argument('--output', type=str, default='output', help="output dir")

    # Parse configuration
    args = parser.parse_args()

    utils.make_dir(os.path.join(args.output, 'pred_alpha'))
    model = networks.get_generator(is_train=False)
    model.cuda()

    # load checkpoint
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(utils.remove_prefix_state_dict(checkpoint['state_dict']), strict=True)
    # inference
    model = model.eval()
    classfy = classifer.build_classifier() 
    classfy = classfy.eval()
    for i, image_name in enumerate(os.listdir(args.image_dir)):

        # assume image and mask have the same file name
        img_name = image_name
        
        image_path = os.path.join(args.image_dir, img_name)
        trimap_path = os.path.join(args.trimap_dir, img_name)
        image_dict = generator_tensor_dict(image_path, trimap_path)
        
        alpha_pred = single_inference(model,classfy, image_dict)

        cv2.imwrite(os.path.join(args.output, 'pred_alpha', image_name), alpha_pred)
        print('[{}/{}] inference done : {}'.format(i, len(os.listdir(args.image_dir)), os.path.join(args.output, 'pred_alpha', image_name)))
