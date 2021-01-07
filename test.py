import argparse

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import glob
import PIL.Image as pil_image

from models import RDN
from utils import convert_rgb_to_y, denormalize, calc_psnr


if __name__ == '__main__':
    weight = '/content/drive/MyDrive/DNN_HW4/epoch_299.pth'
    imgPath = '/content/drive/MyDrive/DNN_HW4/testing_lr_images'
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, required=True)
    parser.add_argument('--image-files', type=str, required=True)
    parser.add_argument('--num-features', type=int, default=64)
    parser.add_argument('--growth-rate', type=int, default=64)
    parser.add_argument('--num-blocks', type=int, default=16)
    parser.add_argument('--num-layers', type=int, default=8)
    parser.add_argument('--scale', type=int, default=3)
    args = parser.parse_args()

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = RDN(scale_factor=args.scale,
                num_channels=3,
                num_features=args.num_features,
                growth_rate=args.growth_rate,
                num_blocks=args.num_blocks,
                num_layers=args.num_layers).to(device)

    state_dict = model.state_dict()
    for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    model.eval()

    image_list = sorted(glob.glob('{}/*'.format(args.image_files)))
    for i, image_path in enumerate(image_list):
        image = pil_image.open(image_path).convert('RGB')

        lr = np.expand_dims(np.array(image).astype(np.float32).transpose([2, 0, 1]), 0) / 255.0
        lr = torch.from_numpy(lr).to(device)
        
        with torch.no_grad():
            preds = model(lr).squeeze(0)

        preds_y = convert_rgb_to_y(denormalize(preds), dim_order='chw')
       
        preds_y = preds_y[args.scale:-args.scale, args.scale:-args.scale]
       
        output = pil_image.fromarray(denormalize(preds).permute(1, 2, 0).byte().cpu().numpy())
        path = '/content/drive/MyDrive/DNN_HW4/output/'+image_path[-6:-4]+'.png'
        output.save(path)
