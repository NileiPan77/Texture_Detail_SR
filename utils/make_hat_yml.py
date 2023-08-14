import argparse
import time
import sys
parser = argparse.ArgumentParser()
default_output_name = "HAT_GAN_Real_SR_" + time.strftime("%Y%m%d%H%M%S", time.localtime())
parser.add_argument('--input', type=str, default='.\datasets\RealSR', help='input folder')
parser.add_argument('--output_name', type=str, default=default_output_name, help='output name')
parser.add_argument('--out_dir', type=str, default='.\options\\', help='output folder')
parser.add_argument('--num_gpu', type=int, default=1, help='number of gpu')
parser.add_argument('--disable_tile_mode', type=bool, default=False, help='Disable tile mode if there are enough GPU memory')
parser.add_argument('--tile_size', type=int, default=128, help='tile size')
parser.add_argument('--tile_pad', type=int, default=64, help='tile pad')
parser.add_argument('--scale', type=int, default=4, help='scale factor, [2 ,3 ,4] are supported')
args = parser.parse_args()

if args.scale not in [2, 3, 4]:
    raise ValueError("Scale factor should be 2, 3 or 4.")

tile_mode = f"tile:\n  tile_size: {args.tile_size}\n  tile_pad: {args.tile_pad}\n" if not args.disable_tile_mode else ""
pretrained_model = [
    "./experiments/pretrained_models/HAT_SRx2.pth", "./experiments/pretrained_models/HAT_SRx3.pth", "./experiments/pretrained_models/HAT_SRx4.pth"
]
template_yml = f"""
name: {args.output_name}
model_type: HATModel
scale: {args.scale}
num_gpu: {args.num_gpu}  # set num_gpu: 0 for cpu mode
manual_seed: 0

{tile_mode}

datasets:
  test_1:  # the 1st test dataset
    name: custom
    type: SingleImageDataset
    dataroot_lq: {args.input}
    io_backend:
      type: disk

# network structures
network_g:
  type: HAT
  upscale: {args.scale}
  in_chans: 3
  img_size: 64
  window_size: 16
  compress_ratio: 3
  squeeze_factor: 30
  conv_scale: 0.01
  overlap_ratio: 0.5
  img_range: 1.
  depths: [6, 6, 6, 6, 6, 6]
  embed_dim: 180
  num_heads: [6, 6, 6, 6, 6, 6]
  mlp_ratio: 2
  upsampler: 'pixelshuffle'
  resi_connection: '1conv'


# path
path:
  pretrain_network_g: {pretrained_model[args.scale - 2]}  # path of the pretrained model
  strict_load_g: true
  param_key_g: 'params_ema'

# validation settings
val:
  save_img: true
  suffix: ~  # add suffix to saved images, if None, use exp name

"""

output_dir = args.out_dir + args.output_name + ".yml"
with open(output_dir, "w") as f:
    f.write(template_yml)
print(output_dir)