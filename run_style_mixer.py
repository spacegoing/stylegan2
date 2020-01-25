# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

import argparse
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import re
import sys
import glob

import pretrained_networks

#----------------------------------------------------------------------------

def style_mixing_example(network_pkl, row_imgs_dir, col_imgs_dir, col_ratio, col_styles, minibatch_size=4):
    print('Loading networks from "%s"...' % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)

    Gs_syn_kwargs = dnnlib.EasyDict()
    Gs_syn_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_syn_kwargs.randomize_noise = False
    Gs_syn_kwargs.minibatch_size = minibatch_size

    print("Loading W vectors...")
    row_dlatents_files = glob.glob(row_imgs_dir + '/*.npy')
    col_dlatents_files = glob.glob(col_imgs_dir + '/*.npy')
    print(row_dlatents_files)
    print(col_dlatents_files)

    row_dlatents = []
    col_dlatents = []
    for row_file in row_dlatents_files:
        row_dlatents.append(np.load(row_file))

    for col_file in col_dlatents_files:
        col_dlatents.append(np.load(col_file))

    all_w = np.array(row_dlatents + col_dlatents)

    all_files = row_dlatents_files + col_dlatents_files

    w_dict = {fl: w for fl, w in zip(all_files, list(all_w))} # [layer, component]

    print('Generating images...')
    all_images = Gs.components.synthesis.run(all_w, **Gs_syn_kwargs) # [minibatch, height, width, channel]
    image_dict = {(files, files): image for files, image in zip(all_files, list(all_images))}

    print('Generating style-mixed images...')
    # Instead of completely replacing the column styles from another image, I felt results are better if you mix them
    # in a ratio. Here, by default mixing 70% of column image with 30% of row image.
    # Original idea from https://github.com/iyaja/stylegan-encoder/blob/master/generate_GoT_characters_with_StyleGAN.ipynb
    for row_file in row_dlatents_files:
        for col_file in col_dlatents_files:
            w = w_dict[row_file].copy()
            avg_col_styles = w_dict[col_file][col_styles]*col_ratio + w_dict[row_file][col_styles]*(1-col_ratio)
            w[col_styles] = avg_col_styles
            image = Gs.components.synthesis.run(w[np.newaxis], **Gs_syn_kwargs)[0]
            image_dict[(row_file, col_file)] = image

    print('Saving images...')
    for (row_file, col_file), image in image_dict.items():
        r = row_file.split('/')[-1].replace('_01.npy', '')
        c = col_file.split('/')[-1].replace('_01.npy', '')
        PIL.Image.fromarray(image, 'RGB').save(dnnlib.make_run_dir_path('%s-%s.png' % (r, c)))

    print('Saving image grid...')
    _N, _C, H, W = Gs.output_shape
    canvas = PIL.Image.new('RGB', (W * (len(col_dlatents_files) + 1), H * (len(row_dlatents_files) + 1)), 'black')
    for row_idx, row_file in enumerate([None] + row_dlatents_files):
        for col_idx, col_file in enumerate([None] + col_dlatents_files):
            if row_file is None and col_file is None:
                continue
            key = (row_file, col_file)
            if row_file is None:
                key = (col_file, col_file)
            if col_file is None:
                key = (row_file, row_file)
            canvas.paste(PIL.Image.fromarray(image_dict[key], 'RGB'), (W * col_idx, H * row_idx))
    canvas.save(dnnlib.make_run_dir_path('grid.png'))

#----------------------------------------------------------------------------

def _parse_num_range(s):
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return range(int(m.group(1)), int(m.group(2))+1)
    vals = s.split(',')
    return [int(x) for x in vals]

#----------------------------------------------------------------------------

_examples = '''examples:

  # Generate style mixing example (matches style mixing video clip)
  python %(prog)s style-mixing-example --network=ffhq_network/stylegan2-ffhq-config-f.pkl
  --row-imgs-dir=generated_images --col-imgs-dir=generated_images
  --col-styles='0-6' --col-ratio=0.7
'''

#----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='''StyleGAN2 generator.

Run 'python %(prog)s <subcommand> --help' for subcommand help.''',
        epilog=_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(help='Sub-commands', dest='command')

    parser_style_mixing_example = subparsers.add_parser('style-mixing-example', help='Generate style mixing video')
    parser_style_mixing_example.add_argument('--network',
                            default='ffhq_network/stylegan2-ffhq-config-f.pkl',
                            help='Network pickle filename', dest='network_pkl') #, required=True)
    parser_style_mixing_example.add_argument('--row-imgs-dir', 
                            help='Files for image rows', required=True)
    parser_style_mixing_example.add_argument('--col-imgs-dir', 
                            help='Files for image columns', required=True)
    parser_style_mixing_example.add_argument('--col-styles', 
                            type=_parse_num_range, 
                            help='Style layer range (default: %(default)s)', default='0-6')
    parser_style_mixing_example.add_argument('--col-ratio', 
                            type=float, 
                            help='Column Ratio (default: %(default)s)', default=0.7)
    parser_style_mixing_example.add_argument('--result-dir', 
                            help='Root directory for run results (default: %(default)s)', 
                            default='results', metavar='DIR')

    args = parser.parse_args()
    kwargs = vars(args)
    subcmd = kwargs.pop('command')

    if subcmd is None:
        print ('Error: missing subcommand.  Re-run with --help for usage.')
        sys.exit(1)

    sc = dnnlib.SubmitConfig()
    sc.num_gpus = 1
    sc.submit_target = dnnlib.SubmitTarget.LOCAL
    sc.local.do_not_copy_source_files = True
    sc.run_dir_root = kwargs.pop('result_dir')
    sc.run_desc = subcmd

    func_name_map = {
        'style-mixing-example': 'run_style_mixer.style_mixing_example'
    }
    dnnlib.submit_run(sc, func_name_map[subcmd], **kwargs)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
