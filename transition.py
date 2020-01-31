import numpy as np
import random
import glob
import re
import argparse
import moviepy.editor

import dnnlib
import dnnlib.tflib as tflib
import pretrained_networks


def get_latent_interpolation(endpoints, num_frames_per, mode, shuffle):
    if shuffle:
        random.shuffle(endpoints)
    num_endpoints, dim1, dim2 = len(endpoints), len(endpoints[0]), len(endpoints[0][0])
    num_frames = num_frames_per * num_endpoints
    endpoints = np.array(endpoints)
    latents = np.zeros((num_frames, dim1, dim2))
    for e in range(num_endpoints):
        e1, e2 = e, (e+1)%num_endpoints
        for t in range(num_frames_per):
            frame = e * num_frames_per + t
            r = 0.5 - 0.5 * np.cos(np.pi*t/(num_frames_per-1)) if mode == 'ease' else float(t) / num_frames_per
            latents[frame, :] = (1.0-r) * endpoints[e1,:] + r * endpoints[e2,:]
    return latents


def render_video(set_images, filename, fps=30, codec='libx264', bitrate='5M'):

    def render_frame(t):
        frame = int(np.clip(np.ceil(t * fps), 1, num_frames))
        return set_images[frame]

    num_frames = len(set_images)
    duration = num_frames / fps
    print(f"num_frames: {num_frames}, duration: {duration}")
    video_clip = moviepy.editor.VideoClip(render_frame, duration=duration)
    video_clip.write_videofile(filename, fps=fps, codec=codec, bitrate=bitrate)


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    from https://stackoverflow.com/a/5967539
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


def generate_transition(src_dir, network_pkl, fps, fpe, shrink):
    print("Loading W vectors...")
    dlatents_files = glob.glob(src_dir + '/*.npy')
    dlatents_files.sort(key=natural_keys)
    print(dlatents_files)

    tflib.init_tf()
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)

    Gs_syn_kwargs = dnnlib.EasyDict()
    # shrink>1 to reduce the output dimension of each image.
    Gs_syn_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True, shrink=shrink)
    Gs_syn_kwargs.randomize_noise = False
    Gs_syn_kwargs.minibatch_size = 4

    dlatents = []
    for f in dlatents_files:
        dlatents.append(np.load(f))

    latents = get_latent_interpolation(dlatents, fpe, 'linear', False)
    images = Gs.components.synthesis.run(latents, **Gs_syn_kwargs) # [minibatch, height, width, channel]

    # stay at the actual endpoints for a while, so that it is clear.
    repeat_count = 20 # repeat same image for few frames.
    for offset, jump in enumerate(range(0, len(dlatents) * fpe, fpe)):
        idx = jump + repeat_count*offset # calculate index of image to be repeated.
        repeat_image = np.repeat(np.expand_dims(images[idx], axis=0), repeat_count, axis=0)
        images = np.insert(images, idx, repeat_image, axis=0) # insert repeated image at index.

    # vid_name = src_dir + 'transition_%04d'%int(1000*random.random()) + '.mp4'
    base_name = src_dir.split('/')[-2]
    vid_name = dnnlib.make_run_dir_path(f'transition_{base_name}.mp4')
    render_video(images, vid_name, fps)


def main():
    parser = argparse.ArgumentParser(description='Get the transition of projected images from a given directory')
    parser.add_argument('src_dir', help='Directory with projected npy images')
    # parser.add_argument('--network-pkl', default='gdrive:networks/stylegan2-ffhq-config-f.pkl', help='StyleGAN2 network pickle filename')
    parser.add_argument('--network-pkl', default='models/stylegan2-ffhq-config-f.pkl', help='StyleGAN2 network pickle filename')
    parser.add_argument('--fps', default=30, help='fps of output video')
    parser.add_argument('--fpe', default=60, help='Number of frames per endpoint')
    parser.add_argument('--shrink', default=1, help='How much to shrink the output image.\
        Set to 1 to get image of size 1024x1024, 2: 512, 4: 256 and so on.')
    parser.add_argument('--result-dir', 
                            help='Root directory for run results (default: %(default)s)', 
                            default='results', metavar='DIR')
    args = parser.parse_args()
    kwargs = vars(args)

    sc = dnnlib.SubmitConfig()
    sc.num_gpus = 1
    sc.submit_target = dnnlib.SubmitTarget.LOCAL
    sc.local.do_not_copy_source_files = True
    sc.run_dir_root = kwargs.pop('result_dir')
    sc.run_desc = 'transition-' + args.src_dir.split('/')[-2] 

    dnnlib.submit_run(sc, 'transition.generate_transition', **kwargs)


if __name__ == "__main__":
    '''
    example:
    > python transition.py src_dir/ --network-pkl=models/stylegan2-ffhq-config-f.pkl --fps=30 --fpe=60 --shrink=4
    '''
    main()
