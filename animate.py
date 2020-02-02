import pathlib
import glob
import numpy as np
import sys
import moviepy.editor
import argparse

import dnnlib
import dnnlib.tflib as tflib
import pretrained_networks

'''
linear_interpolate
Modified from: https://github.com/ShenYujun/InterFaceGAN/blob/b707e942187f464251f855c92f7009b8cf13bf03/utils/manipulator.py
'''
def linear_interpolate(latent_code,
                       boundary,
                       start_distance=-3.0,
                       end_distance=3.0,
                       steps=60):
  linspace = np.linspace(start_distance, end_distance, steps)
  linspace = linspace.reshape(-1, 1, 1).astype(np.float32)
  linspace = linspace * boundary
  return latent_code + linspace


def render_video(set_images, filename, fps=30, codec='libx264', bitrate='5M'):

    def render_frame(t):
        frame = int(np.clip(np.ceil(t * fps), 1, num_frames))
        return set_images[frame]

    num_frames = len(set_images)
    duration = num_frames / fps
    print(f"num_frames: {num_frames}, duration: {duration}")
    video_clip = moviepy.editor.VideoClip(render_frame, duration=duration)
    video_clip.write_videofile(filename, fps=fps, codec=codec, bitrate=bitrate)


def animate(network_pkl, in_file, mode, dir_file, start, stop, steps, reverse, repeat):
    tflib.init_tf()
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)

    Gs_syn_kwargs = dnnlib.EasyDict()
    Gs_syn_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_syn_kwargs.randomize_noise = False
    Gs_syn_kwargs.minibatch_size = 8

    # TODO: can this be refactored?
    if mode == "blink":
        dir_file = 'stylegan2directions/eyes_open.npy'
        start = 30 # start from eye open position
        stop = -40 # close eyes
        reverse = True # and go back to open position. This completes a blink.
        repeat = 2 # blink twice
        steps = 5 # blinks are usually fast.
    elif mode == "smile":
        dir_file = 'stylegan2directions/smile.npy'
        start = -1 # start from almost neutral position
        stop = 15 # and smile
        repeat = 0 # smile is not repeated.
        reverse = False # or reversed, looks odd to unsmile.
        steps = 30 # smiles are usually slow.
    elif mode == "yes":
        # see comments in the if-else statement.
        dir_file = 'stylegan2directions/pitch.npy'
        start = -15
        stop = 15
        repeat = 0
        reverse = True
        steps = 8
    elif mode == "no":
        # see comments in the if-else statement.
        dir_file = 'stylegan2directions/yaw.npy'
        start = -15
        stop = 15
        repeat = 0
        reverse = True
        steps = 8

    x = np.load(in_file)
    direction = np.load(dir_file)

    base_name = in_file.replace('.npy', '').replace('/', '_')
    dir_base_name =  dir_file.replace('.npy', '').split('/')[-1]

    if mode == 'yes' or mode == 'no':
        # to start at neutral position and end at neutral. Else looks a bit odd.
        # go from neutral to head down or neutral to right side.
        latent_batch_1 = linear_interpolate(x.reshape((1, 18, 512)), direction.reshape(18, 512), 0, start, steps=steps)
        # head down to up or right to left
        latent_batch_2 = linear_interpolate(x.reshape((1, 18, 512)), direction.reshape(18, 512), start, stop, steps=steps)
        # later just reverse this.
        latent_batch = np.concatenate((latent_batch_1, latent_batch_2), axis=0)
    else:
        latent_batch = linear_interpolate(x.reshape((1, 18, 512)), direction.reshape(18, 512), start, stop, steps=steps)

    set_images = Gs.components.synthesis.run(latent_batch, **Gs_syn_kwargs)
    first_image = np.repeat(np.expand_dims(set_images[0], axis=0), 20, axis=0)

    if reverse:
        rev_images = np.flipud(set_images)
        set_images = np.concatenate((set_images, rev_images), axis=0)

    if repeat > 0:
        set_images = np.tile(set_images, (repeat, 1, 1, 1))

    last_image = np.repeat(np.expand_dims(set_images[-1], axis=0), 20, axis=0)
    set_images = np.concatenate((first_image, set_images, last_image), axis=0)

    print("############################")
    vid_name = dnnlib.make_run_dir_path('%s_%s.mp4' %(dir_base_name, base_name))
    print("\nGenerating video %s..." %vid_name)
    render_video(set_images, vid_name)


################################################################################
_examples = """
# eyes
python animate.py --in-file=face_datasets/jdepp/4_01.npy
                    --dir-file=stylegan2directions/eyes_open.npy
                    --start=30 --stop=-40 --repeat 2 --reverse --steps=5

# smile
python animate.py --in-file=face_datasets/jdepp/4_01.npy
                    --dir-file=stylegan2directions/smile.npy
                    --start=-1 --stop=15 --repeat 0 --steps=30

# pitch
python animate.py --in-file=face_datasets/jdepp/4_01.npy
                    --dir-file=stylegan2directions/pitch.npy
                    --start=15 --stop=-15 --repeat 2 --steps=8 --reverse

# yaw
python animate.py --in-file=face_datasets/jdepp/4_01.npy
                    --dir-file=stylegan2directions/yaw.npy
                    --start=15 --stop=-15 --repeat 0 --steps=8

"""


def main():
    parser = argparse.ArgumentParser(
        description='StyleGAN2 style mixer - mixes styles of row and column images.',
        epilog=_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--network',
                            default='models/stylegan2-ffhq-config-f.pkl',
                            help='Network pickle filename', dest='network_pkl')
    parser.add_argument('--in-file',
                            help='Path to projected npy file', required=True)
    parser.add_argument('--mode',
                            help='Predifined for smile, blink, yes, no. If given rest of the options are ignored',
                            default='custom')
    parser.add_argument('--dir-file',
                            help='Path to direction npy file')
    parser.add_argument('--steps',
                            help='Number of steps from start to stop. Higher the steps, smoother the interpolation',
                            default=30)
    # TODO: instead of start and stop, accept an array of intermediate values.
    parser.add_argument('--start',
                            help='Interpolation starts from start*direction',
                            type=int,
                            default=-15)
    parser.add_argument('--stop',
                            help='Interpolation ends at stop*direction',
                            type=int,
                            default=15)
    parser.add_argument('--reverse',
                            help='Should the interpolation be reversed?',
                            action='store_true')
    parser.add_argument('--repeat',
                            help='How many times should the interpolation be repeated?',
                            default=2,
                            type=int)
    parser.add_argument('--result-dir',
                            help='Root directory for run results (default: %(default)s)',
                            default='results', metavar='DIR')

    args = parser.parse_args()
    kwargs = vars(args)

    if args.dir_file:
        run_desc = args.in_file.split('/')[-2] + '_' + args.dir_file.split('/')[-1].replace('.npy', '')
    else:
        run_desc = args.in_file.split('/')[-2] + '_' + args.mode

    sc = dnnlib.SubmitConfig()
    sc.num_gpus = 1
    sc.submit_target = dnnlib.SubmitTarget.LOCAL
    sc.local.do_not_copy_source_files = True
    sc.run_dir_root = kwargs.pop('result_dir')
    sc.run_desc = run_desc

    dnnlib.submit_run(sc, 'animate.animate', **kwargs)


if __name__ == "__main__":
    main()
