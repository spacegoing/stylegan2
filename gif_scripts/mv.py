import os
import shutil
from PIL import Image

r_path = ['00002-makeup_kol_rows_kol_cols', '00002-makeup_star_rows_star_cols']
c_path = ['00001-makeup_kol_cols_kol_rows', '00001-makeup_star_cols_star_rows']
path_list = r_path + c_path


def category_pngs(path_list):

  def mv_by_ft(path, png_files):
    for i in range(1, 5):
      target_name = 'ft%d' % i
      ft_files = [f for f in png_files if target_name in f]

      if not os.path.exists(path + '/' + target_name):
        os.mkdir(path + '/' + target_name)
      for f in ft_files:
        shutil.move(path + '/' + f, path + '/' + target_name + '/' + f)

  for path in path_list:
    png_files = [f for f in os.listdir(path) if f.endswith('png')]
    mv_by_ft(path, png_files)


category_pngs(path_list)

path = r_path
row_first = True


## Generate GIFs
def gen_gif(r_path, c_path):

  def generate_gif_by_png_name(target_path, target_name, row_first=True):
    # Create the frames
    frames = []
    for j in range(1, 22):
      if row_first:
        file_name = target_path + '/' + 'f%d-%s.png' % (j, target_name)
      else:
        file_name = target_path + '/' + '%s-f%d.png' % (target_name, j)
      if os.path.exists(file_name):
        new_frame = Image.open(file_name)
        new_frame = new_frame.resize((240, 240), Image.LANCZOS)
        frames.append(new_frame)

    # Save into a GIF file that loops forever
    if frames:
      frames[0].save(
          target_path + '/' + 'png_to_gif.gif',
          format='GIF',
          append_images=frames[1:],
          save_all=True,
          duration=150,
          loop=0)

  for path in r_path:
    row_first = True
    for i in range(1, 5):
      target_name = 'ft%d' % i
      target_path = path + '/' + target_name
      generate_gif_by_png_name(target_path, target_name, row_first)

  for path in c_path:
    row_first = False
    for i in range(1, 5):
      target_name = 'ft%d' % i
      target_path = path + '/' + target_name
      generate_gif_by_png_name(target_path, target_name, row_first)
