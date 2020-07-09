import PIL.Image as Image
import os

filenames = [
    'c1.png', 'c2.png', 'c3.png', 'c4.png', 'c5.png', 'c6.png', 'c7.png',
    'c8.png', 'c9.png', 'c10.png', 'c11.png', 'c12.png', 'c13.png', 'c14.png',
    'f1.png', 'f2.png', 'f3.png', 'f4.png', 'f5.png', 'f6.png', 'f7.png',
    'f8.png', 'f9.png', 'f10.png', 'f11.png', 'f12.png', 'f13.png', 'f14.png',
    'f15.png', 'f16.png', 'f17.png', 'f18.png', 'f19.png', 'f20.png', 'f21.png'
]

# Create the frames
frames = []
for file_name in filenames:
  if os.path.exists(file_name):
    new_frame = Image.open(file_name)
    new_frame = new_frame.resize((240, 240), Image.LANCZOS)
    frames.append(new_frame)

# Save into a GIF file that loops forever
if frames:
  frames[0].save(
      'png_to_gif.gif',
      format='GIF',
      append_images=frames[1:],
      save_all=True,
      duration=150,
      loop=0)
