import os
from multiprocessing import Pool
from functools import partial

from sphere_transforms import zoom_in_on_pixel_coords, up_plateau_down, apply_SL2C_elt_to_image

def process_single_image(in_dirname, out_dirname, M_func, start, end, fstart, fend, out_x_size, filename):
  fullname = in_dirname + '/' + filename
  if not os.path.exists(fullname):
    print('file not found!')
  else:
    if filename[-4:] == '.png':
      frame_num = int(filename[-7:-4])
      if start <= frame_num < end:
        print('frame ', int(filename[-7:-4]))
        t = float(frame_num - fstart)/float((fend-1) - fstart)  ## 0.0 <= t <= 1.0
        M = M_func(t)
        apply_SL2C_elt_to_image( M, fullname, out_x_size = out_x_size, save_filename = out_dirname + '/' + filename )


def batch_transform_frames(in_dirname, out_dirname, M_func, frame_num_range = None, func_range = None, out_x_size = None, parallel = 4):
  """transform a directory of numbered frame images according to a given function, that outputs a matrix as a function of time"""
  ### assumes num_frames is a three digit number
  ### M_func should be a function that takes in a float between 0.0 and 1.0 and returns an SL(2,C) matrix
  ### frame_num_range is a pair, telling us which frames to process
  ### func_range is a pair, the first telling us the frame corresponding to t = 0.0, and the second to t = 1.0
  ### these two ranges could be different if you are running multiple processes simultaneously, 
  ### dealing with different frame_num_ranges, but running the same function with the same func_range
  filenames = os.listdir(in_dirname)
  start, end = frame_num_range
  fstart, fend = func_range
  curried = partial(process_single_image, in_dirname, out_dirname, M_func, start, end, fstart, fend, out_x_size)
  try:
  	pool = Pool(processes = parallel)
  	pool.map(curried, filenames)
  finally:
    pool.close()
    pool.join()

  print('done')

######## Test 

if __name__ == "__main__":

  ###### Apply transformations to a directory of frames

  ### Assume you have a directory called "foo_in" that has 1920x960 png frame images called something like 
  ### "frame_000.png", "frame_001.png", ... "frame_099.png", and a directory "foo_out" you want to save 
  ### images to. Then, first define the transformation function (as a function of time):

  def my_zoom(t):
    return zoom_in_on_pixel_coords((960,480), 1 + up_plateau_down(t), x_size = 1920)

  ### Then run the batch function

  ### Then run the batch function

  batch_transform_frames("foo_in", "foo_out", my_zoom, frame_num_range = (0, 100), func_range = (0, 100), out_x_size = 1920, parallel = 8)
