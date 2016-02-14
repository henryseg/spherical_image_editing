#### Python code for performing Mobius transformations to equirectangular images, by Henry Segerman, Jan 2016

#### This code is provided as is, with no warranty or expectation of correctness. Do not use for mission critical
#### purposes without thoroughly checking it yourself. 

#### This code uses the Python Imaging Library (PIL), currently supported as "Pillow", available from https://python-pillow.github.io

#### For more details, see the blogpost at http://elevr.com/spherical-video-editing-effects-with-mobius-transformations/
#### and the tech demo spherical video at https://www.youtube.com/watch?v=oVwmF_vrZh0

import os
from math import *
from vectors_and_matrices import vector, dot, cross, matrix2_inv, matrix_mult, matrix_mult_vector
from PIL import Image
import cmath

def angles_from_pixel_coords(point, x_size = 1000):
  """map from pixel coords to (0, 2*pi) x (-pi/2, pi/2) rectangle"""
  y_size = x_size/2  #assume equirectangular format
  return (point[0] * 2*pi/float(x_size), point[1] * pi/float(y_size-1) - 0.5*pi)

def pixel_coords_from_angles(point, x_size = 1000):
  """map from (0, 2*pi) x (-pi/2, pi/2) rectangle to pixel coords"""
  y_size = x_size/2  #assume equirectangular format
  return (point[0] * float(x_size)/(2*pi), (point[1] + 0.5*pi)* float(y_size-1)/pi)

def angles_from_sphere(point):
  """map from sphere in R^3 to (0, 2*pi) x (-pi/2, pi/2) rectangle (i.e. perform equirectangular projection)"""
  x,y,z = point
  longitude = atan2(y,x)
  if longitude < 0.0:
    longitude = longitude + 2*pi
  r = sqrt(x*x+y*y)
  latitude = atan2(z,r)
  return (longitude, latitude)

def sphere_from_angles(point): 
  """map from (0, 2*pi) x (-pi/2, pi/2) rectangle to sphere in R^3 (i.e. perform inverse of equirectangular projection)"""
  x,y = point
  horiz_radius = cos(y)
  return vector([horiz_radius*cos(x), horiz_radius*sin(x), sin(y)])
  
def sphere_from_pixel_coords(point, x_size = 1000):
  """map from pixel coords to sphere in R^3"""
  return sphere_from_angles(angles_from_pixel_coords(point, x_size = x_size))

def CP1_from_sphere(point):
  """map from sphere in R^3 to CP^1"""
  x,y,z = point
  if z < 0:
    return (complex(x,y), complex(1-z))
  else:
    return (complex(1+z), complex(x,-y))

def sphere_from_CP1(point):
  """map from CP^1 to sphere in R^3"""
  z1,z2 = point
  if abs(z2) > abs(z1):
    z = z1/z2
    x,y = z.real, z.imag
    denom = 1 + x*x + y*y
    return [2*x/denom, 2*y/denom, (denom - 2.0)/denom]
  else:
    z = (z2/z1).conjugate()
    x,y = z.real, z.imag
    denom = 1 + x*x + y*y
    return [2*x/denom, 2*y/denom, (2.0 - denom)/denom]

def clamp(pt, x_size):
  """clamp to size of input, including wrapping around in the x direction""" 
  y_size = x_size/2       # assume equirectangular format
  pt[0] = pt[0] % x_size  # wrap around in the x direction
  if pt[1] < 0:
    pt[1] = 0
  elif pt[1] > y_size - 1:
    pt[1] = y_size - 1
  return pt

def get_pixel_colour(pt, s_im, x_size = 1000):
  """given pt in integers, get pixel colour on the source image as a vector in the colour cube"""
  pt = clamp(pt, x_size)
  return vector(s_im[pt[0], pt[1]])

def get_interpolated_pixel_colour(pt, s_im, x_size = 1000):
  """given pt in floats, linear interpolate pixel values nearby to get a good colour"""
  ### for proper production software, more than just the four pixels nearest to the input point coordinates should be used in many cases
  x,y = int(floor(pt[0])), int(floor(pt[1]))  #integer part of input coordinates
  f,g = pt[0]-x, pt[1]-y                      #fractional part of input coordinates
  out_colour = (1-f)*( (1-g)*get_pixel_colour([x,y], s_im, x_size = x_size) + g*get_pixel_colour([x,y+1], s_im, x_size = x_size) ) \
          +  f*( (1-g)*get_pixel_colour([x+1,y], s_im, x_size = x_size) + g*get_pixel_colour([x+1,y+1], s_im, x_size = x_size) )
  out_colour = [int(round(coord)) for coord in out_colour]
  return tuple(out_colour)

######## Functions generating SL(2,C) matrices 

# Note that we only care about the matrices projectively. I.e. [[a,b],[c,d]] acts in exactly the same way on points of CP^1 as
# [[-a,-b],[-c,-d]], so we can also think of these matrices as being in PSL(2,C).

def inf_zero_one_to_triple(p,q,r):
  """returns SL(2,C) matrix that sends the three points infinity, zero, one to given input points p,q,r"""
  ### infinity = [1,0], zero = [0,1], one = [1,1] in CP^1
  p1,p2=p
  q1,q2=q
  r1,r2=r
  M = [[p1,q1],[p2,q2]]
  Minv = matrix2_inv(M)
  [mu,lam] = matrix_mult_vector(matrix2_inv([[p1,q1],[p2,q2]]), [r1,r2])
  return [[mu*p1, lam*q1],[mu*p2, lam*q2]]

def two_triples_to_SL(a1,b1,c1,a2,b2,c2):
  """returns SL(2,C) matrix that sends the three CP^1 points a1,b1,c1 to a2,b2,c2"""
  return matrix_mult( inf_zero_one_to_triple(a2,b2,c2), matrix2_inv(inf_zero_one_to_triple(a1,b1,c1) ) ) 

def three_points_to_three_points_pixel_coords(p1, q1, r1, p2, q2, r2, x_size = 1000):
  """returns SL(2,C) matrix that sends the three pixel coordinate points a1,b1,c1 to a2,b2,c2"""
  p1,q1,r1,p2,q2,r2 = [CP1_from_sphere(sphere_from_pixel_coords(point, x_size = x_size)) for point in [p1,q1,r1,p2,q2,r2]]
  return two_triples_to_SL(p1,q1,r1,p2,q2,r2)

def get_vector_perp_to_p_and_q(p, q):
  """p and q are distinct points on sphere, return a unit vector perpendicular to both"""
  if abs(dot(p,q) + 1) < 0.0001: ### deal with the awkward special case when p and q are antipodal on the sphere
      if abs(dot(p, vector([1,0,0]))) > 0.9999: #p is parallel to (1,0,0)
        return vector([0,1,0])
      else:
        return cross(p, vector([1,0,0])).normalised() 
  else:
    return cross(p, q).normalised()

def rotate_sphere_points_p_to_q(p, q):
  """p and q are points on the sphere, return SL(2,C) matrix rotating image of p to image of q on CP^1"""
  p, q = vector(p), vector(q)
  CP1p, CP1q = CP1_from_sphere(p), CP1_from_sphere(q)
  if abs(dot(p,q) - 1) < 0.0001:
    return [[1,0],[0,1]]
  else:
    r = get_vector_perp_to_p_and_q(p, q)
    CP1r, CP1mr = CP1_from_sphere(r), CP1_from_sphere(-r)
    return two_triples_to_SL(CP1p, CP1r, CP1mr, CP1q, CP1r, CP1mr) 

def rotate_pixel_coords_p_to_q(p, q, x_size = 1000):
  """p and q are pixel coordinate points, return SL(2,C) matrix rotating image of p to image of q on CP^1"""
  p = sphere_from_pixel_coords(p, x_size = x_size)
  q = sphere_from_pixel_coords(q, x_size = x_size)
  return rotate_sphere_points_p_to_q(p,q)

def rotate_around_axis_sphere_points_p_q(p,q,theta):
  """p and q are points on sphere, return SL(2,C) matrix rotating by angle theta around the axis from p to q"""
  p, q = vector(p), vector(q)
  CP1p, CP1q = CP1_from_sphere(p), CP1_from_sphere(q)
  assert dot(p,q) < 0.9999, "axis points should not be in the same place!"
  r = get_vector_perp_to_p_and_q(p, q)
  CP1r = CP1_from_sphere(r)
  M_standardise = two_triples_to_SL(CP1p, CP1q, CP1r, [0,1], [1,0], [1,1])
  M_theta = [[complex(cos(theta),sin(theta)),0],[0,1]] #rotate on axis through 0, infty by theta
  return matrix_mult( matrix_mult(matrix2_inv(M_standardise), M_theta), M_standardise )

def rotate_around_axis_pixel_coords_p_q(p,q,theta, x_size = 1000):
  """p and q are pixel coordinate points, return SL(2,C) matrix rotating by angle theta around the axis from p to q"""
  p = sphere_from_pixel_coords(p, x_size = x_size)
  q = sphere_from_pixel_coords(q, x_size = x_size)
  return rotate_around_axis_sphere_points_p_q(p,q,theta)

def rotate_around_axis_pixel_coord_p(p,theta, x_size = 1000):
  """p is a pixel coordinate point, return SL(2,C) matrix rotating by angle theta around the axis from p to its antipode"""
  p = sphere_from_pixel_coords(p, x_size = x_size)
  minus_p = -p
  return rotate_around_axis_sphere_points_p_q(p,minus_p,theta)
     
def zoom_in_on_pixel_coords(p, zoom_factor, x_size = 1000):
  """p is pixel coordinate point, return SL(2,C) matrix zooming in on p by a factor of scale"""
  # Note that the zoom factor is only accurate at the point p itself. As we move away from p, we zoom less and less.
  # We zoom with the inverse zoom_factor towards/away from the antipodal point to p.
  M_rot = rotate_pixel_coords_p_to_q( p, (0,0), x_size = x_size)
  M_scl = [[zoom_factor,0],[0,1]] ### zoom in on zero in CP^1
  return matrix_mult( matrix_mult(matrix2_inv(M_rot), M_scl), M_rot )

def zoom_along_axis_sphere_points_p_q(p, q, zoom_factor):
  """p and q are points on sphere, return SL(2,C) matrix zooming along axis from p to q"""
  p, q = vector(p), vector(q)
  CP1p, CP1q = CP1_from_sphere(p), CP1_from_sphere(q)
  assert dot(p,q) < 0.9999   #points should not be in the same place
  r = get_vector_perp_to_p_and_q(p, q)
  CP1r = CP1_from_sphere(r)
  M_standardise = two_triples_to_SL(CP1p, CP1q, CP1r, [0,1], [1,0], [1,1])
  M_theta = [[zoom_factor,0],[0,1]] 
  return matrix_mult( matrix_mult(matrix2_inv(M_standardise), M_theta), M_standardise )

def zoom_along_axis_pixel_coords_p_q(p, q, zoom_factor, x_size = 1000):
  """p and q are pixel coordinate points, return SL(2,C) matrix zooming along axis from p to q by zoom_factor"""
  # This function does the same thing as zoom_in_on_pixel_coords, but with the 
  # two given points instead of a single point and its antipodal point
  p = sphere_from_pixel_coords(p, x_size = x_size)
  q = sphere_from_pixel_coords(q, x_size = x_size)
  return zoom_along_axis_sphere_points_p_q(p,q,zoom_factor)

def translate_along_axis_pixel_coords(p, q, r1, r2, x_size = 1000):
  """Return SL(2,C) matrix translating/rotating on the axis from p to q, taking r1 to r2"""
  return three_points_to_three_points_pixel_coords(p,q,r1,p,q,r2, x_size = x_size)

##### Apply functions to images

def apply_SL2C_elt_to_image(M, source_image_filename, out_x_size = None, save_filename = "sphere_transforms_test.png"):
  """apply an element of SL(2,C) (i.e. a matrix) to an equirectangular image file"""
  Minv = matrix2_inv(M)  # to push an image forwards by M, we pull the pixel coordinates of the output backwards 
  source_image = Image.open(source_image_filename)
  s_im = source_image.load()  # faster pixel by pixel access 
  in_x_size, in_y_size = source_image.size
  if out_x_size == None:
    out_x_size, out_y_size = source_image.size
  else:
    out_y_size = out_x_size/2
  out_image = Image.new("RGB", (out_x_size, out_y_size))
  o_im = out_image.load()  # faster pixel by pixel access 

  for i in range(out_x_size): 
    for j in range(out_y_size):
      pt = (i,j)
      pt = angles_from_pixel_coords(pt, x_size = out_x_size)
      pt = sphere_from_angles(pt)
      pt = CP1_from_sphere(pt)
      pt = matrix_mult_vector(Minv, pt)
      pt = sphere_from_CP1(pt)
      pt = angles_from_sphere(pt)
      pt = pixel_coords_from_angles(pt, x_size = in_x_size)
      o_im[i,j] = get_interpolated_pixel_colour(pt, s_im, x_size = in_x_size)
  out_image.save(save_filename)

def droste_effect(zoom_center_pixel_coords, zoom_factor, zoom_cutoff, source_image_filename, out_x_size = None, twist = False, zoom_loop_value = 0.0, save_filename = "sphere_transforms_test.png"):
  """produces a zooming Droste effect image from an equirectangular source image"""
  # The source image should be a composite of the original image together with a zoomed version, 
  # fit inside a picture frame or similar in the original image
  source_image = Image.open(source_image_filename)
  s_im = source_image.load()
  in_x_size, in_y_size = source_image.size
  
  M_rot = rotate_pixel_coords_p_to_q(zoom_center_pixel_coords, (0,0), x_size = in_x_size)
  M_rot_inv = matrix2_inv(M_rot)
  if out_x_size == None:
    out_x_size, out_y_size = source_image.size
  else:
    out_y_size = out_x_size/2
  out_image = Image.new("RGB", (out_x_size, out_y_size))
  o_im = out_image.load()

  droste_factor = ( cmath.log(zoom_factor) + complex(0, 2*pi) ) / complex(0, 2*pi)  # used if twisting 

  for i in range(out_x_size): 
    for j in range(out_y_size):
      pt = (i,j)
      pt = angles_from_pixel_coords(pt, x_size = out_x_size)
      pt = sphere_from_angles(pt)
      pt = CP1_from_sphere(pt)
      pt = matrix_mult_vector(M_rot, pt)

      # if ever you don't know how to do some operation in complex projective coordinates, it's almost certainly 
      # safe to just switch back to ordinary complex numbers by pt = pt[0]/pt[1]. The only danger is if pt[1] == 0, 
      # or is near enough to cause floating point errors. In this application, you are probably fine unless you 
      # make some very specific choices of where to zoom to etc. 
      pt = pt[0]/pt[1]  
      pt = cmath.log(pt)
      if twist:  # otherwise straight zoom
        pt = droste_factor * pt  

      # zoom_loop_value is between 0 and 1, vary this from 0.0 to 1.0 to animate frames zooming into the droste image
      pt = complex(pt.real + log(zoom_factor) * zoom_loop_value, pt.imag) 
      
      # zoom_cutoff alters the slice of the input image that we use, so that it covers mostly the original image, together with 
      # some of the zoomed image that was composited with the original. The slice needs to cover the seam between the two
      # (i.e. the picture frame you are using, but should cover as little as possible of the zoomed version of the image.
      pt = complex((pt.real + zoom_cutoff) % log(zoom_factor) - zoom_cutoff, pt.imag) 
      pt = cmath.exp(pt)
      pt = [pt, 1] #back to projective coordinates
      pt = matrix_mult_vector(M_rot_inv, pt)
      pt = sphere_from_CP1(pt)
      pt = angles_from_sphere(pt)
      pt = pixel_coords_from_angles(pt, x_size = in_x_size)
      o_im[i,j] = get_interpolated_pixel_colour(pt, s_im, in_x_size)

  out_image.save(save_filename)

def batch_transform_frames(in_dirname, out_dirname, M_func, frame_num_range = None, func_range = None, out_x_size = None):
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
  for filename in filenames: 
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
  print('done')

######## Useful functions to use when making transformation functions to put into batch_transform_frames

def up_down(t):
  """ graph: /\ """
  if t < 0.5:
    return 2*t
  else:
    return 2-2*t

def up_plateau_down(t):
  """ graph: /"\ """
  if t < 0.33333:
    return 3*t
  elif t < 0.66667:
    return 1.0
  else:
    return 3 - 3*t

######## Test 

if __name__ == "__main__":

  ###### Apply transformations to single frames

  M = zoom_in_on_pixel_coords((360,179.5), 2, x_size = 720) 
  apply_SL2C_elt_to_image( M, 'equirectangular_test_image.png', save_filename = 'scaled_test_image.png' )

  # M = rotate_around_axis_pixel_coord_p((360,179.5), pi/8, x_size = 720)
  # apply_SL2C_elt_to_image( M, 'equirectangular_test_image.png', save_filename = 'rotated_test_image.png' )

  ###### Apply transformations to a directory of frames

  ### Assume you have a directory called "foo_in" that has 1920x960 png frame images called something like 
  ### "frame_000.png", "frame_001.png", ... "frame_099.png", and a directory "foo_out" you want to save 
  ### images to. Then, first define the transformation function (as a function of time):

  # def my_zoom(t):
    # return zoom_in_on_pixel_coords((960,480), 1 + up_plateau_down(t), x_size = 1920) 

  ### Then run the batch function

  # batch_transform_frames("foo_in", "foo_out", my_zoom, frame_num_range = (0, 100), func_range = (0, 100), out_x_size = 1920)

  ###### Make droste animations

  # num_frames = 4
  # for i in range(num_frames):
  #   zoom_loop_value = float(i)/float(num_frames)
  #   droste_effect((2650,1300), 7.3, 1.0, '(elevr+h)x2_one_zoom_7.3.png', out_x_size = 1920, twist = False, zoom_loop_value = zoom_loop_value, save_filename = "droste_straight_anim_frames/droste_anim_straight_" + str(i).zfill(3) + ".png")
  #   droste_effect((2650,1300), 7.3, 1.0, '(elevr+h)x2_one_zoom_7.3.png', out_x_size = 1920, twist = True, zoom_loop_value = zoom_loop_value, save_filename = "droste_twist_anim_frames/droste_anim_twist_" + str(i).zfill(3) + ".png")




