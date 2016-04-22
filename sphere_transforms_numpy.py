
import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import map_coordinates
from numpy import pi
from PIL import Image

def angles_from_pixel_coords(pts, size):
    """map from pixel coordinates to (0, 2*pi) x (-pi/2, pi/2) rectangle
    Parameters
    ----------
    pts : array_like (2,...)
        pts[0,...], and pts[1,...] are row, column image coordinates in the ranges (0,size[0]) and (0,size[1]), respectively
    size : tuple
        rectangular image size (rows, columns)
    
    Returns
    -------
    out : ndarray, same shape as pts
        pts transformed into (0,2pi) x (-pi/2,pi/2) range (and transposed)
    """
    pts = np.asarray(pts,dtype=np.float64) #turn into an ndarray if necessary
    ys,xs = size #row,col indexing
    out = np.empty_like(pts)
    out[0] = (pts[1]+0.5)*2*pi/float(xs)     #+0.5 shift for pixel coords lining up properly
    out[1] = pts[0]*pi/float(ys-1) - 0.5*pi
    return out

def pixel_coords_from_angles(pts, size):
    """map from (0, 2*pi) x (-pi/2, pi/2) rectangle to pixel coordinates
    
    Parameters
    ----------
    pts : array_like (2,...)
        pts[0,...], and pts[1,...] are x and y coordinates in the ranges (0,size[0]) and (0,size[1]), respectively
    size : tuple
        rectangle image size (rows, columns)
    
    Returns
    -------
    out : ndarray, same shape as pts
        pts transformed from (0,2*pi) x (-pi/2,pi/2) range into (0,width) x (0,height) range (and transposed)
    """
    pts = np.asarray(pts) #turn into an ndarray if necessary
    ys,xs = size #row, col indexing
    out = np.empty_like(pts)
    out[1] = pts[0]*float(xs)/(2*pi) - 0.5  #-0.5 shift for pixel coords lining up properly
    out[0] = (pts[1] + 0.5*pi)*float(ys-1)/pi
    return out

def angles_from_sphere(pts):
    """equirectangular projection, ie. map from sphere in R^3 to (0, 2*pi) x (-pi/2, pi/2)
    Parameters
    ----------
    pts : array_like (3,...)
        pts[0,...], pts[1,...], and pts[2,...] are x,y, and z coordinates
    Returns
    -------
    out : ndarray (2,...)
        pts transformed from x,y,z to longitude,latitude
    """
    pts = np.asarray(pts) #turn into an ndarray if necessary
    x,y,z = pts[0], pts[1], pts[2]
    out = np.empty((2,) + pts.shape[1:])
    #longitude:
    out[0] = np.arctan2(y,x)
    out[0] %= 2*pi #wrap negative values around the circle to positive
    #latitude:
    r = np.hypot(x,y)
    out[1] = np.arctan2(z,r)
    return out

def sphere_from_angles(pts):
    """inverse equirectangular projections, ie. map from (0,2*pi) x (-pi/2,pi/2) rectangle to sphere in R^3
    Parameters
    ----------
    pts : array_like (2,...)
        pts[0,...], pts[1,...] are longitude, latitude
    Returns
    -------
    out : ndarray (3,...)
        pts transformed from longitude,latitude to x,y,z
    """
    pts = np.asarray(pts)
    out = np.empty((3,) + pts.shape[1:])
    lon,lat = pts[0],pts[1]
    horiz_radius = np.cos(lat)
    out[0] = horiz_radius*np.cos(lon) #x
    out[1] = horiz_radius*np.sin(lon) #y
    out[2] = np.sin(lat) #z
    return out

def sphere_from_pixel_coords(pts, size):
    """map from pixel coordinates to sphere in R^3
    Parameters
    ----------
    pts : array_like (2,...)
        pts[0,...], and pts[1,...] are u and v coordinates in the ranges (0,size[0]) and (0,size[1]), respectively
    size : tuple
        rectangular image size (width, height)
    Returns
    -------
    out : ndarray (3,...)
        coordinates of pts wrapped around a sphere u,v -> x,y,z
    """
    return sphere_from_angles(angles_from_pixel_coords(pts, size))

def CP1_from_sphere(pts):
    """map from sphere in R^3 to CP^1"""
    pts = np.asarray(pts)
    out = np.empty((2,)+pts.shape[1:],dtype=np.complex128)
    x,y,z = pts[0],pts[1],pts[2]
    mask = z<0
    out[0] = np.where(mask, x + 1j*y, 1 + z)
    out[1] = np.where(mask, 1 - z, x - 1j*y)
    return out

def sphere_from_CP1(pts):
    """map from CP^1 to sphere in R^3"""
    pts = np.asarray(pts)
    out = np.empty((3,) + pts.shape[1:])
    z1,z2 = pts[0],pts[1]
    mask = abs(z2) > abs(z1)
    z = np.where(mask, z1/z2, np.conj(z2/z1))
    x,y = np.real(z), np.imag(z)
    denom = 1 + x**2 + y**2
    out[0] = 2*x/denom
    out[1] = 2*y/denom
    out[2] = (denom-2)/denom*(2*mask-1)  #negate where mask is false
    return out

def clamp(pts, size):
    """clamp to the size of the input, including wrapping around in the x direction"""
    ys,xs = size
    pts = np.asarray(pts)
    out = np.empty_like(pts)
    out[0] = np.clip(pts[0],0,ys-1)
    out[1] = pts[1] % xs
    return out

def get_pixel_color(pts, s_im, size):
    """given pts in integers, get pixel colour on the source image as a vector in the colour cube"""
    pts = clamp(pts,size)
    s_im = np.asarray(s_im)
    return s_im[pts[0], pts[1]]

def get_interpolated_pixel_color(pts, s_im, size):
    """given pts in floats, linear interpolate pixel values nearby to get a good colour"""
    pts = clamp(pts, size)

    s_im = np.atleast_3d(s_im)
    ys,xs = size
    ycoords, xcoords = np.arange(ys), np.arange(xs)
    out = np.empty(pts.shape[1:] + (s_im.shape[-1],),dtype=s_im.dtype)
    for i in range(s_im.shape[-1]): #loop over color channels
        map_coordinates(s_im[...,i],pts,out[...,i],mode='nearest')
    return out

def get_interpolated_pixel_color_rbspline(pts, s_im, size):
    """given pts in floats, linear interpolate pixel values nearby to get a good colour"""
    pts = clamp(pts, size)

    s_im = np.atleast_3d(s_im)
    ys,xs = size
    ycoords, xcoords = np.arange(ys), np.arange(xs)
    out = np.empty(pts.shape[1:] + (s_im.shape[-1],),dtype=s_im.dtype)
    
    pts_vec = pts.reshape((2,-1))
    out_vec = out.reshape((-1,s_im.shape[-1])) #flatten for easier vectorization
    for i in range(s_im.shape[-1]): #loop over color channels
        rbspline = RectBivariateSpline(ycoords, xcoords, s_im[...,i])
        out_vec[:,i] = rbspline.ev(pts_vec[0],pts_vec[1])
    return out


### Functions generating SL(2,C) matrices ###
# Do not need to be vectorized #

def inf_zero_one_to_triple(p,q,r):
    """return SL(2,C) matrix that sends the three points infinity, zero, one to given input points p,q,r"""
    M = np.vstack((p,q)).T
    mu,lam = np.linalg.lstsq(M,r)[0]
    return np.vstack((mu*p,lam*q)).T

def two_triples_to_SL(a1,b1,c1,a2,b2,c2):
    """returns SL(2,C) matrix that sends the three CP^1 points a1,b1,c1 to a2,b2,c2"""
    M1 = inf_zero_one_to_triple(a1,b1,c1)
    M2 = inf_zero_one_to_triple(a2,b2,c2)
    # solve M1.T @ X.T = M2.T  => X = M2 @ inv(M1)
    return np.linalg.lstsq(M1.T, M2.T)[0].T

def three_points_to_three_points_pixel_coords(p1,q1,r1,p2,q2,r2,size):
    """returns SL(2,C) matrix that sends the three pixel coordinate points a1,b1,c1 to a2,b2,c2"""
    #convert to sphereical coordinates
    p1,q1,r1,p2,q2,r2 = [CP1_from_sphere( sphere_from_pixel_coords(point, size = size) ) for point in [p1,q1,r1,p2,q2,r2]]
    return two_triples_to_SL(p1,q1,r1,p2,q2,r2)

def normalize_vectors(vecs,ord=None,axis=None):
    """normalize vectors along axis using numpy.linalg.norm. See its documentation for details on ord"""
    return vecs/np.linalg.norm(vecs,ord=ord,axis=axis,keepdims=True)

def get_vector_perp_to_p_and_q(p,q):
    """p and q are distinct points on sphere, return a unit vector perpendicular to both"""
    if abs(np.dot(p,q)+1) < 0.0001: #awkward case when p and q are antipodal on the sphere
        if abs(np.dot(p, [1,0,0])) > 0.9999: #p is parallel to (1,0,0)
            return np.array([0,1,0])
        else:
            return normalize_vectors(np.cross(p, [1,0,0]))
    else:
        return normalize_vectors(np.cross(p,q))

def rotate_sphere_points_p_to_q(p,q):
    """p and q are points on the sphere, return SL(2,C) matrix rotating image of p to image of q on CP^1"""
    if abs(np.dot(p,q)-1) < 0.0001:
        return np.eye(2) #2d ident matrix
    CP1p, CP1q = CP1_from_sphere(p), CP1_from_sphere(q)
    r = get_vector_perp_to_p_and_q(p,q)
    CP1r, CP1nr = CP1_from_sphere(r), CP1_from_sphere(-r)
    return two_triples_to_SL(CP1p, CP1r, CP1nr, CP1q, CP1r, CP1nr)

def rotate_pixel_coords_p_to_q(p, q, size):
    """p and q are pixel coordinate points, return SL(2,C) matrix rotating image of p to image of q in CP^1"""
    p = sphere_from_pixel_coords(p,size)
    q = sphere_from_pixel_coords(q,size)
    return rotate_sphere_points_p_to_q(p,q)

def rotate_around_axis_sphere_points_p_q(p,q,theta):
    """p and q are points on sphere, return SL(2,C) matrix rotating by angle theta around the axis from p to q"""
    assert dot(p,q) < 0.9999, "axis points should not be in the same place!"
    CP1p, CP1q = CP1_from_sphere(p), CP1_from_sphere(q)
    r = get_vector_perp_to_p_and_q(p,q)
    CP1r = CP1_from_sphere(r)
    M_std = two_triples_to_SL(CP1p, CP1q, CP1r, [0,1], [1,0], [1,1])
    M_th = np.array([[complex(np.cos(theta), np.sin(theta)),0],[0,1]]) #rotate on axis through 0, inf by theta
    return np.dot(np.linalg.lstsq(M_std, M_th)[0], M_std) # inv(M_std) @ M_th @ M_std

def rotate_around_axis_pixel_coords_p_q(p,q,theta,size):
    """p and q are pixel coordinate points, return SL(2,C) matrix rotating by angle theta around the axis from p to q"""
    p = sphere_from_pixel_coords(p,size)
    q = sphere_from_pixel_coords(q,size)
    return rotate_around_axis_sphere_points_p_q(p,q,theta)

def rotate_around_axis_pixel_coord_p(p,theta,size):
    p = sphere_from_pixel_coords(p,size)
    np = -p
    return rotate_around_axis_sphere_points_p_q(p,np,theta)

def zoom_in_on_pixel_coords(p, zoom_factor, size):
    M_rot = rotate_pixel_coords_p_to_q(p,[0,0],size)
    M_scl = np.array([[zoom_factor,0],[0,1]])
    return np.dot(np.linalg.lstsq(M_rot, M_scl)[0], M_rot)

def zoom_along_axis_sphere_points_p_q(p, q, zoom_factor):
    assert np.dot(p,q) < 0.999, "points should not be in the same place!"
    CP1p, CP1q = CP1_from_sphere(p), CP1_from_sphere(q)
    r = get_vector_perp_to_p_and_q(p,q)
    CP1r = CP1_from_sphere(r)
    M_std = two_triples_to_SL(CP1p, CP1q, CP1r, [0,1], [1,0], [1,1])
    M_th = [[zoom_factor,0],[0,1]]
    return np.dot(np.linalg.lstsq(M_std, M_th)[0], M_std)

def zoom_along_axis_pixel_coords_p_q(p, q, zoom_factor, size):
    p = sphere_from_pixel_coords(p,size)
    q = sphere_from_pixel_coords(q,size)
    return zoom_along_axis_sphere_points_p_q(p,q,zoom_factor)

def translate_along_axis_pixel_coords(p,q,r1,r2,size):
    return three_points_to_three_points_pixel_coords(p,q,r1,p,q,r2,size)

### Apply tranformations

def apply_SL2C_elt_to_image(M_SL2C, src_image, out_size=None):

    s_im = np.atleast_3d(src_image)
    in_size = s_im.shape[:-1]
    if out_size is None:
        out_size = in_size
    #We are going to find the location in the source image that each pixel in the output image comes from

    #least squares matrix inversion (find X such that M @ X = I ==> X = inv(M) @ I = inv(M))
    Minv = np.linalg.lstsq(M_SL2C, np.eye(2))[0]
    #all of the x,y pairs in o_im:
    pts_out = np.indices(out_size).reshape((2,-1)) #results in a 2 x (num pixels) array of indices
    pts_out_a = angles_from_pixel_coords(pts_out, out_size)
    pts_out_s = sphere_from_angles(pts_out_a)
    pts_out_c = CP1_from_sphere(pts_out_s)
    pts_in_c = np.dot(Minv, pts_out_c) # (2x2) @ (2xn) => (2xn)
    pts_in_s = sphere_from_CP1(pts_in_c)
    pts_in_a = angles_from_sphere(pts_in_s)
    pts_in = pixel_coords_from_angles(pts_in_a, in_size)
    #reshape pts into 2 x image_shape for the interpolation
    o_im = get_interpolated_pixel_color(pts_in.reshape((2,)+out_size), s_im, in_size)

    return o_im

def main():
    #PIL images index by (x,y), but numpy goes by order in memory: (y,x)
    source_image = np.array(Image.open('equirectangular_test_image.png'),dtype=np.float32)
    sz = source_image.shape[:-1]
    M = zoom_in_on_pixel_coords((179.5,360), 2, sz)

    out_image = apply_SL2C_elt_to_image(M, source_image)
    np.clip(out_image,0,255,out_image)
    Image.fromarray(out_image.astype(np.uint8)).save('test_image.png')



if __name__ == '__main__':
    main()