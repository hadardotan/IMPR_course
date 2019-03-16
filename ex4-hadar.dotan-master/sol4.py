import shutil
import numpy as np
import os
import scipy
import matplotlib.pyplot as plt
from scipy.misc import imsave as imsave
from scipy.ndimage.morphology import generate_binary_structure
from scipy import signal
from scipy.ndimage.filters import convolve
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage import label, center_of_mass, interpolation, map_coordinates
import sol4_utils


def harris_corner_detector(im):
    """
  Detects harris corners.
  Make sure the returned coordinates are x major!!!
  :param im: A 2D array representing an image.
  :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
  """
    d_filter = np.array([[1], [0], [-1]])

    # Get the Ix and Iy derivatives of the image using the filters
    i_x = scipy.ndimage.filters.convolve(im, d_filter.transpose())
    i_y = scipy.ndimage.filters.convolve(im, d_filter)

    # Blur the images: Ix 2 , Iy 2 , Ix Iy and create matrix m for each pixel
    m = np.zeros((im.shape[0], im.shape[1], 2, 2))
    m[:, :, 0, 0] = sol4_utils.blur_spatial((i_x * i_x), 3)
    m[:, :, 1, 1] = sol4_utils.blur_spatial((i_y * i_y), 3)
    m[:, :, 1, 0] = sol4_utils.blur_spatial((i_y * i_x), 3)
    m[:, :, 0, 1] = sol4_utils.blur_spatial((i_y * i_x), 3)

    # find response image R
    k = 0.04
    r = np.linalg.det(m) - k * np.square(np.trace(m, axis1=2, axis2=3))

    # find the corners - the local maximum points of R
    max_r = non_maximum_suppression(r)
    corners = np.array(np.nonzero(max_r))

    # create array with shape (N,2) of [x,y] key point locations in im.
    return np.array([corners[1], corners[0]]).transpose()


def sample_descriptor(im, pos, desc_rad):
    """
  Samples descriptors at the given corners.
  :param im: A 2D array representing an image.
  :param pos: An array with shape (N,2), where pos[i,:] are the [x,y] coordinates of the ith corner point.
  :param desc_rad: "Radius" of descriptors to compute.
  :return: A 3D array with shape (N,K,K) containing the ith descriptor at desc[i,:,:].
  """
    k = 1 + 2 * desc_rad
    n = pos.shape[0]
    descriptors = np.zeros((n, k, k))

    # find top left corner of each patch according to points we want to build their patch
    #  (they are the center of the patch)
    corners = np.array([np.add(pos[:, 0], -desc_rad), np.add(pos[:, 1], -desc_rad)])

    # create general patch for shifting according to corners
    shift_range = range(k)
    shift_patch = np.array(np.meshgrid(shift_range, shift_range))

    for i in range(n):
        # get top left corner of current patch
        x, y = corners[0][i], corners[1][i]

        # create current patch using shifting from corners
        current_patch = np.array([np.add(y, shift_patch[1]), np.add(x, shift_patch[0])])

        # sample at sub-pixel coordinates by interpolating within the pyramid’s 3rd level image
        intensity_matrix = scipy.ndimage.map_coordinates(im, current_patch, order=1, prefilter=False)

        # normalize matrix so that the resulting descriptor is invariant to certain changes of lighting
        mean = np.mean(intensity_matrix)
        d_m_norm = np.linalg.norm(intensity_matrix - mean)
        if d_m_norm == 0:
            # print("error?")
            descriptors[i] = np.zeros(intensity_matrix.shape)
        else:
            descriptors[i] = np.divide((intensity_matrix - mean), d_m_norm)

    return descriptors



def translate_pos(x, y, i, j):
    """

    :param x:
    :param y:
    :param i:
    :param j:
    :return:
    """
    trans_x, trans_y = (2.0 ** (i -j )) * x, (2.0 ** (i - j)) * y
    return np.array([trans_x, trans_y]).transpose()

def find_features(pyr):
    """
  Detects and extracts feature points from a pyramid.
  :param pyr: Gaussian pyramid of a grayscale image having 3 levels.
  :return: A list containing:
              1) An array with shape (N,2) of [x,y] feature location per row found in the image.
                 These coordinates are provided at the pyramid level pyr[0].
              2) A feature descriptor array with shape (N,K,K)
  """
    # call spread_out_corners for getting the keypoints (with original image!)
    corners = spread_out_corners(pyr[0], 7, 7, 12)  # radius = 4  (pyr level 2) * 3 (descriptor radius) = 12
    pos = np.array(translate_pos(corners[:, 0], corners[:, 1], 0, 2))
    # call sample_descriptor for sampling a descriptor for each keypoint
    descriptors = sample_descriptor(pyr[2], pos, 3)  # third level of pyr is 4 times smaller than first level
    return [corners, descriptors]


def match_features(desc1, desc2, min_score):
    """
  Return indices of matching descriptors.
  :param desc1: A feature descriptor array with shape (N1,K,K).
  :param desc2: A feature descriptor array with shape (N2,K,K).
  :param min_score: Minimal match score.
  :return: A list containing:
              1) An array with shape (M,) and dtype int of matching indices in desc1.
              2) An array with shape (M,) and dtype int of matching indices in desc2.
  """

    n1, n2 = desc1.shape[0], desc2.shape[0]
    s = np.array(np.zeros((n1, n2)))

    # flat descriptors array according to arrays shapes format to return (now it will be (n,))
    flat1, flat2 = desc1.reshape(n1, desc1.shape[1]**2), desc2.reshape(n2, desc2.shape[1]**2)

    # find match score we choose between two descriptors by their dot- product
    dot_score = np.dot(flat1, flat2.transpose())

    if dot_score.shape != s.shape:
        print("error")

    # 1 - from im1 to im2
    for j in range(n1):
        i = np.argpartition(dot_score[j, :], -2)[-2:]
        s[j, i] += 1

    # 2 - from im2 to im1
    for k in range(n2):
        i = np.argpartition(dot_score[:, k], -2)[-2:]
        s[i, k] += 1

    # 3 - find matches that their score is greater than min match score
    min_match = dot_score > min_score

    # 1 & 2 & 3
    s = s > 1
    s = s & min_match

    # non zero return tuple of np arrays of indices
    return np.nonzero(s)


def apply_homography(pos1, H12):
    """
  Apply homography to inhomogenous points.
  :param pos1: An array with shape (N,2) of [x,y] point coordinates.
  :param H12: A 3x3 homography matrix.
  :return: An array with the same shape as pos1 with [x,y] point coordinates obtained from transforming pos1 using H12.
  """
    ones = np.array(np.ones((pos1.shape[0], 1)))
    pos1 = np.hstack((pos1, ones))
    pos2 = np.dot(H12, pos1.transpose())
    pos2 = pos2 / pos2[2]
    return pos2[:2, :].transpose()



def ransac_homography(points1, points2, num_iter, inlier_tol, translation_only=False):
    """
  Computes homography between two sets of points using RANSAC.
  :param pos1: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 1.
  :param pos2: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 2.
  :param num_iter: Number of RANSAC iterations to perform.
  :param inlier_tol: inlier tolerance threshold.
  :param translation_only: see estimate rigid transform
  :return: A list containing:
              1) A 3x3 normalized homography matrix.
              2) An Array with shape (S,) where S is the number of inliers,
                  containing the indices in pos1/pos2 of the maximal set of inlier matches found.
  """
    n = points1.shape[0]
    max_num_of_inliers = 0
    max_matches = np.array([])

    if n == 0:
        return [np.array(np.zeros((3, 3))), np.array([])]
    elif num_iter == 0:
        return [np.array([]), np.array([])]

    for i in range(num_iter):
        # Pick a random set of 2 point matches from the supplied N point matches.
        j1, j2 = np.random.choice(n, size=2)
        p_j1, p_j2 = np.array([points1[j1], points1[j2]]), np.array([points2[j1], points2[j2]])

        # Compute the homography that transforms the 2 points1 to the 2 points2
        homography = estimate_rigid_transform(p_j1, p_j2, translation_only)

        # Use homography to transform the set of points points1 in image 1 to the transformed set points2'
        trans_points1 = apply_homography(points1, homography)

        # compute the squared euclidean distance ej for j = 0..N − 1
        e = np.array(np.square(np.linalg.norm(trans_points1 - points2, axis=1)))

        # Mark all matches having ej < inlier_tol as inlier matches and the rest as outlier matches
        num_of_inliers = np.count_nonzero(e < inlier_tol)
        if num_of_inliers > max_num_of_inliers:
            max_num_of_inliers = num_of_inliers
            max_homography = homography
            max_matches = np.array(np.nonzero(e < inlier_tol))[0]

    # recompute homography over the larget set of matches
    p_j_in1, p_j_in2 = np.array(np.take(points1, max_matches, axis=0)), np.array(np.take(points2, max_matches, axis=0)) #  not a good pick of points
    max_homography = estimate_rigid_transform(p_j_in1, p_j_in2, translation_only)

    # obtain the final least squares fit of the homography over the largest inlier set

    return [max_homography, max_matches]


def display_matches(im1, im2, points1, points2, inliers):
    """
  Dispalay matching points.
  :param im1: A grayscale image.
  :param im2: A grayscale image.
  :parma pos1: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im1.
  :param pos2: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im2.
  :param inliers: An array with shape (S,) of inlier matches.
  """
    im = np.hstack((im1, im2))
    n = points1.shape[0]
    plt.imshow(im, cmap='gray')
    x1, y1 = points1[:, 0], points1[:, 1]
    x2, y2 = (points2[:, 0] + im1.shape[0]), points2[:, 1]

    for j in range(n):
        x, y = [x1[j], x2[j]], [y1[j], y2[j]]

        if j in inliers:  # plot yellow line between inlier match
            plt.plot(x, y, c='y', mfc='r', lw=.4, ms=10, marker='o', linestyle='dashed')

        else:  # plot blue line between outlier match
            plt.plot(x, y, c='b', mfc='r', lw=.4, ms=10, marker='o', linestyle='dashed')

    plt.show()


def accumulate_homographies(H_succesive, m):
    """
  Convert a list of succesive homographies to a
  list of homographies to a common reference frame.
  :param H_successive: A list of M-1 3x3 homography
    matrices where H_successive[i] is a homography which transforms points
    from coordinate system i to coordinate system i+1.
  :param m: Index of the coordinate system towards which we would like to
    accumulate the given homographies.
  :return: A list of M 3x3 homography matrices,
    where H2m[i] transforms points from coordinate system i to coordinate system m
  """

    # For i=m we set H ̄i,m to the 3×3 identity matrix
    h_2_m = list(H_succesive)
    h_2_m[m] = np.eye(3)

    # multiply  according to the formula Ha,c = Hb,cHa,b
    # i < m
    for i in range(m-1, -1, -1):
        h_2_m[i] = np.dot(h_2_m[i+1], H_succesive[i])
        h_2_m[i] /= h_2_m[i][2, 2]  # normalize to maintain the property that H[2,2]==1.

    # i > m
    for i in range(m+1, len(H_succesive)):
        h_2_m[i] = np.dot(h_2_m[i-1], np.linalg.inv(H_succesive[i]))
        h_2_m[i] /= h_2_m[i][2, 2]

    h_2_m.append(np.dot(h_2_m[-1], np.linalg.inv(H_succesive[-1])))

    # h_2_m[-1] = np.dot(h_2_m[-1], np.linalg.inv(H_succesive[-1]))
    # h_2_m[-1] /= h_2_m[-1][2, 2]

    return h_2_m



def compute_bounding_box(homography, w, h):
    """
  computes bounding box of warped image under homography, without actually warping the image
  :param homography: homography
  :param w: width of the image
  :param h: height of the image
  :return: 2x2 array, where the first row is [x,y] of the top left corner,
   and the second row is the [x,y] of the bottom right corner
  """
    # compute where the 4 corner pixel coordinates of each frame get mapped to by homography
    t_l, t_r, b_l, b_r = np.array([[0, 0]]), np.array([[w, 0]]), np.array([[0, h]]), np.array([[w, h]])
    t_l_x, t_l_y = apply_homography(t_l, homography).transpose()
    t_r_x, t_r_y = apply_homography(t_r, homography).transpose()
    b_l_x, b_l_y = apply_homography(b_l, homography).transpose()
    b_r_x, b_r_y = apply_homography(b_r, homography).transpose()

    all_x, all_y = [t_l_x, t_r_x, b_l_x, b_r_x],  [t_l_y, t_r_y, b_l_y, b_r_y]

    min_x, min_y = min(all_x)[0], min(all_y)[0]
    max_x, max_y = max(all_x)[0], max(all_y)[0]

    return np.array([[min_x, min_y], [max_x, max_y]]).astype(int)


def warp_channel(image, homography):
    """
  Warps a 2D image with a given homography.
  :param image: a 2D image.
  :param homography: homograhpy.
  :return: A 2d warped image.
  """

    # for an input image prepare coordinate strips to hold the x and y coordinates of each of the warped image.
    t_l, b_r = compute_bounding_box(homography, image.shape[1], image.shape[0])
    x_range = np.arange(t_l[0], b_r[0])
    y_range = np.arange(t_l[1], b_r[1])
    x_coords, y_coords = np.array(np.meshgrid(x_range, y_range))

    # transform the coordinate mesh by the inverse homography back to the coordinate system of frame i
    inverse_homography = np.linalg.inv(homography)
    origin_coords = np.array([x_coords, y_coords]).transpose()

    origin_shape = origin_coords.shape
    bw_coords = apply_homography(origin_coords.reshape(-1, 2), inverse_homography).reshape(origin_shape)

    # interpolate the image
    return map_coordinates(image, [bw_coords[:, :, 1].transpose(), bw_coords[:, :, 0].transpose()], order=1, prefilter=False)



def warp_image(image, homography):
    """
  Warps an RGB image with a given homography.
  :param image: an RGB image.
  :param homography: homograhpy.
  :return: A warped image.
  """
    return np.dstack([warp_channel(image[..., channel], homography) for channel in range(3)])


def filter_homographies_with_translation(homographies, minimum_right_translation):
    """
  Filters rigid transformations encoded as homographies by the amount of translation from left to right.
  :param homographies: homograhpies to filter.
  :param minimum_right_translation: amount of translation below which the transformation is discarded.
  :return: filtered homographies..
  """
    translation_over_thresh = [0]
    last = homographies[0][0, -1]
    for i in range(1, len(homographies)):
        if homographies[i][0, -1] - last > minimum_right_translation:
            translation_over_thresh.append(i)
            last = homographies[i][0, -1]
    return np.array(translation_over_thresh).astype(np.int)


def estimate_rigid_transform(points1, points2, translation_only=False):
    """
  Computes rigid transforming points1 towards points2, using least squares method.
  points1[i,:] corresponds to poins2[i,:]. In every point, the first coordinate is *x*.
  :param points1: array with shape (N,2). Holds coordinates of corresponding points from image 1.
  :param points2: array with shape (N,2). Holds coordinates of corresponding points from image 2.
  :param translation_only: whether to compute translation only. False (default) to compute rotation as well.
  :return: A 3x3 array with the computed homography.
  """
    centroid1 = points1.mean(axis=0)
    centroid2 = points2.mean(axis=0)

    if translation_only:
        rotation = np.eye(2)
        translation = centroid2 - centroid1

    else:
        centered_points1 = points1 - centroid1
        centered_points2 = points2 - centroid2

        sigma = centered_points2.T @ centered_points1
        U, _, Vt = np.linalg.svd(sigma)

        rotation = U @ Vt
        translation = -rotation @ centroid1 + centroid2

    H = np.eye(3)
    H[:2, :2] = rotation
    H[:2, 2] = translation
    return H


def non_maximum_suppression(image):
    """
  Finds local maximas of an image.
  :param image: A 2D array representing an image.
  :return: A boolean array with the same shape as the input image, where True indicates local maximum.
  """
    # Find local maximas.

    neighborhood = generate_binary_structure(2, 2)
    local_max = maximum_filter(image, footprint=neighborhood) == image
    local_max[image < (image.max() * 0.1)] = False

    # Erode areas to single points.
    lbs, num = label(local_max)
    centers = center_of_mass(local_max, lbs, np.arange(num) + 1)
    centers = np.stack(centers).round().astype(np.int)
    ret = np.zeros_like(image, dtype=np.bool)
    ret[centers[:, 0], centers[:, 1]] = True

    return ret


def spread_out_corners(im, m, n, radius):
    """
  Splits the image im to m by n rectangles and uses harris_corner_detector on each.
  :param im: A 2D array representing an image.
  :param m: Vertical number of rectangles.
  :param n: Horizontal number of rectangles.
  :param radius: Minimal distance of corner points from the boundary of the image.
  :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
  """
    corners = [np.empty((0, 2), dtype=np.int)]
    x_bound = np.linspace(0, im.shape[1], n + 1, dtype=np.int)
    y_bound = np.linspace(0, im.shape[0], m + 1, dtype=np.int)
    for i in range(n):
        for j in range(m):
            # Use Harris detector on every sub image.
            sub_im = im[y_bound[j]:y_bound[j + 1], x_bound[i]:x_bound[i + 1]]
            sub_corners = harris_corner_detector(sub_im)
            sub_corners += np.array([x_bound[i], y_bound[j]])[np.newaxis, :]
            corners.append(sub_corners)
    corners = np.vstack(corners)
    legit = ((corners[:, 0] > radius) & (corners[:, 0] < im.shape[1] - radius) &
             (corners[:, 1] > radius) & (corners[:, 1] < im.shape[0] - radius))
    ret = corners[legit, :]
    return ret


class PanoramicVideoGenerator:
  """
  Generates panorama from a set of images.
  """

  def __init__(self, data_dir, file_prefix, num_images):
    """
    The naming convention for a sequence of images is file_prefixN.jpg,
    where N is a running number 001, 002, 003...
    :param data_dir: path to input images.
    :param file_prefix: see above.
    :param num_images: number of images to produce the panoramas with.
    """
    self.file_prefix = file_prefix
    self.files = [os.path.join(data_dir, '%s%03d.jpg' % (file_prefix, i + 1)) for i in range(num_images)]
    self.files = list(filter(os.path.exists, self.files))
    self.panoramas = None
    self.homographies = None
    print('found %d images' % len(self.files))

  def align_images(self, translation_only=False):
    """
    compute homographies between all images to a common coordinate system
    :param translation_only: see estimte_rigid_transform
    """
    # Extract feature point locations and descriptors.
    points_and_descriptors = []
    for file in self.files:
      image = sol4_utils.read_image(file, 1)
      self.h, self.w = image.shape
      pyramid, _ = sol4_utils.build_gaussian_pyramid(image, 3, 7)
      points_and_descriptors.append(find_features(pyramid))

    # Compute homographies between successive pairs of images.
    Hs = []
    for i in range(len(points_and_descriptors) - 1):
      points1, points2 = points_and_descriptors[i][0], points_and_descriptors[i + 1][0]
      desc1, desc2 = points_and_descriptors[i][1], points_and_descriptors[i + 1][1]

      # Find matching feature points.
      ind1, ind2 = match_features(desc1, desc2, .7)
      points1, points2 = points1[ind1, :], points2[ind2, :]

      # Compute homography using RANSAC.
      H12, inliers = ransac_homography(points1, points2, 100, 6, translation_only)

      # Uncomment for debugging: display inliers and outliers among matching points.
      # In the submitted code this function should be commented out!
      # display_matches(self.images[i], self.images[i+1], points1 , points2, inliers)

      Hs.append(H12)

    # Compute composite homographies from the central coordinate system.
    accumulated_homographies = accumulate_homographies(Hs, (len(Hs) - 1) // 2)
    self.homographies = np.stack(accumulated_homographies)
    self.frames_for_panoramas = filter_homographies_with_translation(self.homographies, minimum_right_translation=5)
    self.homographies = self.homographies[self.frames_for_panoramas]


  def generate_panoramic_images(self, number_of_panoramas):
    """
    combine slices from input images to panoramas.
    :param number_of_panoramas: how many different slices to take from each input image
    """
    assert self.homographies is not None

    # compute bounding boxes of all warped input images in the coordinate system of the middle image (as given by the homographies)
    self.bounding_boxes = np.zeros((self.frames_for_panoramas.size, 2, 2))
    for i in range(self.frames_for_panoramas.size):
      self.bounding_boxes[i] = compute_bounding_box(self.homographies[i], self.w, self.h)

    # change our reference coordinate system to the panoramas
    # all panoramas share the same coordinate system
    global_offset = np.min(self.bounding_boxes, axis=(0, 1))
    self.bounding_boxes -= global_offset

    slice_centers = np.linspace(0, self.w, number_of_panoramas + 2, endpoint=True, dtype=np.int)[1:-1]
    warped_slice_centers = np.zeros((number_of_panoramas, self.frames_for_panoramas.size))
    # every slice is a different panorama, it indicates the slices of the input images from which the panorama
    # will be concatenated
    for i in range(slice_centers.size):
      slice_center_2d = np.array([slice_centers[i], self.h // 2])[None, :]
      # homography warps the slice center to the coordinate system of the middle image
      warped_centers = [apply_homography(slice_center_2d, h) for h in self.homographies]
      # we are actually only interested in the x coordinate of each slice center in the panoramas' coordinate system
      warped_slice_centers[i] = np.array(warped_centers)[:, :, 0].squeeze() - global_offset[0]

    panorama_size = np.max(self.bounding_boxes, axis=(0, 1)).astype(np.int) + 1

    # boundary between input images in the panorama
    x_strip_boundary = ((warped_slice_centers[:, :-1] + warped_slice_centers[:, 1:]) / 2)
    x_strip_boundary = np.hstack([np.zeros((number_of_panoramas, 1)),
                                  x_strip_boundary,
                                  np.ones((number_of_panoramas, 1)) * panorama_size[0]])
    x_strip_boundary = x_strip_boundary.round().astype(np.int)

    self.panoramas = np.zeros((number_of_panoramas, panorama_size[1], panorama_size[0], 3), dtype=np.float64)
    for i, frame_index in enumerate(self.frames_for_panoramas):
      # warp every input image once, and populate all panoramas
      image = sol4_utils.read_image(self.files[frame_index], 2)
      warped_image = warp_image(image, self.homographies[i])
      x_offset, y_offset = self.bounding_boxes[i][0].astype(np.int)
      y_bottom = y_offset + warped_image.shape[0]

      for panorama_index in range(number_of_panoramas):
        # take strip of warped image and paste to current panorama
        boundaries = x_strip_boundary[panorama_index, i:i + 2]
        image_strip = warped_image[:, boundaries[0] - x_offset: boundaries[1] - x_offset]
        x_end = boundaries[0] + image_strip.shape[1]
        self.panoramas[panorama_index, y_offset:y_bottom, boundaries[0]:x_end] = image_strip

    # crop out areas not recorded from enough angles
    # assert will fail if there is overlap in field of view between the left most image and the right most image
    crop_left = int(self.bounding_boxes[0][1, 0])
    crop_right = int(self.bounding_boxes[-1][0, 0])
    assert crop_left < crop_right, 'for testing your code with a few images do not crop.'
    print(crop_left, crop_right)
    self.panoramas = self.panoramas[:, :, crop_left:crop_right, :]


  def save_panoramas_to_video(self):
    assert self.panoramas is not None
    out_folder = 'tmp_folder_for_panoramic_frames/%s' % self.file_prefix
    try:
      shutil.rmtree(out_folder)
    except:
      print('could not remove folder')
      pass
    os.makedirs(out_folder)
    # save individual panorama images to 'tmp_folder_for_panoramic_frames'
    for i, panorama in enumerate(self.panoramas):
      imsave('%s/panorama%02d.png' % (out_folder, i + 1), panorama)
    if os.path.exists('%s.mp4' % self.file_prefix):
      os.remove('%s.mp4' % self.file_prefix)
    # write output video to current folder
    os.system('ffmpeg -framerate 3 -i %s/panorama%%02d.png %s.mp4' %
              (out_folder, self.file_prefix))


  def show_panorama(self, panorama_index, figsize=(20, 20)):
    assert self.panoramas is not None
    plt.figure(figsize=figsize)
    plt.imshow(self.panoramas[panorama_index].clip(0, 1))
    plt.show()
