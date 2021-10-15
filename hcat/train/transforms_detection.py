import torch
import scipy.ndimage as ndimage
import skimage.exposure as exposure
import skimage.transform as transform
import numpy as np
import copy
import glob
import skimage.io as io
import skimage.morphology
import cv2
import elasticdeform


# DECORATOR
def joint_transform(func):
    """
    Behavior: Makes a function written to randomly transform an image, and abstracts it to apply identical transforms to
    lists of images.
    i.e.
    [image_1, image_2, image_3] -> joint_transform(fun) = [fun(image_1, seed), fun(image_2, seed), fun(image_3, seed)]

    Will generate a random seed which will be used to randomly transform the image. Passing the seed to each function
    allows the same random function to be applied to every image in the list.

    Each function must be:
        1) __call__ method of a class with arguments self, image, seed
                class.__call__(self, image, seed)
                image: numpy.ndarray which the transform is applies
                seed: number passed to np.random.seed(seed)
                        -- ensures the same transform is identically applied to each image
                        -- Does not have to be used in function, must be an argument!
                            + Applying this decorator to a function without seed WILL NOT WORK
                            i.e.
                                @joint_tranform
                                fun __call__(self, image) <--- MUST HAVE SEED ARG EVEN IF IT ISN'T USD
                                    return image

        2) function with arguments image, seed
                @joint_transform
                def example(image, seed)
                    " DO STUFF HERE "
                    return transformed_image
                image: numpy.ndarray which the transform is applies
                seed: number passed to np.random.seed(seed). This must be an argument even if seed is not used.

    Edge Cases:
    - If the user passes a single image, not in a list, will apply the transform and return the transformed image
      (not in a list).

    - Requires each image in the list to be the same number of dimensions!
    - Requires each image to have a method: ndim.
        -- torch.tensor, numpy.ndarray, etc...
        -- example: numpy.ndarray([1,2,3]).ndim -> 1


    :param func: Function with arguments 'image' and 'seed'
    :return: Wrapped function that can now accept lists
    """

    def wrapper(*args):
        image_list = args[-1]  # In the case of a class function, there may be two args, one is 'self'

        # We only want to take the last argument, which should always be the image list
        # If its not (likely just a single image), put it in a list!
        if not type(image_list) == list:
            image_list = [image_list]

        # Check if all the images are the same size!
        if len(image_list) > 1:
            for i in range(len(image_list) - 1):
                if not image_list[i].ndim == image_list[i + 1].ndim:
                    raise ValueError('Images in joint transforms do not contain identical dimensions.'
                                     + f'Im {i}.ndim:{image_list[i].ndim} != Im {i + 1}.ndim:{image_list[i + 1].ndim} ')
        out = []

        # Generate a random seed value
        seed = np.random.randint(0, 1e8, 1)

        # Evaluate the transform on each function!
        for im in image_list:
            if len(args) > 1:  # Is it a class? Assume at least 2 args... class.method(self, arg_1, ...)
                out.append(func(args[0], image=im, seed=seed))
            else:  # Probably not a class! Dont need to pass 'self'.
                out.append(func(image=im, seed=seed))

        # If a single image was passed initially, return a single image instead of a list of images
        if len(out) == 1:
            out = out[0]
        return out

    return wrapper


class to_float:
    """
    Takes a numpy image matrix of type uint8 or uint16 (standard leica confocal image conventions) and
    rescale to a float between 0 and 1.
    """

    def __init__(self):
        pass

    @joint_transform
    def __call__(self, image, seed=None):
        if image.dtype == 'uint16':
            image = image.astype(dtype='float', casting='same_kind', copy=False)
            image /= 2 ** 16
        elif image.dtype == 'uint8':
            image = image.astype('float', copy=False, casting='same_kind')
            image /= 2 ** 8
        elif image.dtype == 'float':
            pass
        else:
            raise TypeError('Expected image datatype of uint8 or uint16 ')
        return image


class to_tensor:
    def __init__(self):
        pass

    @joint_transform
    def __call__(self, image, seed=None):
        """
        Function which reformats a numpy array of [x,y,z,c] to [1, c, x, y, z]

        :param image: 2D or 3D ndarray with the channel in the last index
        :return: 2D or 3D torch.Tensor formated for input into a convolutional neural net
        """
        if not isinstance(image, np.ndarray):
            raise TypeError(f'Expected list but got {type(image)}')
        num_dims = len(image.shape)
        image = torch.as_tensor(image, dtype=torch.half)
        if not image.shape[0] <= 3:
            # Performs these operations in this order...
            # [x,y,z,c] -> [1,x,y,z,c] -> [c,x,y,z,1] -> [c,x,y,z] -> [1,c,x,y,z]
            return image.unsqueeze(0).transpose(num_dims, 0).squeeze(dim=image.dim()).unsqueeze(0)
        else:
            return image.unsqueeze(0)


class reshape:
    def __init__(self):
        pass

    @joint_transform
    def __call__(self, image, seed=None):
        """
        Expects image dimensions to be [Z,Y,X,C] or [Y,X,C]
            (this is how skimage.io.imread outputs 3D tifs), we add a channel if necessary

        Reshapes to [x,y,z,c]

        :param image:
        :return:
        """
        if not isinstance(image, np.ndarray):
            raise TypeError(f'Expected input type of np.ndarray but got {type(image)}')
        return image.swapaxes(len(image.shape) - 2, 0)


class spekle:
    def __init__(self, gamma=.1):
        """
        :param gamma: float | Maximum intensity of noise added to each channel
        """
        self.gamma = gamma

    def __call__(self, image):
        """
        :param image: np.ndarray(dtype=np.float)
        :return:
        """
        if not image.dtype == 'float':
            raise TypeError(f'Expected image datatype to be float but got {image.dtype}')
        if self.gamma > 1:
            raise ValueError(f'Maximum spekle gamma should be less than 1 [ gamma =/= {self.gamma} ]')

        image_shape = np.shape(image)
        noise = np.random.normal(0, self.gamma, image_shape)
        noise = np.float32(noise)
        image = image + noise
        image[image < 0] = 0
        image[image > 1] = 1

        return image


class random_gamma:
    def __init__(self, gamma_range=(.8, 1.2)):
        self.gamma_range = gamma_range

    def __call__(self, image: np.float):
        if not image.dtype == 'float':
            raise TypeError(f'Expected image dataype to be float but got {image.dtype}')

        factor = np.random.uniform(self.gamma_range[0], self.gamma_range[1], 1)
        if factor < 0:
            factor = 0
        return exposure.adjust_gamma(image, factor)


class random_affine:
    def __init__(self):
        raise NotImplemented('hcat.transforms.random_affine is currently in an unusable state. Please do not use.')
        pass

    @joint_transform
    def __call__(self, image, seed):
        """
        Expects a list of numpy.ndarrays of the same shape. Randomly generates an affine
        transformation matrix that transforms only the x,y dimension
        and applies it to all images in list
        :param image:
        :param seed:
        :return:
        """
        if not isinstance(image, np.ndarray):
            raise TypeError(f'Expected list of images as input, but got {type(image)}')

        np.random.seed(seed)
        translation_x, translation_y = np.random.uniform(0, .5, size=2)

        # generate affine matrix
        mat = np.eye(image.ndim)
        mat[0, 1] = translation_x
        mat[1, 0] = translation_y

        out = ndimage.affine_transform(image.astype(np.float), mat, order=0, output_shape=image.shape, mode='reflect')
        return out.round()


class random_rotate:
    def __init__(self, angle=None):
        self.angle = angle

    @joint_transform
    def __call__(self, image, seed):
        """
        Expects a list of numpy.ndarrays of all the same shape. Randomly rotates the image along x or y dimension
        and returns list of rotated images

        :param image:
        :return:
        """
        if not isinstance(image, np.ndarray):
            raise TypeError(f'Expected list of images as input, but got {type(image)}')

        if not self.angle:
            np.random.seed(seed)
            theta = np.random.randint(0, 360, 1)[0]
        else:
            theta = self.angle

        # We need the shape to be something like [X Y Z C]
        return ndimage.rotate(image.astype(np.float), axes=(0, 1), angle=theta, reshape=False, order=0,
                              mode='constant', prefilter=False)


class normalize:
    def __init__(self, mean=None, std=None):

        if mean is None:
            mean = [0.5, 0.5, 0.5, 0.5]
        if std is None:
            std = [0.5, 0.5, 0.5, 0.5]

        self.mean = mean
        self.std = std

    def __call__(self, image):
        if isinstance(image, list):
            image = image[0]
        shape = image.shape
        for channel in range(shape[0]):
            image[channel-1,...] -= self.mean[channel]
            image[channel-1, ...] /= self.std[channel]

        return image


class drop_channel:
    def __init__(self, chance):
        self.chance = chance

    def __call__(self, image):
        """
        assume in [x,y,z,c]
        :param image:
        :return:
        """
        if np.random.random() > self.chance:
            i = np.random.randint(0, image.shape[-1])
            image[:, :, :, i] = 0
        return image


class random_intensity:
    def __init__(self, range=(-30, 30), chance=0):
        self.range = range
        self.chance = chance

    def __call__(self, image):
        """
        assume in [x,y,z,c]
        :param image:
        :return:
        """
        if not isinstance(image, np.ndarray):
            raise TypeError(f'Expected image to be of type np.ndarray, not {type(image)}')

        val = np.random.randint(self.range[0], self.range[1], image.shape[-1]) / 100

        if image.ndim == 4:
            for c in range(image.shape[-1]):
                if np.random.random() > self.chance:
                    image[:, :, :, c] -= val[c]

        elif image.ndim == 3:
            for c in range(image.shape[-1]):
                if np.random.random() > self.chance:
                    image[:, :, c] -= val[c]

        else:
            raise IndexError(f'Image should have 3 or 4 dimmensions, not {image.ndim} with shape {image.shape}')

        image[image < 0] = 0
        image[np.isnan(image)] = 0
        image[np.isinf(image)] = 1

        return image


class random_crop:
    def __init__(self, dim):
        self.dim = np.array(dim)

    @joint_transform
    def __call__(self, image, seed):
        """
        Expect numpy ndarrays with color in the last channel of the array
        [x,y,z,c] or , [x,y,c]

        :param image:
        :return:
        """
        if not isinstance(image, np.ndarray):
            raise ValueError(f'Expected input to be list but got {type(image)}')
        dim = copy.copy(self.dim)

        np.random.seed(seed)

        # Sometimes we pass a small image in. If the Z is smaller than user specified, just take the whole Z
        # For now only do it with Z. I imagine if an x or y is super small it may be best to just throw out the image
        if not np.all(image.shape[0:-1:1] >= np.array(dim)):
            if image.shape[-2] < dim[-1]:
                dim[-1] = image.shape[-2]
            elif image.shape[0] >= dim[0]:
                dim[0] = image.shape[0]
            elif image.shape[1] >= dim[1]:
                dim[1] = image.shape[1]
            else:
                raise IndexError(f'Output dimensions: {dim} are larger than input image: {image.shape}')

        if image.ndim == 4:  # 3D image
            shape = image.shape[0:-1]
            ind = shape < dim
            for i, val in enumerate(ind):
                if val:
                    dim[i] = shape[i]

            x = int(np.random.randint(0, shape[0] - dim[0] + 1, 1))
            y = int(np.random.randint(0, shape[1] - dim[1] + 1, 1))

            if dim[2] < shape[2]:
                z = int(np.random.randint(0, shape[2] - dim[2] + 1, 1))
            else:
                z = 0
                dim[2] = shape[2]

            out = image[x:x + dim[0]:1, y:y + dim[1]:1, z:z + dim[2]:1, :]

        elif image.ndim == 3:  # 2D image
            shape = image.shape
            x = np.random.randint(0, dim[0] - shape[0] + 1, 1)
            y = np.random.randint(0, dim[1] - shape[1] + 1, 1)

            out = image[x:x + dim[0] - 1:1, y:y + dim[1] - 1:1, :]

        else:
            raise IndexError(f'Image must have dimensions 3 or 4. Not {image.ndim}')

        return out


class elastic_deform:
    def __init__(self, grid_shape=(5, 5, 5), scale=5):
        self.x_grid = grid_shape[0]
        self.y_grid = grid_shape[1]

        if len(grid_shape) > 2:
            self.z_grid = grid_shape[2]
        else:
            self.z_grid = None

        self.scale = scale

    @joint_transform
    def __call__(self, image: np.ndarray, seed) -> np.ndarray:
        """
        Expect numpy ndarrays with color in the last channel of the array
        [x,y,z,c] or , [x,y,c]
        Performs random elastic deformations to the image

        :param image:
        :return:
        """
        if not isinstance(image, np.ndarray):
            raise ValueError(f'Expected input to be numpy.ndarray but got {type(image)}')

        dtype = image.dtype

        np.random.seed(seed)

        if image.ndim == 4:
            # generate a deformation grid
            if self.z_grid is not None:
                displacement = np.random.randn(3, self.x_grid, self.y_grid, self.z_grid) * self.scale
            else:
                raise ValueError('Misspecified deformation vector shape. Should look like: tuple (X, Y, Z)')

            if image.shape[-1] > 1:
                image = elasticdeform.deform_grid(image, displacement, axis=(0, 1, 2))

            # Try to detect if its the mask. Then just dont interpalate.
            elif image.shape[-1] == 1:
                image = elasticdeform.deform_grid(image, displacement, axis=(0, 1, 2), order=0)
            else:
                raise ValueError('Dun Fucked Uterp')

            image[image < 0] = 0
            image[image > 1] = 1
            image.astype(dtype)

        elif image.ndim == 3:
            # generate a deformation grid
            displacement = np.random.randn(2, self.x_grid, self.y_grid) * self.scale
            image = elasticdeform.deform_grid(image, displacement, axis=(0, 1))

        else:
            raise ValueError(f'Expected np.ndarray with 3 or 4 dimmensions, '
                             f'NOT dim: {image.ndim},  shape: {image.shape}')

        return image


# Cant be a @joint_transform because it needs info from one image to affect transforms of other
class nul_crop:

    def __init__(self, rate=1):
        self.rate = rate

    # Cant be a @joint_transform because it needs info from one image to affect transforms of other
    def __call__(self, image_list: list) -> list:
        """
        IMAGE MASK PWL
        :param image_list: list of images with identical number of dimensions
        :return:
        """
        if not isinstance(image_list, list):
            raise ValueError(f'Expected input to be list but got {type(image_list)}')

        if np.random.random() < self.rate:
            out = []
            mask = image_list[1]
            lr = mask.sum(axis=1).sum(axis=1).flatten() > 1
            for i, im in enumerate(image_list):
                image_list[i] = im[lr, :, :, :]

            mask = image_list[1]
            ud = mask.sum(axis=0).sum(axis=1).flatten() > 1
            for i, im in enumerate(image_list):
                out.append(im[:, ud, :, :])
        else:
            out = image_list

        return out


#  FROM FASTER_RCNN CODE

class random_x_flip:
    def __init__(self, rate=.5):
        self.rate = rate

    def __call__(self, image, boxes):
        """
        Code specifically for transforming images and boxes for fasterRCNN
        Not Compatable with UNET

        :param image:
        :param boxes:
        :return:
        """
        flip = np.random.uniform(0, 1, 1) < self.rate

        shape = image.shape
        boxes = np.array(boxes)

        if flip:
            image = np.copy(image[::-1, :, :])
            boxes[:, 1] = (boxes[:, 1] * -1) + shape[0]
            boxes[:, 3] = (boxes[:, 3] * -1) + shape[0]

        new_boxes = np.copy(boxes)
        # new_boxes[:,0] = np.min(boxes[:,0:3:2],axis=1)
        new_boxes[:, 1] = np.min(boxes[:, 1:4:2], axis=1)
        # new_boxes[:,2] = np.max(boxes[:,0:3:2],axis=1)
        new_boxes[:, 3] = np.max(boxes[:, 1:4:2], axis=1)
        boxes = np.array(new_boxes, dtype=np.int64)

        return image, boxes.tolist()


class random_y_flip:
    def __init__(self, rate=.5):
        self.rate = rate

    def __call__(self, image, boxes):
        """
        FASTER RCNN ONLY

        :param image:
        :param boxes:
        :return:
        """

        flip = np.random.uniform(0, 1, 1) > 0.5

        shape = image.shape
        boxes = np.array(boxes, dtype=np.int)

        if flip:
            image = np.copy(image[:, ::-1, :])
            boxes[:, 0] = (boxes[:, 0] * -1) + shape[1]
            boxes[:, 2] = (boxes[:, 2] * -1) + shape[1]

        new_boxes = np.copy(boxes)
        new_boxes[:, 0] = np.min(boxes[:, 0:3:2], axis=1)
        # new_boxes[:,1] = np.min(boxes[:,1:4:2],axis=1)
        new_boxes[:, 2] = np.max(boxes[:, 0:3:2], axis=1)
        # new_boxes[:,3] = np.max(boxes[:,1:4:2],axis=1)
        boxes = np.array(new_boxes, dtype=np.int64)

        return image, boxes.tolist()


class random_resize:
    def __init__(self, rate=.5, scale=(.8, 1.2)):
        self.rate = rate
        self.scale = scale

    def __call__(self, image, boxes):
        """
        FASTER RCNN ONLY


        :param image:
        :param boxes:
        :return:
        """

        scale = np.random.uniform(self.scale[0] * 100, self.scale[1] * 100, 1) / 100
        shape = image.shape

        new_shape = np.round(shape * scale)
        new_shape[2] = shape[2]

        image = transform.resize(image, new_shape)

        boxes = np.array(boxes) * scale
        boxes = np.round(boxes).round()
        boxes = np.array(boxes, dtype=np.int64)

        return image, boxes.tolist()


class remove_channel:

    def __init__(self, remaining_channel_index=(0, 2, 3)):
        self.index_remain = remaining_channel_index

    def __call__(self, image):
        """
        Assumes channels are at last dimension of tensor!

        :param image:
        :return:
        """

        if image.shape[0] == len(self.index_remain):
            return image
        elif image.shape[0] < len(self.index_remain):
            c, x, y = image.shape
            new_image = np.zeros((c + 1, x, y))
            new_image[1:c+1, ...] = image
            return new_image
        elif image.shape[0] > len(self.index_remain):
            return image[self.index_remain, ...]
        else:
            raise NotImplementedError(f'Something is wrong. {image.shape, self.index_remain}',
                                      f'')


class clean_image:
    """
    Simple transform ensuring no nan values are passed to the model.

    """

    def __init__(self):
        pass

    @joint_transform
    def __call__(self, image, seed):
        dtype = image.dtype
        image[np.isnan(image)] = 0
        image[np.isinf(image)] = 1

        return image.astype(dtype=dtype)


class add_junk_image:
    def __init__(self, path, channel_index=(0, 2, 3), junk_image_size=(100, 100), normalize=None):
        """
        Take in a path to images that are p junk. Make sure the images are
        :param path:
        :param channel_index: if the image has more channels than channel index, than it reduces down
        """

        self.index_remain = channel_index
        self.path = path

        if self.path[-1] != '/':
            self.path += '/'

        self.files = glob.glob(self.path + '*.tif')

        if len(self.files) < 1:
            raise FileNotFoundError(f'No valid *.tif files found at {path}')

        self.junk_image_size = junk_image_size
        self.normalize = normalize

        if isinstance(self.normalize, dict):
            self.mean = normalize['mean']
            self.std = normalize['std']

        self.images = []
        for file in self.files:
            im = io.imread(file)
            im = to_float()(im)
            if isinstance(self.normalize, dict):
                for i in range(im.shape[-1]):
                    im[:, :, i] -= self.mean[i]
                    im[:, :, i] /= self.std[i]

            # Check to see if the image has the right number of color channels
            # If not, we drop one channel by the indexes defined by the user with self.index_remain
            if not im.shape[-1] == len(self.index_remain):
                im = im[:, :, self.index_remain]

            self.images.append(im)

    def __call__(self, image, boxes):
        """
        Loads a junk image and crops it randomly. Then adds it to image and removes boxes that overlap with
        the added region.

        :param image:
        :param boxes:
        :return:
        """
        file_index = np.random.randint(0, len(self.files))
        junk_image = self.images[file_index]

        shape = junk_image.shape

        try:
            x = np.random.randint(0, shape[0] - (self.junk_image_size[0] + 1))
            y = np.random.randint(0, shape[1] - (self.junk_image_size[1] + 1))
        except ValueError:
            raise ValueError(
                f'Junk image file {self.files[file_index]} has a shape ({shape}) smaller than max user defined shape')

        junk_image = junk_image[x:x + self.junk_image_size[0], y:y + self.junk_image_size[1], :]

        shape = image.shape
        x = np.random.randint(0, shape[0] - (self.junk_image_size[0] + 1))
        y = np.random.randint(0, shape[1] - (self.junk_image_size[1] + 1))
        image[x:x + self.junk_image_size[0], y:y + self.junk_image_size[1], :] = junk_image

        # VECTORIZE FOR SPEED????

        for i, box in enumerate(boxes):
            box = np.array(box)
            box_x = box[[0, 2]]
            box_y = box[[1, 3]]

            a = box_x < (x + self.junk_image_size[0])
            b = box_x > x
            c = np.logical_and(a, b)
            if np.any(c):
                boxes.pop(i)
                continue

            a = box_y < (y + self.junk_image_size[1])
            b = box_y > y
            c = np.logical_and(a, b)
            if np.any(c):
                boxes.pop(i)
                continue

        return image, boxes


class random_selection:
    def __init__(self, size=(1024, 1024)):
        self.size = size
        self.w = size[0]
        self.h = size[1]

    def __call__(self, image: torch.Tensor, boxes: torch.Tensor, class_labels):
        boxes = torch.tensor(boxes)
        shape = image.shape


        x_max = shape[1] - self.w if shape[1] - self.w > 0 else 1
        y_max = shape[2] - self.h if shape[2] - self.h > 0 else 1

        ind = torch.tensor([0])
        i = 0
        while ind.sum() == 0:
            i+=1
            x = torch.randint(x_max, (1, 1)).item()
            y = torch.randint(y_max, (1, 1)).item()

            # Check if the crop doesnt contain any positive labels.
            # If it does, try generating new points
            # We want to make sure every training image has something to learn

            new_img = self._crop(image, y, x, self.h, self.w)

            # boxes = [N, 4] [x0, y0, x1, y1]
            ind_x = torch.logical_and(boxes[:, 0] > x, boxes[:, 2] < x + self.w)
            ind_y = torch.logical_and(boxes[:, 1] > y, boxes[:, 3] < y + self.h)
            ind = torch.logical_and(ind_y, ind_x)


        box_abridged = boxes[ind, :]
        box_abridged[:, 0] -= x
        box_abridged[:, 1] -= y
        box_abridged[:, 2] -= x
        box_abridged[:, 3] -= y

        return new_img, box_abridged, class_labels[ind]

    @staticmethod
    def _crop(image, x, y, w, h):
        return image[..., x:x + w, y:y + h]


def distance_transform(image):
    """
    Distance transform over z plane for the 3d volume by only going each z stack
    Meant for segmentation mask target.

    :param image: np.ndarray [z, x, y, c]
    :return: distance mat of image with shape [z, x, y, c]
    """
    # Assume image is in a standard state from io.imread [Z, Y/X, X/Y, C?]

    if not isinstance(image, np.ndarray):
        raise ValueError(f'Image must be a numpy ndarray, not {type(image)}...')
    if image.dtype != np.uint8:
        raise ValueError(f'Image dtype is not int: {image.dtype}')

    distance = np.zeros(image.shape, dtype=np.float)
    image = skimage.morphology.binary_dilation(image)

    if image.ndim == 4:
        for i in range(image.shape[0]):
            distance[i, :, :, :] = cv2.distanceTransform(image[i, :, :, :].astype(np.uint8), cv2.DIST_L2, 5)
    else:
        for i in range(image.shape[0]):
            distance[i, :, :] = cv2.distanceTransform(image[i, :, :].astype(np.uint8), cv2.DIST_L2, 5)
    return distance
