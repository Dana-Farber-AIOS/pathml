import numpy as np
import spams
import os

from pathml.preprocessing.base import ImageTransform
from pathml.preprocessing.utils import RGB_to_OD, normalize_matrix_cols


class StainNormalizationHE(ImageTransform):
    """
    Normalize H&E stained images to a reference slide.
    Also can be used to separate hematoxylin and eosin channels.

    H&E images are assumed to be composed of two stains, each one having a vector of its characteristic RGB values.
    The stain matrix is a 3x2 matrix where the first column corresponds to the hematoxylin stain vector and the second
    corresponds to eosin stain vector. The stain matrix can be estimated from a reference image in a number of ways;
    here we provide implementations of two such algorithms from Macenko et al. and Vahadane et al.

    After estimating the stain matrix for an image_ref, the next step is to assign stain concentrations to each pixel.
    Each pixel is assumed to be a linear combination of the two stain vectors, where the coefficients are the
    intensities of each stain vector at that pixel. TO solve for the intensities, we use least squares in Macenko
    method and lasso in vahadane method.

    The image_ref can then be reconstructed by applying those pixel intensities to a stain matrix. This allows you to
    standardize the appearance of an image_ref by reconstructing it using a reference stain matrix. Using this method of
    normalization may help account for differences in slide appearance arising from variations in staining procedure,
    differences between scanners, etc. Images can also be reconstructed using only a single stain vector, e.g. to
    separate the hematoxylin and eosin channels of an H&E image_ref.

    This code is based in part on StainTools: https://github.com/Peter554/StainTools

    :param target: one of 'normalize', 'hematoxylin', or 'eosin'. Defaults to 'normalize'
    :type target: str, optional
    :param stain_estimation_method: method for estimating stain matrix. Must be one of 'macenko' or 'vahadane'.
        Defaults to 'macenko'.
    :type stain_estimation_method: str, optional
    :param optical_density_threshold: Threshold for removing low-optical density pixels when estimating stain
        vectors. Defaults to 0.15
    :type optical_density_threshold: float, optional
    :param sparsity_regularizer: Regularization parameter for dictionary learning when estimating stain vector
        using vahadane method. Ignored if ``concentration_estimation_method!="vahadane"``. Defaults to 1.0
    :type sparsity_regularizer: float, optional
    :param angular_percentile: Percentile for stain vector selection when estimating stain vector
        using Macenko method. Ignored if ``concentration_estimation_method!="macenko"``. Defaults to 0.01
    :type angular_percentile: float, optional
    :param regularizer_lasso: regularization parameter for lasso solver. Defaults to 0.01.
        Ignored if ``method != 'lasso'``
    :type regularizer_lasso: float, optional
    :param background_intensity: Intensity of background light. Must be an integer between 0 and 255.
        Defaults to 245.
    :type background_intensity: int, optional
    :param stain_matrix_target_od: Stain matrix for reference slide.
        Matrix of H and E stain vectors in optical density (OD) space.
        Stain matrix is (3, 2) and first column corresponds to hematoxylin.
        Default stain matrix can be used, or you can also fit to a reference slide of your choosing by calling
        :meth:`~pathml.preprocessing.stains.StainNormalizationHE.fit_to_reference`.
    :param max_c_target: Maximum concentrations of each stain in reference slide.
        Default can be used, or you can also fit to a reference slide of your choosing by calling
        :meth:`~pathml.preprocessing.stains.StainNormalizationHE.fit_to_reference`.

    References
        Macenko, M., Niethammer, M., Marron, J.S., Borland, D., Woosley, J.T., Guan, X., Schmitt, C. and Thomas, N.E.,
        2009, June. A method for normalizing histology slides for quantitative analysis. In 2009 IEEE International
        Symposium on Biomedical Imaging: From Nano to Macro (pp. 1107-1110). IEEE.

        Vahadane, A., Peng, T., Sethi, A., Albarqouni, S., Wang, L., Baust, M., Steiger, K., Schlitter, A.M., Esposito,
        I. and Navab, N., 2016. Structure-preserving color normalization and sparse stain separation for histological
        images. IEEE transactions on medical imaging, 35(8), pp.1962-1971.
    """

    def __init__(
            self,
            target="normalize",
            stain_estimation_method="macenko",
            optical_density_threshold=0.15,
            sparsity_regularizer=1.0,
            angular_percentile=0.01,
            regularizer_lasso=0.01,
            background_intensity=245,
            stain_matrix_target_od=np.array([[0.5626, 0.2159], [0.7201, 0.8012], [0.4062, 0.5581]]),
            max_c_target=np.array([1.9705, 1.0308])
    ):
        # verify inputs
        assert target.lower() in ["normalize", "eosin", "hematoxylin"], \
            f"Error input target {target} must be one of 'normalize', 'eosin', 'hematoxylin'"
        assert stain_estimation_method.lower() in ['macenko', 'vahadane'],\
            f"Error: input stain estimation method {stain_estimation_method} must be one of 'macenko' or 'vahadane'"
        assert 0 <= background_intensity <= 255, \
            f"Error: input background intensity {background_intensity} must be an integer between 0 and 255"

        self.target = target.lower()
        self.stain_estimation_method = stain_estimation_method.lower()
        self.optical_density_threshold = optical_density_threshold
        self.sparsity_regularizer = sparsity_regularizer
        self.angular_percentile = angular_percentile
        self.regularizer_lasso = regularizer_lasso
        self.background_intensity = background_intensity
        self.stain_matrix_target_od = stain_matrix_target_od
        self.max_c_target = max_c_target

    def fit_to_reference(self, image_ref):
        """
        Fit ``stain_matrix`` and ``max_c`` to a reference slide. This allows you to use a specific slide as the
        reference for stain normalization. Works by first estimating stain matrix from input reference image,
        then estimating pixel concentrations. Newly computed stain matrix and maximum concentrations are then used
        for any future color normalization.

        :param image_ref: RGB image
        :type image_ref: np.ndarray
        """
        # first estimate stain matrix for reference image_ref
        stain_matrix = self._estimate_stain_vectors(image = image_ref)

        # next get pixel concentrations for reference image_ref
        C = self._estimate_pixel_concentrations(image = image_ref, stain_matrix = stain_matrix)

        # get max concentrations
        # actually use 99th percentile so it's more robust
        max_C = np.percentile(C, 99, axis = 0).reshape((1, 2))

        # put the newly determined stain matrix and max C matrix for reference slide into class attrs
        self.stain_matrix_target_od = stain_matrix
        self.max_c_target = max_C

    def _estimate_stain_vectors(self, image):
        """
        Estimate stain vectors using appropriate method

        :param image: RGB image_ref
        :type image: np.ndarray
        """
        # first estimate stain matrix for reference image_ref
        if self.stain_estimation_method == "macenko":
            stain_matrix = self._estimate_stain_vectors_macenko(image)
        elif self.stain_estimation_method == "vahadane":
            stain_matrix = self._estimate_stain_vectors_vahadane(image)
        else:
            raise Exception(f"Error: input stain estimation method {self.stain_estimation_method} must be one of "
                            f"'macenko' or 'vahadane'")
        return stain_matrix

    def _estimate_pixel_concentrations(self, image, stain_matrix):
        """
        Estimate pixel concentrations from a given stain matrix using appropriate method

        :param image: RGB image_ref
        :param stain_matrix: matrix of H and E stain vectors in optical density (OD) space.
            Stain_matrix is (3, 2) and first column corresponds to hematoxylin by convention.
        """

        if self.stain_estimation_method == "macenko":
            C = self._estimate_pixel_concentrations_lstsq(image, stain_matrix)
        elif self.stain_estimation_method == "vahadane":
            C = self._estimate_pixel_concentrations_lasso(image, stain_matrix)
        else:
            raise Exception(f"Provided target {self.target} invalid")
        return C

    def _estimate_stain_vectors_vahadane(self, image):
        """
        Estimate stain vectors using dictionary learning method from Vahadane et al.

        :param image: RGB image_ref
        :type image: np.ndarray
        """
        # convert to Optical Density (OD) space
        image_OD = RGB_to_OD(image)
        # reshape to (M*N)x3
        image_OD = image_OD.reshape(-1, 3)
        # drop pixels with low OD
        OD = image_OD[np.all(image_OD > self.optical_density_threshold, axis = 1)]

        # dictionary learning
        # need to first update
        # see https://github.com/dmlc/xgboost/issues/1715#issuecomment-420305786
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        dictionary = spams.trainDL(X = OD.T, K = 2, lambda1 = self.sparsity_regularizer, mode = 2,
                                   modeD = 0, posAlpha = True, posD = True, verbose = False)
        dictionary = normalize_matrix_cols(dictionary)
        # order H and E.
        # H on first col.
        if dictionary[0, 0] > dictionary[1, 0]:
            dictionary = dictionary[:, [1, 0]]
        return dictionary

    def _estimate_stain_vectors_macenko(self, image):
        """
        Estimate stain vectors using Macenko method. Returns a (3, 2) matrix with first column corresponding to
        hematoxylin and second column corresponding to eosin in OD space.

        :param image: RGB image_ref
        :type image: np.ndarray
        """
        # convert to Optical Density (OD) space
        image_OD = RGB_to_OD(image)
        # reshape to (M*N)x3
        image_OD = image_OD.reshape(-1, 3)
        # drop pixels with low OD
        OD = image_OD[np.all(image_OD > self.optical_density_threshold, axis = 1)]
        # get top 2 PCs. PCs are eigenvectors of covariance matrix
        try:
            _, v = np.linalg.eigh(np.cov(OD.T))
        except np.linalg.LinAlgError as err:
            print(f"Error in computing eigenvectors: {err}")
            raise
        pcs = v[:, 1:3]
        # project OD pixels onto plane of first 2 PCs
        projected = OD @ pcs
        # Calculate angle of each point on projection plane
        angles = np.arctan2(projected[:, 1], projected[:, 0])
        # get robust min and max angles
        max_angle = np.percentile(angles, 100 * (1 - self.angular_percentile))
        min_angle = np.percentile(angles, 100 * self.angular_percentile)
        # get vector of unit length pointing in that angle, in projection plane
        # unit length vector of angle theta is <cos(theta), sin(theta)>
        v_max = np.array([np.cos(max_angle), np.sin(max_angle)])
        v_min = np.array([np.cos(min_angle), np.sin(min_angle)])
        # project back to OD space
        stain1 = pcs @ v_max
        stain2 = pcs @ v_min
        # a heuristic to make the vector corresponding to hematoxylin first and the
        # one corresponding to eosin second
        if stain2[0] > stain1[0]:
            HE = np.array((stain2, stain1)).T
        else:
            HE = np.array((stain1, stain2)).T
        return HE

    def _estimate_pixel_concentrations_lstsq(self, image, stain_matrix):
        """
        estimate concentrations of each stain at each pixel using least squares

        :param image: RGB image_ref
        :param stain_matrix: matrix of H and E stain vectors in optical density (OD) space.
            Stain_matrix is (3, 2) and first column corresponds to hematoxylin by convention.
        """
        image_OD = RGB_to_OD(image).reshape(-1, 3)

        # Get concentrations of each stain at each pixel
        # image_ref.T = S @ C.T
        #   image_ref.T is 3x(M*N)
        #   stain matrix S is 3x2
        #   concentration matrix C.T is 2x(M*N)
        # solve for C using least squares
        C = np.linalg.lstsq(stain_matrix, image_OD.T, rcond = None)[0].T
        return C

    def _estimate_pixel_concentrations_lasso(self, image, stain_matrix):
        """
        estimate concentrations of each stain at each pixel using lasso

        :param image: RGB image_ref
        :param stain_matrix: matrix of H and E stain vectors in optical density (OD) space.
            Stain_matrix is (3, 2) and first column corresponds to hematoxylin by convention.
        """
        image_OD = RGB_to_OD(image).reshape(-1, 3)

        # Get concentrations of each stain at each pixel
        # image_ref.T = S @ C.T
        #   image_ref.T is 3x(M*N)
        #   stain matrix S is 3x2
        #   concentration matrix C.T is 2x(M*N)
        # solve for C using lasso
        lamb = self.regularizer_lasso
        C = spams.lasso(X = image_OD.T, D = stain_matrix, mode = 2, lambda1 = lamb, pos = True).toarray().T
        return C

    def _reconstruct_image(self, pixel_intensities):
        """
        Reconstruct an image from pixel intensities. Uses reference stain matrix and max_c
        from `fit_to_reference`, if that method has been called, otherwise uses defaults.

        :param pixel_intensities: matrix of stain intensities for each pixel.
            If image_ref is MxN, stain matrix is 2x(M*M)
        :type pixel_intensities: np.ndarray
        """
        # scale to max intensities
        # actually use 99th percentile so it's more robust
        max_c = np.percentile(pixel_intensities, 99, axis = 0).reshape((1, 2))
        pixel_intensities *= (self.max_c_target / max_c)

        if self.target == "normalize":
            im = np.exp(-self.stain_matrix_target_od @ pixel_intensities.T)
        elif self.target == "hematoxylin":
            im = np.exp(- self.stain_matrix_target_od[:, 0].reshape(-1, 1) @ pixel_intensities[:, 0].reshape(-1, 1).T)
        elif self.target == "eosin":
            im = np.exp(- self.stain_matrix_target_od[:, 1].reshape(-1, 1) @ pixel_intensities[:, 1].reshape(-1, 1).T)
        else:
            raise Exception(
                f"Error: input target {self.target} is invalid. Must be one of 'normalize', 'eosin', 'hematoxylin'"
            )

        im = im * self.background_intensity
        im = np.clip(im, a_min = 0, a_max = 255)
        im = im.T.astype(np.uint8)
        return im

    def apply(self, image):
        """
        Apply stain separation and normalization to input image.

        :param image: RGB reference image
        :type image: np.ndarray
        :return: Reconstructed image
        :rtype: np.ndarray
        """
        # first estimate stain matrix for reference image_ref
        stain_matrix = self._estimate_stain_vectors(image = image)

        # next get pixel concentrations for reference image_ref
        C = self._estimate_pixel_concentrations(image = image, stain_matrix = stain_matrix)

        # next reconstruct the image_ref
        im_reconstructed = self._reconstruct_image(pixel_intensities = C)

        im_reconstructed = im_reconstructed.reshape(image.shape)
        return im_reconstructed
