Models
======

``PathML`` comes with model architectures ready to use out of the box.

.. table::
    :widths: 20, 20, 60

    ===================================== ============ =============
    Model                                 Reference    Description
    ===================================== ============ =============
    U-net    (in progress)                [Unet]_      A model for segmentation in biomedical images
    :class:`~pathml.ml.hovernet.HoVerNet` [HoVerNet]_  A model for nucleus segmentation and classification in H&E images
    ===================================== ============ =============

You can also use models from fantastic resources such as
`torchvision.models <https://pytorch.org/docs/stable/torchvision/models.html>`_ and
`pytorch-image-models (timm) <https://rwightman.github.io/pytorch-image-models/>`_.

References
----------

..  [Unet] Ronneberger, O., Fischer, P. and Brox, T., 2015, October.
    U-net: Convolutional networks for biomedical image segmentation.
    In International Conference on Medical image computing and computer-assisted intervention (pp. 234-241). Springer, Cham.
..  [HoVerNet] Graham, S., Vu, Q.D., Raza, S.E.A., Azam, A., Tsang, Y.W., Kwak, J.T. and Rajpoot, N., 2019.
    Hover-Net: Simultaneous segmentation and classification of nuclei in multi-tissue histology images.
    Medical Image Analysis, 58, p.101563.
