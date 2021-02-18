Models
======

``PathML`` comes with several model architectures ready to use out of the box.

.. list-table:: Models included in PathML
   :widths: 15, 70, 15
   :header-rows: 1

   * - Model
     - Description
     - Reference
   * - :class:`~pathml.ml.unet.UNet`
     - A standard general-purpose model designed for segmentation in biomedical images.
       Architecture consists of an 4 encoder blocks followed by 4 decoder blocks.
       Skip connections propagate information from each layer of the encoder to the corresponding layer in
       the decoder.
     - [Unet]_
   * - :class:`~pathml.ml.hovernet.HoVerNet`
     - A model for simultaneous nucleus segmentation and classification in H&E images.
       Architecture consists of a single encoder with three separate decoder branches: one to perform binary
       classification of nuclear pixels (NP), one to compute horizontal and vertical nucleus maps (HV), and one which
       is used in the classification setting to perform classification of nuclear pixels (NC).
     - [HoVerNet]_

You can also use models from `torchvision.models <https://pytorch.org/docs/stable/torchvision/models.html>`_, or create your own!

References
----------

..  [Unet] Ronneberger, O., Fischer, P. and Brox, T., 2015, October.
    U-net: Convolutional networks for biomedical image segmentation.
    In International Conference on Medical image computing and computer-assisted intervention (pp. 234-241). Springer, Cham.
..  [HoVerNet] Graham, S., Vu, Q.D., Raza, S.E.A., Azam, A., Tsang, Y.W., Kwak, J.T. and Rajpoot, N., 2019.
    Hover-Net: Simultaneous segmentation and classification of nuclei in multi-tissue histology images.
    Medical Image Analysis, 58, p.101563.