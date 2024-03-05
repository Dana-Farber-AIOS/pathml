Models
======

``PathML`` comes with model architectures ready to use out of the box.

.. table::
    :widths: 20, 20, 60

    ============================================ ============ =============
    Model                                        Reference    Description
    ============================================ ============ =============
    :class:`~pathml.ml.models.hovernet.HoVerNet` [HoVerNet]_  A model for nucleus segmentation and classification in H&E images
    :class:`~pathml.ml.models.hactnet.HACTNet`   [HACTNet]_   A graph neural network (GNN) for cancer subtyping
    ============================================ ============ =============

You can also use models from fantastic resources such as
`torchvision.models <https://pytorch.org/docs/stable/torchvision/models.html>`_ and
`pytorch-image-models (timm) <https://rwightman.github.io/pytorch-image-models/>`_.

References
----------

..  [HoVerNet] Graham, S., Vu, Q.D., Raza, S.E.A., Azam, A., Tsang, Y.W., Kwak, J.T. and Rajpoot, N., 2019.
    Hover-Net: Simultaneous segmentation and classification of nuclei in multi-tissue histology images.
    Medical Image Analysis, 58, p.101563.
..  [HACTNet] Pati, P., Jaume, G., Foncubierta-Rodriguez, A., Feroce, F., Anniciello, A.M., Scognamiglio, G., Brancati, N., Fiche, M., Dubruc, E., Riccio, D. and Di Bonito, M., 2022. 
    Hierarchical graph representations in digital pathology. 
    Medical image analysis, 75, p.102264.