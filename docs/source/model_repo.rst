Model Repository
================

The PathML Model Repository enables sharing of pretrained machine learning models.

After training a model, the parameters can be written to disk and the file can be uploaded to a central repository.
From the repository, other users of PathML are able to download the model weights, and run the model locally.

This has several benefits for the computational pathology research community:

- Facilitates collaboration
- Enables federated learning at multiple institutions without data ever leaving premises
- Reduce computation time by using pretrained models
- Helps ensure reproducibility of methods
- Allows for independent benchmarking of model performances.
- Promotes open science

[update this section when we have settled on more details of implementation, have working examples]

Using a pretrained model from the Model Repository
--------------------------------------------------

How to download a model from the repository, run on local images, and see model metadata

[code snippet]

Fine tuning a pretrained model on domain specific data
------------------------------------------------------

How to download a model from the repository and fine tune.

[example segmentation model]

Contributing a pretrained model to the Model Repository
-------------------------------------------------------

- Why you should want to contribute (open science, reproducibility, etc)
- How to upload (exact procedures / code snippet)
- Is there a review process? Requirements for acceptance into the Model Repo
