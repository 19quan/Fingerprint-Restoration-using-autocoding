autoencoder_v4.py is the final working version of the project's convolutional autoencoder. This file when run will read the dataset and train a model to be saved.

autoencoder_predict uses trained models to reconstruct the fingerprint. It uses the altered fingerprints to reconstruct. It will automatically save the
reconstructed fingerprints to 'results/reconstructed_fingerprints'.

ssim.py will give a structural similarity index score, change the original_path as needed to compare.

I've included a small sample dataset (in dataset/Altered/altered_fingerprint_specific) that specific_trained.h5 is already trained on.

I've also included a small sample dataset that consists of original fingerprints, so you can run autoencoder_predict.py and ssim.py without having to modify anything.

autoencoder_predict.py will attempt to reconstruct the fingerprint using that small sample dataset.

The autoencoders in the old_autoencoders are previous versions, they may not work as most of them come from the beginning of the project.