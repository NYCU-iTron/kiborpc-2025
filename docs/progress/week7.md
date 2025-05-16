# Week 7 (05/12-05/18)

## Meeting Record

## Progress

- Release `v1.2.0`.
- Collect simulation data for calculating the statistics of the `v1.1.0` in normal and very hard mode.
- Add `vm/` and files inside to train in the virtual machine on Google Could Platform (with Nvidia T4 GPU).
- Played with V100 GPU on GCP, found that it is super expensive but 2x faster than T4.
- Train a `s-model` with 15000 images.

## Next

- Implement [WBF](https://medium.com/analytics-vidhya/weighted-boxes-fusion-86fad2c6be16) to emsemble the results of the models.
- Implement `Navigator::navigateToAreaBackup()` to navigate to the backup area.
- Train model with manual labeled data.
- Collect simulation data of `v1.0.0` and `v1.2.0`.
