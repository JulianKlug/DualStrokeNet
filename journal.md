# Testing Unet Architecture and Segmentation loss functions

|Start Date|End Date  |
|----------|----------|
|2019-12-01|2019-12-31|

## Description

Testing and implementing UNet architecture and loss functions for segmentation task using a simpler dataset (Carvana car segmentation)


## Delivrables

- [x] Pytorch code with the unet architecture and the loss functions
  - `/metrics.py` contains several loss functions

## Interpretation

On the carvana dataset, a quick ranking of loss functions (30 epochs):

1. Focal Tversky (max test dice: 0.976)
2. Surface + Focal Tversky
3. Tversky 
4. Dice + binary cross entropy 

Below, training curves for the Focal Tversky loss


![Focal Tversky Loss on Carvana dataset](/static/focal_Tversky_carvana_training_.png?ra=true "Focal Tversky Loss on Carvana dataset")

## Conclusion

- Unet is correctly implemented
- Use Focal Tversky as loss function
- Surface loss should be tried out on the real dataset for late stage training