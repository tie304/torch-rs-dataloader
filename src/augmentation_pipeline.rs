use std::sync::Arc;

use image::{DynamicImage, GenericImageView, GrayImage, RgbImage};
use tch::Tensor;

use crate::utils::{rgb_image_to_tensor, tensor_to_rgb_image};

type Augmentation = Arc<dyn Fn(&Tensor) -> Tensor + Send + Sync>;

#[derive(Clone)]
pub struct AugmentationPipeline {
    pub augmentations: Vec<Augmentation>,
}

impl AugmentationPipeline {
    pub fn new() -> Self {
        Self {
            augmentations: Vec::new(),
        }
    }
    pub fn add_augmentation(&mut self, augmentation: Augmentation) {
        self.augmentations.push(augmentation);
    }
    pub fn augment(self, img: &Tensor) -> Vec<Tensor> {
        if self.augmentations.is_empty() {
            panic!("Augmentations Defined But Empty");
        }
        let mut augmentations = Vec::new();
        for a in self.augmentations {
            let aug = a(img);
            augmentations.push(aug)
        }

        augmentations
    }
}

pub fn blur(t: &Tensor, sigma: f32) -> Tensor {
    let image = tensor_to_rgb_image(t);
    let image: RgbImage = image::imageops::blur(&image, sigma);
    rgb_image_to_tensor(image)
}

pub fn flip(t: &Tensor) -> Tensor {
    let image = tensor_to_rgb_image(t);
    let image: RgbImage = image::imageops::flip_vertical(&image);
    rgb_image_to_tensor(image)
}
