use anyhow::Result;
use std::sync::{Arc, Mutex};
use tch::Tensor;

use std::time::Instant;
use tch::nn::ModuleT;
use tch::{nn, nn::OptimizerConfig, Device};

use crate::augmentation_pipeline::{blur, flip, AugmentationPipeline};
use crate::dataloader::{DataLoader, FsDataLoader, FsLabelDelimiter};
use crate::transform_pipeline::{
    greyscale, normalize, resize, to_cuda, to_float, TransformPipeline,
};

#[derive(Debug)]
pub struct Net {
    conv1: nn::Conv2D,
    conv2: nn::Conv2D,
    conv3: nn::Conv2D,
    fc1: nn::Linear,
    fc2: nn::Linear,
    fc3: nn::Linear,
}

impl Net {
    pub fn new(vs: &nn::Path) -> Net {
        let conv1 = nn::conv2d(
            vs,
            3,
            16,
            5,
            nn::ConvConfig {
                stride: 2,
                padding: 1,
                ..Default::default()
            },
        );
        let conv2 = nn::conv2d(
            vs,
            16,
            32,
            5,
            nn::ConvConfig {
                stride: 2,
                padding: 1,
                ..Default::default()
            },
        );
        let conv3 = nn::conv2d(
            vs,
            32,
            64,
            3,
            nn::ConvConfig {
                padding: 1,
                ..Default::default()
            },
        );
        let fc1 = nn::linear(vs, 64 * 7 * 7, 500, Default::default());
        let fc2 = nn::linear(vs, 500, 50, Default::default());
        let fc3 = nn::linear(vs, 50, 2, Default::default());

        Net {
            conv1,
            conv2,
            conv3,
            fc1,
            fc2,
            fc3,
        }
    }
}

impl nn::ModuleT for Net {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let convs = xs
            .apply(&self.conv1)
            .relu()
            .max_pool2d(2, 2, 1, 2, true)
            .apply(&self.conv2)
            .relu()
            .max_pool2d(2, 2, 1, 2, true)
            .apply(&self.conv3)
            .relu()
            .max_pool2d(2, 2, 0, 1, true);

        convs
            .view([-1, 64 * 7 * 7])
            .apply(&self.fc1)
            .relu()
            .apply(&self.fc2)
            .relu()
            .apply(&self.fc3)
    }
}

pub fn run() -> Result<()> {
    let start = Instant::now();

    let mut dels: Vec<FsLabelDelimiter> = vec![];
    let dog = FsLabelDelimiter::new(0, "dog".to_string(), "dog.".to_string());
    let cat = FsLabelDelimiter::new(1, "cat".to_string(), "cat.".to_string());

    dels.push(dog);
    dels.push(cat);

    let mut augmentation_pipeline = AugmentationPipeline::new();

    augmentation_pipeline.add_augmentation(Arc::new(move |img| -> Tensor { blur(&img, 3.0) }));
    augmentation_pipeline.add_augmentation(Arc::new(flip));

    let mut tsp = TransformPipeline::new();
    let resize_width = 224;
    let resize_height = 224;
    tsp.add_transform(Arc::new(move |img: Tensor| -> Tensor {
        // Call resize with parameters here
        resize(img, resize_width, resize_height)
    }));

    //tsp.add_transform(Arc::new(greyscale));
    tsp.add_transform(Arc::new(to_float));

    tsp.add_transform(Arc::new(normalize));

    let mut train_loader = Arc::new(Mutex::new(FsDataLoader::new(
        "./train",
        100,
        dels.clone(),
        tsp.clone(),
        None,
    )));
    let mut test_loader = Arc::new(Mutex::new(FsDataLoader::new(
        "./test",
        10,
        dels.clone(),
        tsp.clone(),
        None,
    )));

    let vs = nn::VarStore::new(Device::Cpu);
    let net = Net::new(&vs.root());

    let mut opt = nn::Adam::default().build(&vs, 1e-3)?;

    for _epoch in 0..10 {
        let mut train_loader_guard = train_loader.lock().unwrap();
        while let Some(batch) = train_loader_guard.next() {
            let forward = net.forward_t(&batch.images, true);
            println!("NETWORK BATCH: {:?}", batch.index);
            let loss = forward.cross_entropy_for_logits(&batch.labels);
            opt.zero_grad();
            opt.backward_step(&loss);
        }

        let mut test_loader_guard = test_loader.lock().unwrap();
        let mut correct = 0;
        let mut total = 0;
        while let Some(batch) = test_loader_guard.next() {
            println!("TEST BATCH: {:?}", batch.index);
            let outputs = net.forward_t(&batch.images, false);
            let predicted = outputs.argmax(-1, false);
            total += batch.labels.size()[0];
            correct += predicted
                .iter::<i64>()?
                .zip(batch.labels.iter::<i64>().unwrap())
                .filter(|&(pred, label)| pred == label)
                .count();
        }
        println!(
            "Accuracy of the network on the test images: {}",
            100.0 * correct as f64 / total as f64
        );
    }

    let duration = start.elapsed();
    println!("Time elapsed is: {:?}", duration);

    Ok(())
}
