//make sure to use pytorch 2.0
//export LD_LIBRARY_PATH=/path/to/your/libtorch/lib

use anyhow::{Ok, Result};

mod augmentation_pipeline;
mod cats_dogs;
mod dataloader;
mod transform_pipeline;
mod utils;

fn main() -> Result<()> {
    cats_dogs::run()?;
    Ok(())
}
