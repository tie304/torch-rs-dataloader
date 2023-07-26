


//make sure to use pytoch 2.0
//export LD_LIBRARY_PATH=/path/to/your/libtorch/lib


use anyhow::{Result, Ok};



mod mnist;
mod cats_dogs;
mod dataloader;

const IMAGE_DIM: i64 = 784;
const HIDDEN_NODES: i64 = 128;
const LABELS: i64 = 10;
const EPOCHS: i32 = 10;


fn main() -> Result<()> {
   // mnist::run(EPOCHS)?;
   cats_dogs::run()?;
    Ok(())
}