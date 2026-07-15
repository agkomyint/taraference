fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = cudarc::driver::CudaContext::new(0)?;
    println!("device ok: {:?}", ctx.ordinal());
    let stream = ctx.default_stream();
    let mut a = stream.alloc_zeros::<f32>(4)?;
    stream.htod_copy(&mut a, &[1.0f32, 2.0, 3.0, 4.0])?;
    let host = stream.dtoh_sync_copy(&a)?;
    println!("roundtrip {:?}", host);
    Ok(())
}
