fn main() {
    let specs = hematite_benchmarks::spec::kernel_specs();
    let spec = specs.iter().find(|s| s.name.contains("64x1x1x64")).expect("conv1x1 spec");
    let lay = hematite_benchmarks::spec::layout(spec);
    let mut arena = vec![0u8; lay.input_len + lay.weights_len + lay.bias_len*4 + lay.output_len + 128];
    let mut bufs = hematite_benchmarks::spec::carve_into(&mut arena, &lay).unwrap();
    hematite_benchmarks::spec::fill_pattern(&mut bufs);
    let p = match spec.params { hematite_benchmarks::spec::KernelParams::Conv(p) => p, _ => panic!() };
    let mut scratch = [0u8; 0];
    hematite_ref::conv::conv2d(bufs.input, bufs.weights, bufs.bias, p, bufs.output, &mut scratch).unwrap();
    let mut h: u32 = 2166136261;
    for &b in bufs.output.iter() { h ^= b as u32; h = h.wrapping_mul(16777619); }
    println!("ref fnv1a = 0x{:08x}", h);
    print!("ref out[0..64] =");
    for (i,&b) in bufs.output.iter().enumerate() { if i%8==0 {print!("\n  ");} print!("{:02x} ", b as u8); }
    println!();
}
