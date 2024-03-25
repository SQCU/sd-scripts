1:add --timestep_noise_compensation to your training command. SDXL is supported.
2:make sure to use lycoris-locon with the 'full-lin' preset and at least 32 alpha and 32 dim.
if this code does what it is supposed to, you are seriously messing with the way your model represents training data.
think critically about whether you can refuse to train most of the modules, e.g. attn-mlp training.
training seems to benefit from repeats*epochs>=30.
3:see what happens :)
