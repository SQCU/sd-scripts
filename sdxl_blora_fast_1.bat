set NAME="tinysixblora"
set REVISION="2"
set PROJECT="%NAME%_%REVISION%"
set RANK=1024
call ./venv/Scripts/activate

accelerate launch --num_cpu_threads_per_process 8 sdxl_train_network.py ^
	--pretrained_model_name_or_path="F:/dox/ai/stable-diffusion-1.5/models/Stable-diffusion/ponyDiffusionV6XL_v6.safetensors" ^
	--dataset_config="F:/dox/ai/training-data/inbound/train_class/tiny_bundled_class/tinysixblora.toml" ^
	--output_dir="F:/dox/ai/stable-diffusion-1.5/models/Lora" ^
	--output_name="%PROJECT%" ^
	--network_args="preset=F:/dox/ai/sd-scripts/kohya_ss/sd-scripts/lycoris_presets/blora_content_layout_style.toml" ^
	--resolution="1024,1024" ^
	--save_model_as="safetensors" ^
	--network_module="lycoris.kohya" ^
	--max_train_steps=3000 ^
	--save_every_n_steps=200  ^
	--save_every_n_epochs=0 ^
	--network_dim=%RANK% ^
	--network_alpha=%RANK% ^
	--gradient_checkpointing ^
	--persistent_data_loader_workers ^
	--enable_bucket ^
	--random_crop ^
	--bucket_reso_steps=32 ^
	--min_bucket_reso=512 ^
	--xformers ^
	--mixed_precision="bf16" ^
	--caption_extension=".txt" ^
	--lr_scheduler="cosine" ^
	--lr_warmup_steps=0 ^
	--network_train_unet_only ^
	--prior_loss_weight=0 ^
	--optimizer_type="prodigy" ^
	--max_grad_norm=1.0 ^
	--learning_rate=1.0 ^
	--seed=0 ^
	--optimizer_args weight_decay=1e-04 betas=(0.9,0.999) eps=1e-08 decouple=True use_bias_correction=False safeguard_warmup=True beta3=None

pause