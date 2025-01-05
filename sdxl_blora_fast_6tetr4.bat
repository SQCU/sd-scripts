set NAME="flr2lrg_tetrad_i"
set REVISION="4"
set PROJECT="%NAME%_%REVISION%"
set RANK=1024
call ../venv/Scripts/activate

accelerate launch --num_cpu_threads_per_process 8 sdxl_train_network.py ^
	--pretrained_model_name_or_path="F:/dox/ai/stable-diffusion-1.5/models/Stable-diffusion/ponyDiffusionV6XL_v6.safetensors" ^
	--dataset_config="F:/dox/ai/training-data/inbound/train_class/shift_study/errata/shift_study_tetrad_i.toml" ^
	--output_dir="F:/dox/ai/stable-diffusion-1.5/models/Lora" ^
	--output_name=%PROJECT% ^
	--network_args="preset=F:/dox/ai/sd-scripts/kohya_ss/sd-scripts/lycoris_presets/blora_content_layout_style.toml" ^
	--resolution="1024,1024" ^
	--save_model_as="safetensors" ^
	--network_module="lycoris.kohya" ^
	--max_train_steps=3000 ^
	--save_every_n_steps=600  ^
	--network_dim=%RANK% ^
	--network_alpha=%RANK% ^
	--gradient_checkpointing ^
	--gradient_accumulation_steps=4 ^
	--persistent_data_loader_workers ^
	--enable_bucket ^
	--bucket_reso_steps=32 ^
	--min_bucket_reso=512 ^
	--max_bucket_reso=2048 ^
	--xformers ^
	--mixed_precision="bf16" ^
	--full_bf16 ^
	--caption_extension=".txt" ^
	--lr_scheduler="constant" ^
	--lr_warmup_steps=0 ^
	--network_train_unet_only ^
	--prior_loss_weight=0 ^
	--optimizer_type="AdamW8bit" ^
	--max_grad_norm=10.0 ^
	--learning_rate=2e-05 ^
	--seed=0 ^
	--optimizer_args weight_decay=1e-03 betas=(0.9,0.999)

pause