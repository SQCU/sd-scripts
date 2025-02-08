# training with captions

import argparse
import math
import os
from multiprocessing import Value
from typing import List
import toml

from tqdm import tqdm

import torch
from library.device_utils import init_ipex, clean_memory_on_device

from torchjd import backward
from torchjd.aggregation import UPGrad

init_ipex()

from accelerate.utils import set_seed, FP8RecipeKwargs
from diffusers import DDPMScheduler
from library import deepspeed_utils #sdxl_model_util
from library import sdxl_model_util_lambdanet

import library.train_util as train_util


from library.utils import setup_logging, add_logging_arguments

setup_logging()
import logging

logger = logging.getLogger(__name__)

import library.config_util as config_util
#import library.sdxl_train_util as sdxl_train_util
import library.sdxl_train_util_lambdanet as sdxl_train_util


from library.config_util import (
    ConfigSanitizer,
    BlueprintGenerator,
)
import library.custom_train_functions as custom_train_functions
from library.custom_train_functions import (
    apply_snr_weight,
    prepare_scheduler_for_custom_training,
    scale_v_prediction_loss_like_noise_prediction,
    add_v_prediction_like_loss,
    apply_debiased_estimation,
    apply_masked_loss,
    apply_karras_edm_weighting,
    apply_salimans_vpred_weighting,
    apply_sigmoid_k_weighting,
    nullify_implicit_v_lossweight,
    slam_implicit_v_lossweight
)
#from library.sdxl_original_unet import SdxlUNet2DConditionModel
from library.sdxl_original_unet_skiplambda import SdxlUNet2DConditionModel

UNET_NUM_BLOCKS_FOR_BLOCK_LR = 23


def get_block_params_to_optimize(unet: SdxlUNet2DConditionModel, block_lrs: List[float]) -> List[dict]:
    block_params = [[] for _ in range(len(block_lrs))]

    for i, (name, param) in enumerate(unet.named_parameters()):
        if name.startswith("time_embed.") or name.startswith("label_emb."):
            block_index = 0  # 0
        elif name.startswith("input_blocks."):  # 1-9
            block_index = 1 + int(name.split(".")[1])
        elif name.startswith("middle_block."):  # 10-12
            block_index = 10 + int(name.split(".")[1])
        elif name.startswith("output_blocks."):  # 13-21
            block_index = 13 + int(name.split(".")[1])
        elif name.startswith("out."):  # 22
            block_index = 22
        else:
            raise ValueError(f"unexpected parameter name: {name}")

        block_params[block_index].append(param)

    params_to_optimize = []
    for i, params in enumerate(block_params):
        if block_lrs[i] == 0:  # 0のときは学習しない do not optimize when lr is 0
            continue
        params_to_optimize.append({"params": params, "lr": block_lrs[i]})

    return params_to_optimize

def get_norm_params(unet: SdxlUNet2DConditionModel):     #you would never believe what they do with they adaptive optimussy around here 
    prosaic_params = []
    norming_params = []
    def sdxlunetoriginal_normyoink(name):
        check = "norm"
        return check in name.lower()
        #return name.startswith("input_blocks.norm") or name.startswith("middle_block.norm") or name.startswith("output_blocks.norm") or name.startswith("out.Group")
        #chotto a mate

    for i, (name, param) in enumerate(unet.named_parameters()):
        if sdxlunetoriginal_normyoink(name):
            norming_params.append(param)
        else:
            prosaic_params.append(param)
    # returns two lists, the gradient_clipping prosaics
    # and the no-weight-decay non_clipped normings

    return prosaic_params, norming_params


def get_named_params(unet: SdxlUNet2DConditionModel , namestring: str):
    prosaic_params = []
    perspicacious_params = []
    def unet_nameyoink(name, checkstring):
        #check = namestring.lower()
        check = namestring#.lower()
        return check in name#.lower()
        
    for i, (name, param) in enumerate(unet.named_parameters()):
        if unet_nameyoink(name, checkstring=namestring):
            perspicacious_params.append(param)
        else:
            prosaic_params.append(param)

    return prosaic_params, perspicacious_params

def bias_yoinkems(unet, affinenormbiases=[]):
    usd = unet.state_dict()
    bvalues = {affinenormbias:usd[affinenormbias] for affinenormbias in affinenormbiases}
    #enumerated = enumerate([usd[str(bvalue)] for bvalue in bvalues.keys()])
    tiniest = min(torch.min(bvalues[bkey]) for bkey in bvalues.keys())
    biggest = max(torch.max(bvalues[bkey]) for bkey in bvalues.keys())
    #return min(enumerate([bvalue for bvalue in unet[str(affinenormbiases+".data")]]))
    return tiniest, biggest

def beta_clampy(term_for_clampy, beta, mink=None, maxk=None):
    return torch.clamp(term_for_clampy,min=mink, max=maxk)*beta + (1-beta)*term_for_clampy
def nonbeta_clampy(term_for_clampy, mink=None, maxk=None):
    #return torch.clamp_(term_for_clampy,min=mink, max=maxk)
    torch.clamp_(term_for_clampy,min=mink, max=maxk)

def cleanup(unet:SdxlUNet2DConditionModel, affinenormbiases=[], affinenormweights=[], learnedlambdas=[], incremental_abolish=None):
    # we want norm123.bias.data
    usd = unet.state_dict()
    def continuous_abolisher(unet:SdxlUNet2DConditionModel, affinenormbiases=[], affinenormweights=[], incremental_abolish=None):
        for bias in affinenormbiases:
            if incremental_abolish:
                usd[str(bias)] = beta_clampy(usd[str(bias)], incremental_abolish[0], mink=0, maxk=2)
            else:
                nonbeta_clampy(usd[str(bias)], mink=0, maxk=2)

        for weight in affinenormweights:
            wei = usd[str(weight)]
            weitarget = 1 #"""for a*gamma equals a nonoperation"""
            local_epsilon = 1e-08 # look... i...
            if incremental_abolish:
                #can't use unless inc_abolish exists!
                inc_alpha = 1 - incremental_abolish[0] 
                wei = beta_clampy(wei, incremental_abolish[0], mink=weitarget-inc_alpha, maxk=weitarget+inc_alpha)
            else: 
                nonbeta_clampy(wei, mink=weitarget-local_epsilon, maxk=weitarget+local_epsilon)
        
        if incremental_abolish:
            incremental_abolish[0] = incremental_abolish[0]+0.01    #its so funny having to write an iterator in pydiom.
            if incremental_abolish[0] >= 1:
                incremental_abolish.clear()

    def lambda_clampbda(unet:SdxlUNet2DConditionModel, learnedlambdas=[]):
        for lambdas in learnedlambdas:
            #unet[str(lambdas+".data")] = torch.clamp_(unet[str(lambdas+".data")], min=1e-2, max=2.0)
            lambdas.data = torch.clamp(lambdas.data, min=1e-2, max=2.0)     #so much exciting pedantry about leaf nodes!
            #migrating clamp action here because of suspicions about forwards pass calculation time.

    continuous_abolisher(unet, affinenormbiases=affinenormbiases, affinenormweights=affinenormweights, incremental_abolish=incremental_abolish)
    lambda_clampbda(unet, learnedlambdas=learnedlambdas)

def append_block_lr_to_logs(block_lrs, logs, lr_scheduler, optimizer_type):
    names = []
    block_index = 0
    while block_index < UNET_NUM_BLOCKS_FOR_BLOCK_LR + 2:
        if block_index < UNET_NUM_BLOCKS_FOR_BLOCK_LR:
            if block_lrs[block_index] == 0:
                block_index += 1
                continue
            names.append(f"block{block_index}")
        elif block_index == UNET_NUM_BLOCKS_FOR_BLOCK_LR:
            names.append("text_encoder1")
        elif block_index == UNET_NUM_BLOCKS_FOR_BLOCK_LR + 1:
            names.append("text_encoder2")

        block_index += 1

    train_util.append_lr_to_logs_with_names(logs, lr_scheduler, optimizer_type, names)


def train(args):
    train_util.verify_training_args(args)
    train_util.prepare_dataset_args(args, True)
    sdxl_train_util.verify_sdxl_training_args(args)
    deepspeed_utils.prepare_deepspeed_args(args)
    setup_logging(args, reset=True)

    norm_params = None
    clippables = None
    global_optim_man = None

    skipweight_params = []
    affinenormbiases= []
    affinenormweights= []
    incremental_abolish_ls = []

    assert (
        not args.weighted_captions
    ), "weighted_captions is not supported currently / weighted_captionsは現在サポートされていません"
    assert (
        not args.train_text_encoder or not args.cache_text_encoder_outputs
    ), "cache_text_encoder_outputs is not supported when training text encoder / text encoderを学習するときはcache_text_encoder_outputsはサポートされていません"

    if args.block_lr:
        block_lrs = [float(lr) for lr in args.block_lr.split(",")]
        assert (
            len(block_lrs) == UNET_NUM_BLOCKS_FOR_BLOCK_LR
        ), f"block_lr must have {UNET_NUM_BLOCKS_FOR_BLOCK_LR} values / block_lrは{UNET_NUM_BLOCKS_FOR_BLOCK_LR}個の値を指定してください"
    else:
        block_lrs = None

    cache_latents = args.cache_latents
    use_dreambooth_method = args.in_json is None

    if args.seed is not None:
        set_seed(args.seed)  # 乱数系列を初期化する

    tokenizer1, tokenizer2 = sdxl_train_util.load_tokenizers(args)

    # データセットを準備する
    if args.dataset_class is None:
        blueprint_generator = BlueprintGenerator(ConfigSanitizer(True, True, args.masked_loss, True))
        if args.dataset_config is not None:
            logger.info(f"Load dataset config from {args.dataset_config}")
            user_config = config_util.load_user_config(args.dataset_config)
            ignored = ["train_data_dir", "in_json"]
            if any(getattr(args, attr) is not None for attr in ignored):
                logger.warning(
                    "ignore following options because config file is found: {0} / 設定ファイルが利用されるため以下のオプションは無視されます: {0}".format(
                        ", ".join(ignored)
                    )
                )
        else:
            if use_dreambooth_method:
                logger.info("Using DreamBooth method.")
                user_config = {
                    "datasets": [
                        {
                            "subsets": config_util.generate_dreambooth_subsets_config_by_subdirs(
                                args.train_data_dir, args.reg_data_dir
                            )
                        }
                    ]
                }
            else:
                logger.info("Training with captions.")
                user_config = {
                    "datasets": [
                        {
                            "subsets": [
                                {
                                    "image_dir": args.train_data_dir,
                                    "metadata_file": args.in_json,
                                }
                            ]
                        }
                    ]
                }

        blueprint = blueprint_generator.generate(user_config, args, tokenizer=[tokenizer1, tokenizer2])
        train_dataset_group = config_util.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    else:
        train_dataset_group = train_util.load_arbitrary_dataset(args, [tokenizer1, tokenizer2])

    current_epoch = Value("i", 0)
    current_step = Value("i", 0)
    ds_for_collator = train_dataset_group if args.max_data_loader_n_workers == 0 else None
    collator = train_util.collator_class(current_epoch, current_step, ds_for_collator)

    train_dataset_group.verify_bucket_reso_steps(32)

    if args.debug_dataset:
        train_util.debug_dataset(train_dataset_group, True)
        return
    if len(train_dataset_group) == 0:
        logger.error(
            "No data found. Please verify the metadata file and train_data_dir option. / 画像がありません。メタデータおよびtrain_data_dirオプションを確認してください。"
        )
        return

    if cache_latents:
        assert (
            train_dataset_group.is_latent_cacheable()
        ), "when caching latents, either color_aug or random_crop cannot be used / latentをキャッシュするときはcolor_augとrandom_cropは使えません"

    if args.cache_text_encoder_outputs:
        assert (
            train_dataset_group.is_text_encoder_output_cacheable()
        ), "when caching text encoder output, either caption_dropout_rate, shuffle_caption, token_warmup_step or caption_tag_dropout_rate cannot be used / text encoderの出力をキャッシュするときはcaption_dropout_rate, shuffle_caption, token_warmup_step, caption_tag_dropout_rateは使えません"

    # acceleratorを準備する
    logger.info("prepare accelerator")
    accelerator = train_util.prepare_accelerator(args)

    # mixed precisionに対応した型を用意しておき適宜castする
    weight_dtype, save_dtype = train_util.prepare_dtype(args)
    vae_dtype = torch.float32 if args.no_half_vae else weight_dtype

    # モデルを読み込む
    # unet loaded here!!!
    # add keys to initdict{} here to pipe configuration to model init
    initdict={}
    initdict.update({"learnable_lambdas_level":args.learnable_lambdas_level})
    initdict.update({"swiglu_switcharoo":args.swiglu_switcharoo})
    (
        load_stable_diffusion_format,
        text_encoder1,
        text_encoder2,
        vae,
        unet,
        logit_scale,
        ckpt_info,
    ) = sdxl_train_util.load_target_model(args, accelerator, "sdxl", weight_dtype, initdict=initdict)
    # logit_scale = logit_scale.to(accelerator.device, dtype=weight_dtype)

    if args.auxiliary_TE_loss_anthropic_style_baybee:
        import copy
        #oh yeah we need this haha
        if args.auxiliary_TE_from_path is not None:
            quasi_args = copy.deepcopy(args) #don't trust the snake here
            #the sd-scripts imports are serpentine but here's what they do:
            #take args.pretrained_model_name_or_path from sdxl_train_util.load_target_model(...)
            #use that to pick the path for all further path requiring actions.
            #
            # here we overwrite a copy of the args with the path of the sdxl model we're using for a KL divergence measure
            quasi_args.pretrained_model_name_or_path = args.auxiliary_TE_from_path
            ( shim_format, auxiliary_text_encoder1, auxiliary_text_encoder2, shim_vae, shim_unet, shim_logit_scale, shim_ckpt_info, 
            ) = sdxl_train_util.load_target_model(args, accelerator, "sdxl", weight_dtype, initdict=initdict)
            del shim_format, shim_vae, shim_unet, shim_logit_scale, shim_ckpt_info
        elif args.auxiliary_TE_from_same: #i do not recommend this yet it should be offered 
            auxiliary_text_encoder1 = copy.deepcopy(text_encoder1)
            auxiliary_text_encoder2 = copy.deepcopy(text_encoder2)
        else:
            accelerator.print("critical show-stopping configuration error. do not use auxiliary TE loss without examining source more closely.")
        auxiliary_text_encoder1.requires_grad_(False)
        auxiliary_text_encoder2.requires_grad_(False)
        auxiliary_text_encoder1.eval()
        auxiliary_text_encoder2.eval()

    # verify load/save model formats
    if load_stable_diffusion_format:
        src_stable_diffusion_ckpt = args.pretrained_model_name_or_path
        src_diffusers_model_path = None
    else:
        src_stable_diffusion_ckpt = None
        src_diffusers_model_path = args.pretrained_model_name_or_path

    if args.save_model_as is None:
        save_stable_diffusion_format = load_stable_diffusion_format
        use_safetensors = args.use_safetensors
    else:
        save_stable_diffusion_format = args.save_model_as.lower() == "ckpt" or args.save_model_as.lower() == "safetensors"
        use_safetensors = args.use_safetensors or ("safetensors" in args.save_model_as.lower())
        # assert save_stable_diffusion_format, "save_model_as must be ckpt or safetensors / save_model_asはckptかsafetensorsである必要があります"

    # Diffusers版のxformers使用フラグを設定する関数
    def set_diffusers_xformers_flag(model, valid):
        def fn_recursive_set_mem_eff(module: torch.nn.Module):
            if hasattr(module, "set_use_memory_efficient_attention_xformers"):
                module.set_use_memory_efficient_attention_xformers(valid)

            for child in module.children():
                fn_recursive_set_mem_eff(child)

        fn_recursive_set_mem_eff(model)

    # モデルに xformers とか memory efficient attention を組み込む
    if args.diffusers_xformers:
        # もうU-Netを独自にしたので動かないけどVAEのxformersは動くはず
        accelerator.print("Use xformers by Diffusers")
        # set_diffusers_xformers_flag(unet, True)
        set_diffusers_xformers_flag(vae, True)
    else:
        # Windows版のxformersはfloatで学習できなかったりするのでxformersを使わない設定も可能にしておく必要がある
        accelerator.print("Disable Diffusers' xformers")
        train_util.replace_unet_modules(unet, args.mem_eff_attn, args.xformers, args.sdpa)
        if torch.__version__ >= "2.0.0":  # PyTorch 2.0.0 以上対応のxformersなら以下が使える
            vae.set_use_memory_efficient_attention_xformers(args.xformers)

    if args.use_laser_sdpa:
        logger.info("Enable LAZER-SDPA for U-Net")
        unet.set_use_laser_sdpa(True)

    if args.use_qknorm:
        logger.info("Enable qk-norm for U-Net, you scoundrel ;')")
        unet.set_use_qknorm(True)

    if args.use_laser_qknorm:
        logger.info("Enable laser-qk-norm for U-Net, you scoundrel ;')")
        unet.set_use_laser_qknorm(True)

    if args.use_layerqknorm:
        logger.info("Enable layer-qk-norm for U-Net, you scoundrel ;')")
        unet.set_use_layerqknorm(True)

    # 学習を準備する
    if cache_latents:
        vae.to(accelerator.device, dtype=vae_dtype)
        vae.requires_grad_(False)
        vae.eval()
        with torch.no_grad():
            train_dataset_group.cache_latents(vae, args.vae_batch_size, args.cache_latents_to_disk, accelerator.is_main_process)
        vae.to("cpu")
        clean_memory_on_device(accelerator.device)

        accelerator.wait_for_everyone()

    # 学習を準備する：モデルを適切な状態にする
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    train_unet = args.learning_rate != 0
    train_text_encoder1 = False
    train_text_encoder2 = False

    if args.train_text_encoder:
        # TODO each option for two text encoders?
        accelerator.print("enable text encoder training")
        if args.gradient_checkpointing:
            text_encoder1.gradient_checkpointing_enable()
            text_encoder2.gradient_checkpointing_enable()
        lr_te1 = args.learning_rate_te1 if args.learning_rate_te1 is not None else args.learning_rate  # 0 means not train
        lr_te2 = args.learning_rate_te2 if args.learning_rate_te2 is not None else args.learning_rate  # 0 means not train
        train_text_encoder1 = lr_te1 != 0
        train_text_encoder2 = lr_te2 != 0

        # caching one text encoder output is not supported
        if not train_text_encoder1:
            text_encoder1.to(weight_dtype)
        if not train_text_encoder2:
            text_encoder2.to(weight_dtype)
        text_encoder1.requires_grad_(train_text_encoder1)
        text_encoder2.requires_grad_(train_text_encoder2)
        text_encoder1.train(train_text_encoder1)
        text_encoder2.train(train_text_encoder2)
    else:
        text_encoder1.to(weight_dtype)
        text_encoder2.to(weight_dtype)
        text_encoder1.requires_grad_(False)
        text_encoder2.requires_grad_(False)
        text_encoder1.eval()
        text_encoder2.eval()

        # TextEncoderの出力をキャッシュする
        if args.cache_text_encoder_outputs:
            # Text Encodes are eval and no grad
            with torch.no_grad(), accelerator.autocast():
                train_dataset_group.cache_text_encoder_outputs(
                    (tokenizer1, tokenizer2),
                    (text_encoder1, text_encoder2),
                    accelerator.device,
                    None,
                    args.cache_text_encoder_outputs_to_disk,
                    accelerator.is_main_process,
                )
            accelerator.wait_for_everyone()



    if not cache_latents:
        vae.requires_grad_(False)
        vae.eval()
        vae.to(accelerator.device, dtype=vae_dtype)

    unet.requires_grad_(train_unet)
    if not train_unet:
        unet.to(accelerator.device, dtype=weight_dtype)  # because of unet is not prepared

    logger.info("disable grad on unused learnable_lambdas")
    unet.set_lambdagrad_hard_recurse(args.learnable_lambdas_level)

    training_models = []
    params_to_optimize = []
    if train_unet:
        training_models.append(unet)
        if block_lrs is None:
            params_to_optimize.append({"params": list(unet.parameters()), "lr": args.learning_rate})
        else:
            params_to_optimize.extend(get_block_params_to_optimize(unet, block_lrs))

    if args.norm_salvation:
        clippables, norm_params = get_norm_params(unet)
    clippables_beta, skipweight_params = get_named_params(unet=unet, namestring="learnedlambda") # mask=norm_params)
    if clippables:
        clippables = set(clippables) - set(clippables_beta)
        clippables = list(clippables)
    else:
        clippables = clippables_beta
    if args.outblock_bias_abolisher:
        prefix = 'output_blocks.'
        modulefilter = ('norm1', 'norm2', 'norm3')
        for name, param in unet.named_parameters():
            for mod in modulefilter:
                if mod in name:
                    if prefix in name and 'bias' in name:
                        affinenormbiases.append(name)
            if args.groupnorm_bias_abolisher_auxiliary:
                if prefix in name and 'GroupNorm' in name:
                    if 'bias' in name:
                            affinenormbiases.append(name)
    elif args.bias_abolisher:
        for name, param in unet.named_parameters():
            if 'norm1' in name or 'norm2' in name or 'norm3' in name and 'model.diffusion_model' in name:
                if 'bias' in name:
                    affinenormbiases.append(name)
            else:
                if args.groupnorm_bias_abolisher_auxiliary:
                    #if 'norm' in name or 'GroupNorm' in name:
                    if 'norm' in name or 'GroupNorm' in name and 'model.diffusion_model' in name:
                        if 'bias' in name:
                            affinenormbiases.append(name)
    if args.layernorm_simple:   # should be roughly equivalent to bias_abolisher, 
                                #but interpolate between layernorm and layernorm-simple
        modulefilter = ('norm1', 'norm2', 'norm3')
        #subparamfilter = ('bias', 'weight')
        for name, param in unet.named_parameters():
            for mod in modulefilter:
                if mod in name:
                    if 'bias' in name:
                        affinenormbiases.append(name)
                    if 'weight' in name:
                        affinenormweights.append(name)
    
    if args.incremental_abolish:
        incremental_abolish_ls.append(args.incremental_abolish)

    if train_text_encoder1:
        training_models.append(text_encoder1)
        params_to_optimize.append({"params": list(text_encoder1.parameters()), "lr": args.learning_rate_te1 or args.learning_rate})
    if train_text_encoder2:
        training_models.append(text_encoder2)
        params_to_optimize.append({"params": list(text_encoder2.parameters()), "lr": args.learning_rate_te2 or args.learning_rate})

    # calculate number of trainable parameters
    n_params = 0
    for group in params_to_optimize:
        for p in group["params"]:
            n_params += p.numel()
    n_norm_params = 0
    if args.norm_salvation:
        for np in norm_params:
            n_norm_params += np.numel()
    n_notnorm_params = 0
    if args.norm_salvation:
        for nnp in clippables:
            n_notnorm_params += nnp.numel()

    accelerator.print(f"train unet: {train_unet}, text_encoder1: {train_text_encoder1}, text_encoder2: {train_text_encoder2}")
    accelerator.print(f"number of models: {len(training_models)}")
    accelerator.print(f"number of trainable parameters: {n_params}")
    if args.norm_salvation:
        accelerator.print(f"number of split norm parameters: {n_norm_params}")
        accelerator.print(f"number of split non-norm parameters: {n_notnorm_params}\nplease be {n_params-n_norm_params}.")
    if args.bias_abolisher or args.layernorm_simple:
        accelerator.print(f"onload biasminmax:{bias_yoinkems(unet=unet,affinenormbiases=affinenormbiases)}")
    if args.layernorm_simple:
        accelerator.print(f"onload weightminmax:{bias_yoinkems(unet=unet,affinenormbiases=affinenormweights)}")

    # 学習に必要なクラスを準備する
    accelerator.print("prepare optimizer, data loader etc.")

    if args.fused_optimizer_groups:
        # fused backward pass: https://pytorch.org/tutorials/intermediate/optimizer_step_in_backward_tutorial.html
        # Instead of creating an optimizer for all parameters as in the tutorial, we create an optimizer for each group of parameters.
        # This balances memory usage and management complexity.

        # calculate total number of parameters
        n_total_params = sum(len(params["params"]) for params in params_to_optimize)
        params_per_group = math.ceil(n_total_params / args.fused_optimizer_groups)

        # split params into groups, keeping the learning rate the same for all params in a group
        # this will increase the number of groups if the learning rate is different for different params (e.g. U-Net and text encoders)
        grouped_params = []
        param_group = []
        param_group_lr = -1
        for group in params_to_optimize:
            lr = group["lr"]
            for p in group["params"]:
                # if the learning rate is different for different params, start a new group
                if lr != param_group_lr:
                    if param_group:
                        grouped_params.append({"params": param_group, "lr": param_group_lr})
                        param_group = []
                    param_group_lr = lr

                param_group.append(p)

                # if the group has enough parameters, start a new group
                if len(param_group) == params_per_group:
                    grouped_params.append({"params": param_group, "lr": param_group_lr})
                    param_group = []
                    param_group_lr = -1

        if param_group:
            grouped_params.append({"params": param_group, "lr": param_group_lr})

        # prepare optimizers for each group
        optimizers = []
        for group in grouped_params:
            _, _, optimizer = train_util.get_optimizer(args, trainable_params=[group])
            optimizers.append(optimizer)
        optimizer = optimizers[0]  # avoid error in the following code

        logger.info(f"using {len(optimizers)} optimizers for fused optimizer groups")

    else:
        _, _, optimizer = train_util.get_optimizer(args, trainable_params=params_to_optimize)
        aggregator = UPGrad()
        if args.norm_salvation: #dict notation is here for a rason. veary import raseon.
            norms_kv ={"weight_decay":args.layernorm_decay}
            if args.layernorm_adambetas is not None:
                norms_kv["betas"]=tuple(args.layernorm_adambetas)
            if args.layernorm_lr is not None:
                norms_kv["lr"]=args.layernorm_lr
            #global_optim_man=optimizer.mng
            optimizer.mng.override_config(parameters=norm_params, key_value_dict=norms_kv)
            logger.info(f"using {norms_kv} overrides to optimizer config.")
        
        lskip_kv ={"weight_decay":0}
        if args.ll_lr:
            lskip_kv.update({"lr":args.ll_lr})
        optimizer.mng.override_config(parameters=skipweight_params, key_value_dict=lskip_kv)
        logger.info(f"using {lskip_kv} overrides to skipweight optimizer config.")

    # dataloaderを準備する
    # DataLoaderのプロセス数：0 は persistent_workers が使えないので注意
    n_workers = min(args.max_data_loader_n_workers, os.cpu_count())  # cpu_count or max_data_loader_n_workers
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset_group,
        batch_size=1,
        shuffle=True,
        collate_fn=collator,
        num_workers=n_workers,
        persistent_workers=args.persistent_data_loader_workers,
    )

    # 学習ステップ数を計算する
    if args.max_train_epochs is not None:
        args.max_train_steps = args.max_train_epochs * math.ceil(
            len(train_dataloader) / accelerator.num_processes / args.gradient_accumulation_steps
        )
        accelerator.print(
            f"override steps. steps for {args.max_train_epochs} epochs is / 指定エポックまでのステップ数: {args.max_train_steps}"
        )

    # データセット側にも学習ステップを送信
    train_dataset_group.set_max_train_steps(args.max_train_steps)

    # lr schedulerを用意する
    if args.fused_optimizer_groups:
        # prepare lr schedulers for each optimizer
        lr_schedulers = [train_util.get_scheduler_fix(args, optimizer, accelerator.num_processes) for optimizer in optimizers]
        lr_scheduler = lr_schedulers[0]  # avoid error in the following code
    else:
        lr_scheduler = train_util.get_scheduler_fix(args, optimizer, accelerator.num_processes)

    # 実験的機能：勾配も含めたfp16/bf16学習を行う　モデル全体をfp16/bf16にする
    if args.full_fp16:
        assert (
            args.mixed_precision == "fp16"
        ), "full_fp16 requires mixed precision='fp16' / full_fp16を使う場合はmixed_precision='fp16'を指定してください。"
        accelerator.print("enable full fp16 training.")
        unet.to(weight_dtype)
        text_encoder1.to(weight_dtype)
        text_encoder2.to(weight_dtype)
    elif args.full_bf16:
        assert (
            args.mixed_precision == "bf16"
        ), "full_bf16 requires mixed precision='bf16' / full_bf16を使う場合はmixed_precision='bf16'を指定してください。"
        accelerator.print("enable full bf16 training.")
        unet.to(weight_dtype)
        text_encoder1.to(weight_dtype)
        text_encoder2.to(weight_dtype)
   
    if args.torch_compilealter:
        torch._inductor.config.conv_1x1_as_mm = True
        torch._inductor.config.coordinate_descent_tuning = True
        torch._inductor.config.epilogue_fusion = False
        torch._inductor.config.coordinate_descent_check_all_directions = True
    
    unet_weight_dtype = te_weight_dtype = weight_dtype
    # Experimental Feature: Put base model into fp8 to save vram
    if args.fp8_base:
        assert torch.__version__ >= "2.1.0", "fp8_base requires torch>=2.1.0 / fp8を使う場合はtorch>=2.1.0が必要です。"
        assert (
            args.mixed_precision != "no"
        ), "fp8_base requires mixed precision='fp16' or 'bf16' / fp8を使う場合はmixed_precision='fp16'または'bf16'が必要です。"
        accelerator.print("enable fp8 training.")
        unet_weight_dtype = torch.float8_e4m3fn
        te_weight_dtype = torch.float8_e4m3fn
    
    unet.to(dtype=unet_weight_dtype)

    if text_encoder1.device.type != "cpu":
        text_encoder1.to(dtype=te_weight_dtype)
        # nn.Embedding not support FP8
        text_encoder1.text_model.embeddings.to(dtype=(weight_dtype if te_weight_dtype != weight_dtype else te_weight_dtype))
    if text_encoder1.device.type != "cpu":
        text_encoder2.to(dtype=te_weight_dtype)
        # nn.Embedding not support FP8
        text_encoder2.text_model.embeddings.to(dtype=(weight_dtype if te_weight_dtype != weight_dtype else te_weight_dtype))


    # freeze last layer and final_layer_norm in te1 since we use the output of the penultimate layer
    if train_text_encoder1:
        text_encoder1.text_model.encoder.layers[-1].requires_grad_(False)
        text_encoder1.text_model.final_layer_norm.requires_grad_(False)

    #same for TE2!
    if train_text_encoder2:
        text_encoder2.text_model.encoder.layers[-1].requires_grad_(False)
        text_encoder2.text_model.final_layer_norm.requires_grad_(False)

    if args.deepspeed:
        ds_model = deepspeed_utils.prepare_deepspeed_model(
            args,
            unet=unet if train_unet else None,
            text_encoder1=text_encoder1 if train_text_encoder1 else None,
            text_encoder2=text_encoder2 if train_text_encoder2 else None,
        )
        # most of ZeRO stage uses optimizer partitioning, so we have to prepare optimizer and ds_model at the same time. # pull/1139#issuecomment-1986790007
        ds_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            ds_model, optimizer, train_dataloader, lr_scheduler
        )
        training_models = [ds_model]

    else:
        # acceleratorがなんかよろしくやってくれるらしい
        if train_unet:
            unet = accelerator.prepare(unet)
            if args.torch_compilealter:
                unet = torch.compile(unet, fullgraph=False)
        if train_text_encoder1:
            text_encoder1 = accelerator.prepare(text_encoder1)
        if train_text_encoder2:
            text_encoder2 = accelerator.prepare(text_encoder2)
        optimizer, train_dataloader, lr_scheduler = accelerator.prepare(optimizer, train_dataloader, lr_scheduler)
        if args.auxiliary_TE_loss_anthropic_style_baybee:
            if train_text_encoder1:
                auxiliary_text_encoder1.to(weight_dtype)
                auxiliary_text_encoder1 = accelerator.prepare(auxiliary_text_encoder1)
                training_models.append(auxiliary_text_encoder1)
                auxiliary_text_encoder1.to(accelerator.device)
            if train_text_encoder2:
                auxiliary_text_encoder2.to(weight_dtype)
                auxiliary_text_encoder2 = accelerator.prepare(auxiliary_text_encoder2)
                training_models.append(auxiliary_text_encoder2)
                auxiliary_text_encoder2.to(accelerator.device)
             

    # TextEncoderの出力をキャッシュするときにはCPUへ移動する
    if args.cache_text_encoder_outputs:
        # move Text Encoders for sampling images. Text Encoder doesn't work on CPU with fp16
        text_encoder1.to("cpu", dtype=torch.float32)
        text_encoder2.to("cpu", dtype=torch.float32)
        clean_memory_on_device(accelerator.device)
    else:
        # make sure Text Encoders are on GPU
        text_encoder1.to(accelerator.device)
        text_encoder2.to(accelerator.device)

    # 実験的機能：勾配も含めたfp16学習を行う　PyTorchにパッチを当ててfp16でのgrad scaleを有効にする
    if args.full_fp16:
        # During deepseed training, accelerate not handles fp16/bf16|mixed precision directly via scaler. Let deepspeed engine do.
        # -> But we think it's ok to patch accelerator even if deepspeed is enabled.
        train_util.patch_accelerator_for_fp16_training(accelerator)

    # resumeする
    train_util.resume_from_local_or_hf_if_specified(accelerator, args)

    if args.clamp_grad_value: #i shuppose --clip_grad_norm 0.0 evaluates to falsy.
        for param_group in optimizer.param_groups:
            for parameter.requires_grad in param_group["params"]:
                parameter.register_hook(lambda grad: torch.clamp(grad, -clamp_grad_value, clamp_grad_value))

    if args.fused_backward_pass:
        # use fused optimizer for backward pass: other optimizers will be supported in the future
        import library.adafactor_fused

        library.adafactor_fused.patch_adafactor_fused(optimizer)
        for param_group in optimizer.param_groups:
            for parameter in param_group["params"]:
                if parameter.requires_grad:

                    def __grad_hook(tensor: torch.Tensor, param_group=param_group):
                        if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                            accelerator.clip_grad_norm_(tensor, args.max_grad_norm)
                        optimizer.step_param(tensor, param_group)
                        tensor.grad = None

                    parameter.register_post_accumulate_grad_hook(__grad_hook)

    elif args.fused_optimizer_groups:
        # prepare for additional optimizers and lr schedulers
        for i in range(1, len(optimizers)):
            optimizers[i] = accelerator.prepare(optimizers[i])
            lr_schedulers[i] = accelerator.prepare(lr_schedulers[i])

        # counters are used to determine when to step the optimizer
        global optimizer_hooked_count
        global num_parameters_per_group
        global parameter_optimizer_map

        optimizer_hooked_count = {}
        num_parameters_per_group = [0] * len(optimizers)
        parameter_optimizer_map = {}

        for opt_idx, optimizer in enumerate(optimizers):
            for param_group in optimizer.param_groups:
                for parameter in param_group["params"]:
                    if parameter.requires_grad:

                        def optimizer_hook(parameter: torch.Tensor):
                            if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                                accelerator.clip_grad_norm_(parameter, args.max_grad_norm)

                            i = parameter_optimizer_map[parameter]
                            optimizer_hooked_count[i] += 1
                            if optimizer_hooked_count[i] == num_parameters_per_group[i]:
                                optimizers[i].step()
                                optimizers[i].zero_grad(set_to_none=True)
                                if args.bias_abolisher or args.layernorm_simple:
                                    for model in accelerator._models:
                                        cleanup(
                                            model, affinenormbiases=affinenormbiases, affinenormweights=affinenormweights, 
                                            learnedlambdas=skipweight_params, 
                                            incremental_abolish=incremental_abolish_ls)

                        parameter.register_post_accumulate_grad_hook(optimizer_hook)
                        parameter_optimizer_map[parameter] = opt_idx
                        num_parameters_per_group[opt_idx] += 1

    # epoch数を計算する
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    if (args.save_n_epoch_ratio is not None) and (args.save_n_epoch_ratio > 0):
        args.save_every_n_epochs = math.floor(num_train_epochs / args.save_n_epoch_ratio) or 1

    # 学習する
    # total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    accelerator.print("running training / 学習開始")
    accelerator.print(f"  num examples / サンプル数: {train_dataset_group.num_train_images}")
    accelerator.print(f"  num batches per epoch / 1epochのバッチ数: {len(train_dataloader)}")
    accelerator.print(f"  num epochs / epoch数: {num_train_epochs}")
    accelerator.print(
        f"  batch size per device / バッチサイズ: {', '.join([str(d.batch_size) for d in train_dataset_group.datasets])}"
    )
    # accelerator.print(
    #     f"  total train batch size (with parallel & distributed & accumulation) / 総バッチサイズ（並列学習、勾配合計含む）: {total_batch_size}"
    # )
    accelerator.print(f"  gradient accumulation steps / 勾配を合計するステップ数 = {args.gradient_accumulation_steps}")
    accelerator.print(f"  total optimization steps / 学習ステップ数: {args.max_train_steps}")

    progress_bar = tqdm(range(args.max_train_steps), smoothing=0, disable=not accelerator.is_local_main_process, desc="steps")
    global_step = 0

    if args.sigmaximum_overdrive:
        noise_scheduler = DDPMScheduler(
            beta_start=0.00085, beta_end=(0.012*math.sqrt(args.sigmaximum_overdrive/256)), beta_schedule=args.beta_schedule, num_train_timesteps=1000, clip_sample=False
        )
        #beta_parameterization_keys = ("num_train_timesteps","beta_start","beta_end","beta_schedule")
        betascale_kwargs={"num_train_timesteps":1000,"beta_start":0.00085,"beta_end":(0.012*math.sqrt(args.sigmaximum_overdrive/256)),"beta_schedule":args.beta_schedule}
    else:    
        noise_scheduler = DDPMScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule=args.beta_schedule, num_train_timesteps=1000, clip_sample=False
        )
        betascale_kwargs={}
    prepare_scheduler_for_custom_training(noise_scheduler, accelerator.device)
    if args.zero_terminal_snr:
        custom_train_functions.fix_noise_scheduler_betas_for_zero_terminal_snr(noise_scheduler)

    if accelerator.is_main_process:
        init_kwargs = {}
        if args.wandb_run_name:
            init_kwargs["wandb"] = {"name": args.wandb_run_name}
        if args.log_tracker_config is not None:
            init_kwargs = toml.load(args.log_tracker_config)
        accelerator.init_trackers(
            "finetuning" if args.log_tracker_name is None else args.log_tracker_name,
            config=train_util.get_sanitized_config_or_none(args),
            init_kwargs=init_kwargs,
        )

    # For --sample_at_first
    sdxl_train_util.sample_images(
        accelerator, args, 0, global_step, accelerator.device, vae, [tokenizer1, tokenizer2], [text_encoder1, text_encoder2], unet,
        kwargs=betascale_kwargs
    )

    loss_recorder = train_util.LossRecorder()
    aux_loss_recorder = train_util.LossRecorder()
    for epoch in range(num_train_epochs):
        accelerator.print(f"\nepoch {epoch+1}/{num_train_epochs}")
        current_epoch.value = epoch + 1

        for m in training_models:
            m.train()

        for step, batch in enumerate(train_dataloader):
            current_step.value = global_step

            if args.fused_optimizer_groups:
                optimizer_hooked_count = {i: 0 for i in range(len(optimizers))}  # reset counter for each step

            with accelerator.accumulate(*training_models):
                if "latents" in batch and batch["latents"] is not None:
                    latents = batch["latents"].to(accelerator.device).to(dtype=weight_dtype)
                else:
                    with torch.no_grad():
                        # latentに変換
                        latents = vae.encode(batch["images"].to(vae_dtype)).latent_dist.sample().to(weight_dtype)

                        # NaNが含まれていれば警告を表示し0に置き換える
                        if torch.any(torch.isnan(latents)):
                            accelerator.print("NaN found in latents, replacing with zeros")
                            latents = torch.nan_to_num(latents, 0, out=latents)
                latents = latents * sdxl_model_util_lambdanet.VAE_SCALE_FACTOR
                aux_loss = 0

                #this is when youre using the text encoders in live fire mode
                if "text_encoder_outputs1_list" not in batch or batch["text_encoder_outputs1_list"] is None:
                    input_ids1 = batch["input_ids"]
                    input_ids2 = batch["input_ids2"]
                    with torch.set_grad_enabled(args.train_text_encoder):   #fussy way of 
                        # Get the text embedding for conditioning
                        # TODO support weighted captions
                        # if args.weighted_captions:
                        #     encoder_hidden_states = get_weighted_text_embeddings(
                        #         tokenizer,
                        #         text_encoder,
                        #         batch["captions"],
                        #         accelerator.device,
                        #         args.max_token_length // 75 if args.max_token_length else 1,
                        #         clip_skip=args.clip_skip,
                        #     )
                        # else:
                        input_ids1 = input_ids1.to(accelerator.device)
                        input_ids2 = input_ids2.to(accelerator.device)
                        # unwrap_model is fine for models not wrapped by accelerator
                        encoder_hidden_states1, encoder_hidden_states2, pool2 = train_util.get_hidden_states_sdxl(
                            args.max_token_length,
                            input_ids1,
                            input_ids2,
                            tokenizer1,
                            tokenizer2,
                            text_encoder1,
                            text_encoder2,
                            None if not args.full_fp16 else weight_dtype,
                            accelerator=accelerator,
                        )
                    if args.auxiliary_TE_loss_anthropic_style_baybee:
                        au_h_args = [args.max_token_length,
                            input_ids1,
                            input_ids2,
                            tokenizer1,
                            tokenizer2,
                            text_encoder1,
                            text_encoder2,
                            None if not args.full_fp16 else weight_dtype,
                            ]
                        au_h_kwargs = {"accelerator":accelerator}
                        if train_text_encoder1:
                            au_h_args[-3]=auxiliary_text_encoder1
                        if train_text_encoder2:
                            au_h_args[-2]=auxiliary_text_encoder2

                        auxiliary_encoder_hidden_states1, auxiliary_encoder_hidden_states2, shim_pool2 = train_util.get_hidden_states_sdxl(
                            *au_h_args, **au_h_kwargs
                        )
                        del shim_pool2


                else:   #this is when ur reusing cached TE embeddings
                    encoder_hidden_states1 = batch["text_encoder_outputs1_list"].to(accelerator.device).to(weight_dtype)
                    encoder_hidden_states2 = batch["text_encoder_outputs2_list"].to(accelerator.device).to(weight_dtype)
                    pool2 = batch["text_encoder_pool2_list"].to(accelerator.device).to(weight_dtype)

                    # # verify that the text encoder outputs are correct
                    # ehs1, ehs2, p2 = train_util.get_hidden_states_sdxl(
                    #     args.max_token_length,
                    #     batch["input_ids"].to(text_encoder1.device),
                    #     batch["input_ids2"].to(text_encoder1.device),
                    #     tokenizer1,
                    #     tokenizer2,
                    #     text_encoder1,
                    #     text_encoder2,
                    #     None if not args.full_fp16 else weight_dtype,
                    # )
                    # b_size = encoder_hidden_states1.shape[0]
                    # assert ((encoder_hidden_states1.to("cpu") - ehs1.to(dtype=weight_dtype)).abs().max() > 1e-2).sum() <= b_size * 2
                    # assert ((encoder_hidden_states2.to("cpu") - ehs2.to(dtype=weight_dtype)).abs().max() > 1e-2).sum() <= b_size * 2
                    # assert ((pool2.to("cpu") - p2.to(dtype=weight_dtype)).abs().max() > 1e-2).sum() <= b_size * 2
                    # logger.info("text encoder outputs verified")

                # get size embeddings
                orig_size = batch["original_sizes_hw"]
                crop_size = batch["crop_top_lefts"]
                target_size = batch["target_sizes_hw"]
                embs = sdxl_train_util.get_size_embeddings(orig_size, crop_size, target_size, accelerator.device).to(weight_dtype)

                # concat embeddings
                vector_embedding = torch.cat([pool2, embs], dim=1).to(weight_dtype)
                text_embedding = torch.cat([encoder_hidden_states1, encoder_hidden_states2], dim=2).to(weight_dtype)
                if args.auxiliary_TE_loss_anthropic_style_baybee:
                    #auxiliary_text_embedding = torch.cat([auxiliary_encoder_hidden_states1, auxiliary_encoder_hidden_states2], dim=2).to(weight_dtype)

                    kloss = torch.nn.KLDivLoss(log_target= True, reduction = "batchmean")
                    mseloss = torch.nn.MSELoss(reduction='none')
                    te_aux_loss=0
                    te_aux_loss1 = torch.zeros_like(encoder_hidden_states1)
                    te_aux_loss2 = torch.zeros_like(encoder_hidden_states2)

                    #haha
                    #te_aux_loss = kloss(textembed_old, textembed_new)
                    #te_aux_loss = kloss(encoder_hidden_states1.log(), auxiliary_encoder_hidden_states1.log())
                    #te_aux_loss += kloss(encoder_hidden_states2.log(), auxiliary_encoder_hidden_states2.log())

                    if train_text_encoder1:
                        te_aux_loss1 = mseloss(encoder_hidden_states1,auxiliary_encoder_hidden_states1)
                        if torch.any(torch.isnan(te_aux_loss1)):
                            #rev1
                            accelerator.print("NaN found in embedding comparison, replacing with zeros")
                            #te_aux_loss1 = torch.nan_to_num(te_aux_loss1, nan=0, posinf=0, neginf=0)
                            #rev2
                            te_aux_loss1 = torch.zeros_like(encoder_hidden_states1)
                        te_aux_loss1 = te_aux_loss1.mean()
                        te_aux_loss += te_aux_loss1
                    if train_text_encoder2:
                        te_aux_loss2 = mseloss(encoder_hidden_states2,auxiliary_encoder_hidden_states2)
                        if torch.any(torch.isnan(te_aux_loss2)):
                            #rev1
                            accelerator.print("NaN found in embedding comparison, replacing with zeros")
                            #te_aux_loss2 = torch.nan_to_num(te_aux_loss2, nan=0, posinf=0, neginf=0)
                            #rev2
                            te_aux_loss2 = torch.zeros_like(encoder_hidden_states2)
                        te_aux_loss2 = te_aux_loss2.mean()
                        te_aux_loss += te_aux_loss2
                    
 
                # Sample noise, sample a random timestep for each image, and add noise to the latents,
                # with noise offset and/or multires noise if specified
                noise, noisy_latents, timesteps, huber_c = train_util.get_noise_noisy_latents_and_timesteps(
                    args, noise_scheduler, latents
                )

                noisy_latents = noisy_latents.to(weight_dtype)  # TODO check why noisy_latents is not weight_dtype

                if args.noisepred_timestep_compensation:
                    noise_comp = (noise_scheduler.alphas_cumprod.sqrt() + (1 - noise_scheduler.alphas_cumprod).sqrt()).to(dtype=weight_dtype, device=latents.device)
                    noisy_latents = noisy_latents / noise_comp[timesteps].reshape(-1, 1, 1, 1)

                # Predict the noise residual
                #z_list = torch.tensor(0.)
                with accelerator.autocast():
                    noise_pred = unet(noisy_latents, timesteps, text_embedding, vector_embedding, ) #auxcum=z_list

                # restore missing vpred training
                if args.v_parameterization:
                    # v-parameterization training
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    target = noise

                if ( 
                    args.min_snr_gamma
                    or args.scale_v_pred_loss_like_noise_pred
                    or args.v_pred_like_loss
                    or args.debiased_estimation_loss
                    or args.masked_loss
                    or args.nullify_implicit_v_lossweight
                    or args.slam_implicit_v_lossweight
                    or args.karras_edm_loss
                    or args.salimans_vpred_weighting
                    or args.sigmoid_k_weighting
                    or args.khrapov_et_al_scheduled_huber_coefficient   #this means that huber coefficient can be different within batches. invalidates batch mean reduction.
                ):
                    # do not mean over batch dimension for snr weight or scale v-pred loss
                    # why was this block hidden here?
                    loss = train_util.conditional_loss(
                        noise_pred.float(), target.float(), reduction="none", loss_type=args.loss_type, huber_c=huber_c, cumtoggle = True, args=args
                    )
                    if args.masked_loss or ("alpha_masks" in batch and batch["alpha_masks"] is not None):
                        loss = apply_masked_loss(loss, batch)
                    if args.use_torchjd:
                        loss = loss
                        #don't mean for torchjd backward case?
                    else:
                        loss = loss.mean([1, 2, 3]) #mean over non-batch dimensions
                    #but wait, does that even matter if the loss weighting is samplewise, 
                    #e.g. bc the timestep parameter only varies along batch dimension?
                    #this should already work fine with huber and smoothed loss. not sure why this wasn't in place forever ago.


                    if args.nullify_implicit_v_lossweight:  #unlike other loss weights this is a mix and match. go wild!
                        loss = nullify_implicit_v_lossweight(loss, timesteps, noise_scheduler, v_prediction=args.v_parameterization)
                    if args.slam_implicit_v_lossweight:  #unlike other loss weights this is a mix and match. go wild!
                        loss = slam_implicit_v_lossweight(loss, timesteps, noise_scheduler, args.v_parameterization)
                    if args.min_snr_gamma:
                        loss = apply_snr_weight(loss, timesteps, noise_scheduler, args.min_snr_gamma, v_prediction=args.v_parameterization)
                    if args.scale_v_pred_loss_like_noise_pred:
                        loss = scale_v_prediction_loss_like_noise_prediction(loss, timesteps, noise_scheduler)
                    if args.v_pred_like_loss:
                        loss = add_v_prediction_like_loss(loss, timesteps, noise_scheduler, args.v_pred_like_loss)
                    if args.debiased_estimation_loss:
                        loss = apply_debiased_estimation(loss, timesteps, noise_scheduler)
                    if args.karras_edm_loss:
                        loss = apply_karras_edm_weighting(loss, timesteps, noise_scheduler)
                    if args.salimans_vpred_weighting:
                        loss = apply_salimans_vpred_weighting(loss, timesteps, noise_scheduler, v_prediction=args.v_parameterization)
                    if args.sigmoid_k_weighting:
                        loss = apply_sigmoid_k_weighting(loss, timesteps, noise_scheduler, v_prediction=args.v_parameterization, k_const=args.sigmoid_k_weighting)
                    
                    if args.use_torchjd:
                        loss = loss
                        #batchsize = loss.size(0) # get batch 
                        #losses = loss.chunk(chunks=batchsize, dim=0)
                    else:
                        loss = loss.mean()  # mean over batch dimension
                else:
                    loss = train_util.conditional_loss(
                        noise_pred.float(), target.float(), reduction="mean", loss_type=args.loss_type, huber_c=huber_c
                    )

                # this is only a convenience wrapper wtf!
                if args.use_torchjd:
                    #loss.backward()
                    #mtl_backward(losses=losses, features=features, aggregator=aggregator)
                    backward(loss, aggregator)
                else:
                    aux_loss = te_aux_loss*args.te_aux_loss_scale
                    
                    accelerator.backward(loss+aux_loss) #+z_list
                

                if not (args.fused_backward_pass or args.fused_optimizer_groups):
                    #post-backpass non-hooked grad clipping implementation.
                    if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                        params_to_clip = []
                        if clippables:  #are we clipping norms? no we're *training* norms!
                            params_to_clip.extend(clippables)
                            params_to_clip.extend(skipweight_params)
                        else:
                            for m in training_models:
                                params_to_clip.extend(m.parameters())

                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                        if args.layernorm_gradient_clipping:
                            params_to_clip = []
                            params_to_clip.extend(norm_params)
                            accelerator.clip_grad_norm_(params_to_clip, args.layernorm_gradient_clipping)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    if args.bias_abolisher or args.layernorm_simple:
                        #for model in accelerator._models:
                        #    cleanup( model, affinenormbiases=affinenormbiases, learnedlambdas=skipweight_params, incremental_abolish=incremental_abolish_ls)
                        cleanup(unet, affinenormbiases=affinenormbiases, affinenormweights=affinenormweights, learnedlambdas=skipweight_params, incremental_abolish=incremental_abolish_ls)
                else:
                    # optimizer.step() and optimizer.zero_grad() are called in the optimizer hook
                    lr_scheduler.step()
                    if args.fused_optimizer_groups:
                        for i in range(1, len(optimizers)):
                            lr_schedulers[i].step()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                sdxl_train_util.sample_images(
                    accelerator,
                    args,
                    None,
                    global_step,
                    accelerator.device,
                    vae,
                    [tokenizer1, tokenizer2],
                    [text_encoder1, text_encoder2],
                    unet,
                    kwargs=betascale_kwargs
                )

                # 指定ステップごとにモデルを保存
                if args.save_every_n_steps is not None and global_step % args.save_every_n_steps == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        src_path = src_stable_diffusion_ckpt if save_stable_diffusion_format else src_diffusers_model_path
                        sdxl_train_util.save_sd_model_on_epoch_end_or_stepwise(
                            args,
                            False,
                            accelerator,
                            src_path,
                            save_stable_diffusion_format,
                            use_safetensors,
                            save_dtype,
                            epoch,
                            num_train_epochs,
                            global_step,
                            accelerator.unwrap_model(text_encoder1),
                            accelerator.unwrap_model(text_encoder2),
                            accelerator.unwrap_model(unet),
                            vae,
                            logit_scale,
                            ckpt_info,
                        )
                        if args.bias_abolisher or args.layernorm_simple:
                            accelerator.print(f"onsave biasminmax:{bias_yoinkems(unet=unet,affinenormbiases=affinenormbiases)}")
                        if args.layernorm_simple:
                            accelerator.print(f"onsave weightminmax:{bias_yoinkems(unet=unet,affinenormbiases=affinenormweights)}")

            current_loss = loss.detach().item()  # 平均なのでbatch sizeは関係ないはず
            aux_current_loss = aux_loss.detach().item()
            if args.logging_dir is not None:
                logs = {"loss": current_loss}
                if args.auxiliary_TE_loss_anthropic_style_baybee:
                    logs["aux_loss"] = aux_current_loss
                if block_lrs is None:
                    train_util.append_lr_to_logs(logs, lr_scheduler, args.optimizer_type, including_unet=train_unet)
                else:
                    append_block_lr_to_logs(block_lrs, logs, lr_scheduler, args.optimizer_type)  # U-Net is included in block_lrs

                accelerator.log(logs, step=global_step)

            loss_recorder.add(epoch=epoch, step=step, loss=current_loss)
            avr_loss: float = loss_recorder.moving_average
            logs = {"avr_loss": avr_loss}  # , "lr": lr_scheduler.get_last_lr()[0]}
            if args.auxiliary_TE_loss_anthropic_style_baybee:
                aux_loss_recorder.add(epoch=epoch, step=step, loss=aux_current_loss)
                aux_avr_loss: float = aux_loss_recorder.moving_average
                logs["aux_avr_loss"] = aux_avr_loss
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        if args.logging_dir is not None:
            logs = {"loss/epoch": loss_recorder.moving_average}
            if args.auxiliary_TE_loss_anthropic_style_baybee:
                logs["aux_loss/epoch"] = aux_loss_recorder.moving_average
            accelerator.log(logs, step=epoch + 1)

        accelerator.wait_for_everyone()

        if args.save_every_n_epochs is not None:
            if accelerator.is_main_process:
                src_path = src_stable_diffusion_ckpt if save_stable_diffusion_format else src_diffusers_model_path
                sdxl_train_util.save_sd_model_on_epoch_end_or_stepwise(
                    args,
                    True,
                    accelerator,
                    src_path,
                    save_stable_diffusion_format,
                    use_safetensors,
                    save_dtype,
                    epoch,
                    num_train_epochs,
                    global_step,
                    accelerator.unwrap_model(text_encoder1),
                    accelerator.unwrap_model(text_encoder2),
                    accelerator.unwrap_model(unet),
                    vae,
                    logit_scale,
                    ckpt_info,
                )

        sdxl_train_util.sample_images(
            accelerator,
            args,
            epoch + 1,
            global_step,
            accelerator.device,
            vae,
            [tokenizer1, tokenizer2],
            [text_encoder1, text_encoder2],
            unet,
            kwargs=betascale_kwargs
        )

    is_main_process = accelerator.is_main_process
    # if is_main_process:
    unet = accelerator.unwrap_model(unet)
    text_encoder1 = accelerator.unwrap_model(text_encoder1)
    text_encoder2 = accelerator.unwrap_model(text_encoder2)

    accelerator.end_training()
    if args.bias_abolisher or args.layernorm_simple:
        accelerator.print(f"onsave biasminmax:{bias_yoinkems(unet=unet,affinenormbiases=affinenormbiases)}")
    if args.layernorm_simple:
        accelerator.print(f"onsave weightminmax:{bias_yoinkems(unet=unet,affinenormbiases=affinenormweights)}")

    if args.save_state or args.save_state_on_train_end:
        train_util.save_state_on_train_end(args, accelerator)

    del accelerator  # この後メモリを使うのでこれは消す

    if is_main_process:
        src_path = src_stable_diffusion_ckpt if save_stable_diffusion_format else src_diffusers_model_path
        sdxl_train_util.save_sd_model_on_train_end(
            args,
            src_path,
            save_stable_diffusion_format,
            use_safetensors,
            save_dtype,
            epoch,
            global_step,
            text_encoder1,
            text_encoder2,
            unet,
            vae,
            logit_scale,
            ckpt_info,
        )
        logger.info("model saved.")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    add_logging_arguments(parser)
    train_util.add_sd_models_arguments(parser)
    train_util.add_dataset_arguments(parser, True, True, True)
    train_util.add_training_arguments(parser, False)
    train_util.add_masked_loss_arguments(parser)
    deepspeed_utils.add_deepspeed_arguments(parser)
    train_util.add_sd_saving_arguments(parser)
    train_util.add_optimizer_arguments(parser)
    config_util.add_config_arguments(parser)
    custom_train_functions.add_custom_train_arguments(parser)
    sdxl_train_util.add_sdxl_training_arguments(parser)

    parser.add_argument(
        "--learning_rate_te1",
        type=float,
        default=None,
        help="learning rate for text encoder 1 (ViT-L) / text encoder 1 (ViT-L)の学習率",
    )
    parser.add_argument(
        "--learning_rate_te2",
        type=float,
        default=None,
        help="learning rate for text encoder 2 (BiG-G) / text encoder 2 (BiG-G)の学習率",
    )

    parser.add_argument(
        "--diffusers_xformers", action="store_true", help="use xformers by diffusers / Diffusersでxformersを使用する"
    )
    parser.add_argument("--train_text_encoder", action="store_true", help="train text encoder / text encoderも学習する")
    parser.add_argument(
        "--no_half_vae",
        action="store_true",
        help="do not use fp16/bf16 VAE in mixed precision (use float VAE) / mixed precisionでも fp16/bf16 VAEを使わずfloat VAEを使う",
    )
    parser.add_argument(
        "--block_lr",
        type=str,
        default=None,
        help=f"learning rates for each block of U-Net, comma-separated, {UNET_NUM_BLOCKS_FOR_BLOCK_LR} values / "
        + f"U-Netの各ブロックの学習率、カンマ区切り、{UNET_NUM_BLOCKS_FOR_BLOCK_LR}個の値",
    )
    parser.add_argument(
        "--fused_optimizer_groups",
        type=int,
        default=None,
        help="number of optimizers for fused backward pass and optimizer step / fused backward passとoptimizer stepのためのoptimizer数",
    )
    parser.add_argument(
        "--learnable_lambdas_level",
        type=int,
        default=1,
        help="porting all lambdalevel configurations back to the first trainer and net! 0=off, 1=mlp&interblock, 2=&xattn, 3=&sattn, 4=&resnets",
    )
    parser.add_argument(
        "--ll_lr",
        type=float,
        default=None,
        help="pass 1 learned lambda lr vlues.",
    )
    parser.add_argument(
        "--swiglu_switcharoo",
        action="store_true",
        default=False,
        help="what if our network was a swiglunet instead of a geglunet? switch this to True to find out!",
    )
    parser.add_argument(
        "--norm_salvation",
        action="store_true",
        default=None,
        help="peel apart normalization scale and shift parameters from like all other parameters.",
    )
    parser.add_argument(
        "--layernorm_gradient_clipping",
        type=float,
        default=None,
        help="then constrain the layernorm gradients anyways.",
    )
    parser.add_argument(
        "--layernorm_decay",
        type=float,
        default=0.0,
        help="then decay the layernorm gradients anyways.",
    )
    parser.add_argument(
        "--layernorm_adambetas",
        nargs=2,
        type=float,
        default=None,
        help="pass 2 space separated beta1 and beta2 vlues.",
    )
    parser.add_argument(
        "--layernorm_lr",
        type=float,
        default=None,
        help="pass 1 lr vlues.",
    )
    parser.add_argument(
        "--bias_abolisher",
        action="store_true",
        default=None,
        help="get rid of layernorm biases.",
    )
    parser.add_argument(
        "--outblock_bias_abolisher",
        action="store_true",
        default=None,
        help="get rid of layernorm biases but only in the out blocks.",
    )
    parser.add_argument(
        "--groupnorm_bias_abolisher_auxiliary",
        action="store_true",
        default=None,
        help="get rid of groupnorm affine biases too.",
    )
    parser.add_argument(
        "--incremental_abolish",
        type=float,
        default=None,
        help="interpolation term between clamped and nonclamped layernorm biases. pick something between 0.01 and 0.9.",
    )
    parser.add_argument(
        "--layernorm_simple",
        action="store_true",
        default=None,
        help="turn a layernorm into a layernorm-simple from https://arxiv.org/pdf/1911.07013, (making layernorm a gradient conditioner).",
    )
    parser.add_argument(
        "--pytorch_pinned_dataloader",
        action="store_true",
        default=None,
        help="cuda pinned memory dataloader.",
    )
    parser.add_argument(
        "--use_laser_sdpa",
        action="store_true",
        default=None,
        help="rescale those numbers",
    )
    parser.add_argument(
        "--use_qknorm",
        action="store_true",
        default=None,
        help="rescale query, key to minimize saturation. also another learnable parameter :)",
    )
    parser.add_argument(
        "--use_laser_qknorm",
        action="store_true",
        default=None,
        help="rescale those numbers, then rescale query, key to minimize saturation. also another learnable parameter :)",
    )
    parser.add_argument(
        "--use_layerqknorm",
        action="store_true",
        default=None,
        help="rescale those numbers, then rescale query, key to minimize saturation. no learnable parameter :)",
    )
    parser.add_argument(
        "--use_torchjd",
        action="store_true",
        default=None,
        help="multiple gradient loss",
    )
    parser.add_argument(
        "--auxiliary_TE_loss_anthropic_style_baybee",
        action="store_true",
        default=None,
        help="""amend TE training with a MSE distance between snapshotted text encoder hidden states and present text encoder hidden states.
        total memory hog; training both te1 and te2 is likely not possible in a consumer system.
        several model loading doohickies have been implemented to reduce memory consumption for training either te1 *or* te2.
        this method is brutally unstable, no checkpoints have ever been trained with a --learning_rate_te2 greater than 1e-07.""",
    )
    parser.add_argument(
        "--auxiliary_TE_from_path",
        type=str,
        default=None,
        help="path to reference checkpoint for text encoder aux loss",
    )
    parser.add_argument(
        "--auxiliary_TE_from_same",
        action="store_true",
        default=None,
        help="snapshot own pretrained model's checkpoints, not recommended unless you're confident you won't lose the checkpoint",
    )
    parser.add_argument(
        "--te_aux_loss_scale",
        type=float,
        default=1e-3,
        help="scale of auxiliary loss relative to base loss. 1e-3 recommended starting value.",
    )
    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)

    train(args)
