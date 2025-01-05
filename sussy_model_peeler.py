import os, sys

sys.path.insert(0, os.getcwd())
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "base_model",
        help="The model which use it to train the dreambooth model",
        default="",
        type=str,
    )
    parser.add_argument(
        "db_model",
        help="the dreambooth model you want to extract the locon",
        default="",
        type=str,
    )
    parser.add_argument(
        "output_name", help="the output model", default="./out.pt", type=str
    )
    parser.add_argument(
        "--is_v2",
        help="Your base/db model is sd v2 or not",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--is_sdxl",
        help="Your base/db model is sdxl or not",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--device",
        help="Which device you want to use to extract the locon",
        default="cpu",
        type=str,
    )
    parser.add_argument(
        "--mode",
        help=(
            'extraction mode, can be "full", "fixed", "threshold", "ratio", "quantile". '
            'If not "fixed", network_dim and conv_dim will be ignored'
        ),
        default="fixed",
        type=str,
    )
    parser.add_argument(
        "--safetensors",
        help="use safetensors to save locon model",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--linear_dim",
        help="network dim for linear layer in fixed mode",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--conv_dim",
        help="network dim for conv layer in fixed mode",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--linear_threshold",
        help="singular value threshold for linear layer in threshold mode",
        default=0.0,
        type=float,
    )
    parser.add_argument(
        "--conv_threshold",
        help="singular value threshold for conv layer in threshold mode",
        default=0.0,
        type=float,
    )
    parser.add_argument(
        "--linear_ratio",
        help="singular ratio for linear layer in ratio mode",
        default=0.0,
        type=float,
    )
    parser.add_argument(
        "--conv_ratio",
        help="singular ratio for conv layer in ratio mode",
        default=0.0,
        type=float,
    )
    parser.add_argument(
        "--linear_quantile",
        help="singular value quantile for linear layer quantile mode",
        default=1.0,
        type=float,
    )
    parser.add_argument(
        "--conv_quantile",
        help="singular value quantile for conv layer quantile mode",
        default=1.0,
        type=float,
    )
    parser.add_argument(
        "--use_sparse_bias",
        help="enable sparse bias",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--sparsity", help="sparsity for sparse bias", default=0.98, type=float
    )
    parser.add_argument(
        "--disable_cp",
        help="don't use cp decomposition",
        default=False,
        action="store_true",
    )
    return parser.parse_args()


ARGS = get_args()


from lycoris.utils import extract_linear, extract_conv, make_sparse
from library.model_util import load_models_from_stable_diffusion_checkpoint
from library.sdxl_model_util import load_models_from_sdxl_checkpoint

import torch
from safetensors.torch import save_file


from tqdm import tqdm

@torch.no_grad()
def extract_diff(
    base_tes,
    db_tes,
    base_unet,
    db_unet,
    mode="fixed",
    linear_mode_param=0,
    conv_mode_param=0,
    extract_device="cpu",
    use_bias=False,
    sparsity=0.98,
    small_conv=True,
):
    UNET_TARGET_REPLACE_MODULE = [
        "Linear",
        "Conv2d",
        "LayerNorm",
        "GroupNorm",
        "GroupNorm32",
    ]
    TEXT_ENCODER_TARGET_REPLACE_MODULE = [
        "Embedding",
        "Linear",
        "Conv2d",
        "LayerNorm",
        "GroupNorm",
        "GroupNorm32",
    ]
    LORA_PREFIX_UNET = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"

    def make_state_dict(
        prefix,
        root_module: torch.nn.Module,
        target_module: torch.nn.Module,
        target_replace_modules,
    ):
        loras = {}
        temp = {}

        for name, module in root_module.named_modules():
            if module.__class__.__name__ in target_replace_modules:
                temp[name] = module

        for name, module in tqdm(
            list((n, m) for n, m in target_module.named_modules() if n in temp)
        ):
            weights = temp[name].to(torch.float)
            lora_name = prefix + "." + name
            lora_name = lora_name.replace(".", "_")
            layer = module.__class__.__name__

            if layer in {
                "Linear",
                "Conv2d",
                "LayerNorm",
                "GroupNorm",
                "GroupNorm32",
                "Embedding",
            }:
                root_weight = module.weight.to(torch.float)
                if torch.allclose(root_weight, weights.weight):
                    continue
            else:
                continue
            module = module.to(extract_device)
            weights = weights.to(extract_device)

            if mode == "full":
                decompose_mode = "full"
            elif layer == "Linear":
                weight, decompose_mode = extract_linear(
                    (root_weight - weights.weight),
                    mode,
                    linear_mode_param,
                    device=extract_device,
                )
                if decompose_mode == "low rank":
                    extract_a, extract_b, diff = weight
            elif layer == "Conv2d":
                is_linear = root_weight.shape[2] == 1 and root_weight.shape[3] == 1
                weight, decompose_mode = extract_conv(
                    (root_weight - weights.weight),
                    mode,
                    linear_mode_param if is_linear else conv_mode_param,
                    device=extract_device,
                )
                if decompose_mode == "low rank":
                    extract_a, extract_b, diff = weight
                if small_conv and not is_linear and decompose_mode == "low rank":
                    dim = extract_a.size(0)
                    (extract_c, extract_a, _), _ = extract_conv(
                        extract_a.transpose(0, 1),
                        "fixed",
                        dim,
                        extract_device,
                        True,
                    )
                    extract_a = extract_a.transpose(0, 1)
                    extract_c = extract_c.transpose(0, 1)
                    loras[f"{lora_name}.lora_mid.weight"] = (
                        extract_c.detach().cpu().contiguous().half()
                    )
                    diff = (
                        (
                            root_weight
                            - torch.einsum(
                                "i j k l, j r, p i -> p r k l",
                                extract_c,
                                extract_a.flatten(1, -1),
                                extract_b.flatten(1, -1),
                            )
                        )
                        .detach()
                        .cpu()
                        .contiguous()
                    )
                    del extract_c
            else:
                module = module.to("cpu")
                weights = weights.to("cpu")
                continue

            if decompose_mode == "low rank":
                loras[f"{lora_name}.lora_down.weight"] = (
                    extract_a.detach().cpu().contiguous().half()
                )
                loras[f"{lora_name}.lora_up.weight"] = (
                    extract_b.detach().cpu().contiguous().half()
                )
                loras[f"{lora_name}.alpha"] = torch.Tensor([extract_a.shape[0]]).half()
                if use_bias:
                    diff = diff.detach().cpu().reshape(extract_b.size(0), -1)
                    sparse_diff = make_sparse(diff, sparsity).to_sparse().coalesce()

                    indices = sparse_diff.indices().to(torch.int16)
                    values = sparse_diff.values().half()
                    loras[f"{lora_name}.bias_indices"] = indices
                    loras[f"{lora_name}.bias_values"] = values
                    loras[f"{lora_name}.bias_size"] = torch.tensor(diff.shape).to(
                        torch.int16
                    )
                del extract_a, extract_b, diff
            elif decompose_mode == "full":
                if "Norm" in layer:
                    w_key = "w_norm"
                    b_key = "b_norm"
                else:
                    w_key = "diff"
                    b_key = "diff_b"
                weight_diff = module.weight - weights.weight
                loras[f"{lora_name}.{w_key}"] = (
                    weight_diff.detach().cpu().contiguous().half()
                )
                if getattr(weights, "bias", None) is not None:
                    bias_diff = module.bias - weights.bias
                    loras[f"{lora_name}.{b_key}"] = (
                        bias_diff.detach().cpu().contiguous().half()
                    )
            else:
                raise NotImplementedError
            module = module.to("cpu")
            weights = weights.to("cpu")
        return loras

    all_loras = {}

    all_loras |= make_state_dict(
        LORA_PREFIX_UNET,
        base_unet,
        db_unet,
        UNET_TARGET_REPLACE_MODULE,
    )
    del base_unet, db_unet
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # for idx, (te1, te2) in enumerate(zip(base_tes, db_tes)):
    #     if len(base_tes) > 1:
    #         prefix = f"{LORA_PREFIX_TEXT_ENCODER}{idx+1}"
    #     else:
    #         prefix = LORA_PREFIX_TEXT_ENCODER
    #     all_loras |= make_state_dict(
    #         prefix,
    #         te1,
    #         te2,
    #         TEXT_ENCODER_TARGET_REPLACE_MODULE,
    #     )
    #     del te1, te2

    all_lora_name = set()
    for k in all_loras:
        lora_name, weight = k.rsplit(".", 1)
        all_lora_name.add(lora_name)
    print(len(all_lora_name))
    return all_loras


def main():
    args = ARGS
    if args.is_sdxl:
        text_model1, text_model2, vae, unet, logit_scale, ckpt_info = load_models_from_sdxl_checkpoint(None, args.base_model, "cuda:0", dtype=torch.float16)
        del text_model1
        del text_model2
        del vae
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        
        base = [{}, {}, {}, unet, {}, {}]
        base_tes = [base[0], base[1]]
        base_unet = base[3]
            
        text_model1, text_model2, vae, unet, logit_scale, ckpt_info = load_models_from_sdxl_checkpoint(None, args.db_model, "cuda:0", dtype=torch.float16)
        del text_model1
        del text_model2
        del vae
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        
        db = [{}, {}, {}, unet, {}, {}]
        db_tes = [db[0], db[1]]
        db_unet = db[3]
        
        
            
    else:
        base = load_models_from_stable_diffusion_checkpoint(args.is_v2, args.base_model)
        base_tes = [base[0]]
        base_unet = base[2]
        
        db = load_models_from_stable_diffusion_checkpoint(args.is_v2, args.db_model)
        db_tes = [db[0]]
        db_unet = db[2]

    linear_mode_param = {
        "fixed": args.linear_dim,
        "threshold": args.linear_threshold,
        "ratio": args.linear_ratio,
        "quantile": args.linear_quantile,
        "full": None,
    }[args.mode]
    conv_mode_param = {
        "fixed": args.conv_dim,
        "threshold": args.conv_threshold,
        "ratio": args.conv_ratio,
        "quantile": args.conv_quantile,
        "full": None,
    }[args.mode]
        

    state_dict = extract_diff(
        base_tes,
        db_tes,
        base_unet,
        db_unet,
        args.mode,
        linear_mode_param,
        conv_mode_param,
        args.device,
        args.use_sparse_bias,
        args.sparsity,
        not args.disable_cp,
    )

    if args.safetensors:
        save_file(state_dict, args.output_name)
    else:
        torch.save(state_dict, args.output_name)


if __name__ == "__main__":
    main()
