#coding:utf8

import os
import sys
import copy     
import argparse
import torch.jit
from tqdm import tqdm
import torch
import torch.nn as nn

import csv
import json
from datetime import datetime

from utils.data_utils import *
from component.svd_llama import SVD_LlamaAttention, SVD_LlamaMLP
from component.svd_mistral import SVD_MistralAttention, SVD_MistralMLP
from component.svd_opt import SVDOPTDecoderLayer
from utils.model_utils import *
from evaluater import *

from pathlib import Path
import time

import math

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_path)


import psutil
import os

def print_cpu_mem(tag=""):
    process = psutil.Process(os.getpid())
    mem_gb = process.memory_info().rss / 1024**3
    print(f"[MEM {tag}] Process RAM: {mem_gb:.2f} GB")


def save_experiment(args, ppls, save_dir="results"):

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/runs", exist_ok=True)

    run_dir = f"{save_dir}/runs"
    os.makedirs(run_dir, exist_ok=True)

    row = vars(args).copy()

    for dataset, ppl in ppls.items():
        row[f"ppl_{dataset}"] = ppl

    row["ppl_avg"] = sum(ppls.values()) / len(ppls)

    row["gpu_mem_mib"] = (
        torch.cuda.memory_allocated() / 1024 / 1024
    )

    if args.model_path is not None and args.model_path != "original":

        if args.model_path == "svd_llm":
            model_path_time = get_svd_llm_save_path(args)
        elif args.model_path == "svd_surgeon":
            model_path_time = get_model_save_path(args)
        else:
            model_path_time = args.model_path 
            
        timing_path = model_path_time.replace('.pt', '_timing.json')
        if os.path.exists(timing_path):
            with open(timing_path, 'r') as f:
                timing = json.load(f)
            row["pruning_time_seconds"] = timing.get('pruning_time_seconds', None)
            row["algorithm_time_seconds"] = timing.get('algorithm_time_seconds', None)
        else:
            row["pruning_time_seconds"] = None
            row["algorithm_time_seconds"] = None
    else:
        row["pruning_time_seconds"] = None
        row["algorithm_time_seconds"] = None


    with open(f"{run_dir}/config.json", "w") as f:
        json.dump(vars(args), f, indent=4)

    with open(f"{run_dir}/metrics.json", "w") as f:
        json.dump({
            "ppls": ppls,
            "ppl_avg": row["ppl_avg"],
            "gpu_mem_mib": row["gpu_mem_mib"]
        }, f, indent=4)

    csv_path = f"{save_dir}/summary.csv"

    file_exists = os.path.isfile(csv_path)

    fieldnames = sorted(row.keys())

    with open(csv_path, "a", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)

    print(f"Saved experiment to {run_dir}")


def get_model_save_path(args):
    def fmt(x):
        return f"{float(x):.10g}"
    return (
        args.save_path + "/" +
        args.model.replace("/", "_").replace("-", "_") +
        '_' + args.dataset +
        '_svd_surgeon' +
        '_sel' + str(args.select_by_loss) +
        '_r' + fmt(args.ratio) +
        '_sc' + fmt(args.obs_scale) +
        '_d' + fmt(args.obs_damping) +
        '_hd' + fmt(args.obs_hdamping) +
        '_a' + fmt(args.alpha) +
        '_seed' + str(args.seed) +
        '.pt'
    )

def get_svd_llm_save_path(args):
    return (
        args.save_path + "/" +
        args.model.replace("/", "_").replace("-", "_") +
        '_' + args.dataset +
        '_svd_llm' +
        '_r' + f"{float(args.ratio):.10g}" +
        '_seed' + str(args.seed) +
        '.pt'
    )

def get_vanilla_svd_save_path(args):
    return (
        args.save_path + "/" +
        args.model.replace("/", "_").replace("-", "_") +
        '_' + args.dataset +
        '_vanilla_svd' +
        '_r' + f"{float(args.ratio):.10g}" +
        '_seed' + str(args.seed) +
        '.pt'
    )


# ============================================================
# OBS helper functions
# ============================================================

def spectral_grad_from_weight_grad(G, U, Vh, full_k):
    """
    Project the weight-space gradient G (out, in) onto the
    SVD basis to get a spectral gradient vector of length full_k.
    """
    device = G.device
    U_k  = U[:, :full_k].to(device)
    V_k  = Vh[:full_k, :].T.to(device)
    gbar = (U_k * (G.float() @ V_k)).sum(dim=0)
    return gbar


def obs_select_by_score(S, Hbar, rank, full_k, damping=1e-5, ratio=0.1):
    """
    Choose which rank singular values to keep based on the
    OBS sensitivity score  s_i = sigma_i^2 / [H^{-1}]_{ii}.
    Returns (keep_idx, drop_idx) as sorted index tensors.
    """
    device = S.device
    dtype  = S.dtype
    H      = Hbar.to(device=device, dtype=dtype)
    dmean  = torch.diag(H).mean().clamp(min=1e-12)

    H      = H + damping * dmean * torch.eye(full_k, device=device, dtype=dtype)
    scores = S[:full_k].pow(2) / torch.diag(torch.linalg.pinv(H))
    keep_idx, _ = torch.sort(torch.topk(scores, rank).indices)
    mask = torch.ones(full_k, dtype=torch.bool, device=device)
    mask[keep_idx] = False
    return keep_idx, torch.arange(full_k, device=device)[mask]


def obs_update_singular_values_with_selection(
        S, rank, Hbar, damping=1e-5, hdamping=1e-1,
        do_update=True, obs_scale=1.0, alpha=0.1, select_by_loss=True, ratio=0.1):
    """
    Core OBS correction in the SVD basis.

    Given singular values S (full vector, descending), the target
    rank, the spectral Fisher Hbar, and a pool size
    full_k = rank + int((S.numel() - rank) * alpha):

      1. Optionally select which `rank` indices to keep by loss
         sensitivity (select_by_loss=True) rather than by magnitude.
      2. Solve for the correction delta that compensates for the
         dropped singular values:
             delta = H_SS^{-1} H_SC sigma_C
      3. Return the corrected kept singular values.
    """
    device  = S.device
    rank    = min(rank, S.numel())

    full_k = rank + int((S.numel() - rank) * alpha)


    S_block = S[:full_k]
    Hbar    = Hbar[:full_k, :full_k].to(device=device, dtype=S.dtype)

    if select_by_loss:
        keep_idx, drop_idx = obs_select_by_score(S_block, Hbar, rank, full_k, hdamping, ratio=ratio)
    else:
        keep_idx = torch.arange(rank,         device=device)
        drop_idx = torch.arange(rank, full_k, device=device)

    sigma_S = S_block[keep_idx]
    sigma_C = S_block[drop_idx]

    
    if (not do_update) or sigma_C.numel() == 0:
        return sigma_S, keep_idx, drop_idx

    H_SS = Hbar[keep_idx][:, keep_idx]
    H_SC = Hbar[keep_idx][:, drop_idx]
    
    dmean = torch.diag(H_SS).mean().clamp(min=1e-12)
    H_SS = H_SS + damping * dmean * torch.eye(H_SS.shape[0], device=device, dtype=H_SS.dtype)

    delta = torch.linalg.solve(H_SS, H_SC @ sigma_C)

    return torch.clamp(sigma_S + obs_scale * delta, min=0.0), keep_idx, drop_idx


# ============================================================
# Collect Spectral Fisher
# ============================================================

def collect_spectral_fisher_small(model, calib_loader, dev,
                                    layer_svd_info, num_batches=16,
                                    hbar_save_path=None, reuse_hbars=False):
    """
    Accumulate the spectral Fisher (Hbar) for every layer listed
    in layer_svd_info by running num_batches backward passes.
    layer_svd_info: dict  name -> {"U", "Vh", "full_k", ...}
                    as built inside whitening_obs().
    hbar_save_path: str or Path, optional. Where to save/load Hbars.
    reuse_hbars:    if True and hbar_save_path exists, load and return directly.
    Returns: dict  name -> Hbar tensor of shape (full_k, full_k).
    """
    if reuse_hbars and hbar_save_path is not None:
        path = Path(hbar_save_path)
        if path.exists():
            print(f"[Hbar] Loading cached Hbars from {path}")
            return torch.load(path, map_location=dev)
        else:
            print(f"[Hbar] reuse_hbars=True but {path} not found — recomputing.")

    print("info[""full_k""]:", [info["full_k"] for info in layer_svd_info.values()])

    Hbars = {
        n: torch.zeros(info["full_k"], info["full_k"], device=dev)
        for n, info in layer_svd_info.items()
    }
    model = model.to(dev)
    model.train()
    for i, batch in enumerate(calib_loader):
        if i >= num_batches:
            break
        model.zero_grad(set_to_none=True)
        batch = {k: v.to(dev) for k, v in batch.items()}
        loss  = model(**batch, labels=batch["input_ids"]).loss
        loss.backward()
        for name, info in layer_svd_info.items():
            module = model
            for part in name.split("."):
                module = getattr(module, part)
            grad = module.weight.grad
            if grad is None:
                continue
            if info["scaling_matrix_inv"] is not None:
                scaling_matrix_inv = info["scaling_matrix_inv"].to(grad.device).float()
                grad_scaled = grad.float() @ scaling_matrix_inv.T
            else:
                grad_scaled = grad.float()
            
            gbar = spectral_grad_from_weight_grad(
                grad_scaled.detach(), info["U"], info["Vh"], info["full_k"])
            Hbars[name] += torch.outer(gbar, gbar)
    model.eval()

    if hbar_save_path is not None:
        path = Path(hbar_save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(Hbars, path)
        print(f"[Hbar] Saved Hbars to {path}")
        
    return Hbars


def collect_spectral_fisher_large(model, calib_loader, dev,
                                    layer_svd_info, num_batches=16,
                                    hbar_save_path=None, reuse_hbars=False):
    """
    Accumulate the spectral Fisher (Hbar) for every layer listed
    in layer_svd_info by running num_batches backward passes.
    layer_svd_info: dict  name -> {"U", "Vh", "full_k", ...}
                    as built inside whitening_obs().
    hbar_save_path: str or Path, optional. Where to save/load Hbars.
    reuse_hbars:    if True and hbar_save_path exists, load and return directly.
    Returns: dict  name -> Hbar tensor of shape (full_k, full_k).
    """
    # --- Load cached Hbars if requested ---
    if reuse_hbars and hbar_save_path is not None:
        path = Path(hbar_save_path)
        if path.exists():
            print(f"[Hbar] Loading cached Hbars from {path}")
            return torch.load(path, map_location=dev)
        else:
            print(f"[Hbar] reuse_hbars=True but {path} not found — recomputing.")

    Hbars = {
        n: torch.zeros(info["full_k"], info["full_k"], device='cpu')
        for n, info in layer_svd_info.items()
    }

    # enable gradient checkpointing for large models
    model_params = sum(p.numel() for p in model.parameters())
    use_grad_ckpt = model_params > 1e9

    model = model.to(dev)
    if use_grad_ckpt:
        model = model.half()
        model.gradient_checkpointing_enable()
        print(f"Gradient checkpointing enabled ({model_params/1e9:.1f}B params)")
    model.train()
    print(f"Model loaded: {torch.cuda.memory_allocated()/1024**3:.2f} GB")


    hooks = []
    for name, info in layer_svd_info.items():
        module = model
        for part in name.split("."):
            module = getattr(module, part)

        def make_hook(n, inf):
            def hook(grad):
                if inf["scaling_matrix_inv"] is not None:
                    scaling_matrix_inv = inf["scaling_matrix_inv"].to(grad.device).float()
                    grad_scaled = grad.float() @ scaling_matrix_inv.T
                    del scaling_matrix_inv
                else:
                    grad_scaled = grad.float()

                U_gpu  = inf["U"].to(grad.device)
                Vh_gpu = inf["Vh"].to(grad.device)
                gbar = spectral_grad_from_weight_grad(
                    grad_scaled.detach(), U_gpu, Vh_gpu, inf["full_k"])
                Hbars[n] += torch.outer(gbar, gbar).cpu()

                del gbar, grad_scaled, U_gpu, Vh_gpu
                return None
            return hook

        hooks.append(module.weight.register_hook(make_hook(name, info)))

    for i, batch in enumerate(calib_loader):
        if i >= num_batches:
            break
        if i % 100 == 0:
            print(f"Fisher batch {i}/{num_batches}, "
                  f"GPU mem: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        model.zero_grad(set_to_none=True)
        batch = {k: v.to(dev) for k, v in batch.items()}
        loss = model(**batch, labels=batch["input_ids"]).loss
        loss.backward()
        del loss
        for k in batch:
            batch[k] = batch[k].cpu()
        torch.cuda.empty_cache()

    for h in hooks:
        h.remove()

    if use_grad_ckpt:
        model.gradient_checkpointing_disable()
    model.eval()
    ## model = model.float()
    model = model.cpu().float()
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    print(f"After Fisher collection: {torch.cuda.memory_allocated()/1024**3:.2f} GB GPU, "
         f"{torch.cuda.memory_reserved()/1024**3:.2f} GB reserved")

    if hbar_save_path is not None:
        path = Path(hbar_save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(Hbars, path)
        print(f"[Hbar] Saved Hbars to {path}")

    return Hbars




def collect_spectral_fisher_in_chunks(model, calib_loader, dev,
                                    layer_svd_info, num_batches=16,
                                    hbar_save_path=None,
                                    chunk_id=0, num_chunks=1):
    chunk_size  = num_batches // num_chunks
    start_batch = chunk_id * chunk_size
    end_batch   = start_batch + chunk_size
    print(f"[Hbar] Chunk {chunk_id}/{num_chunks}: batches {start_batch} to {end_batch}")

    Hbars = {
        n: torch.zeros(info["full_k"], info["full_k"], device='cpu')
        for n, info in layer_svd_info.items()
    }

    # enable gradient checkpointing for large models
    model_params = sum(p.numel() for p in model.parameters())
    use_grad_ckpt = model_params > 1e9

    model = model.to(dev)
    if use_grad_ckpt:
        model = model.half()
        model.gradient_checkpointing_enable()
        print(f"Gradient checkpointing enabled ({model_params/1e9:.1f}B params)")
    model.train()
    print(f"Model loaded: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    hooks = []
    for name, info in layer_svd_info.items():
        module = model
        for part in name.split("."):
            module = getattr(module, part)

        def make_hook(n, inf):
            def hook(grad):
                if inf["scaling_matrix_inv"] is not None:
                    scaling_matrix_inv = inf["scaling_matrix_inv"].to(grad.device).float()
                    grad_scaled = grad.float() @ scaling_matrix_inv.T
                    del scaling_matrix_inv
                else:
                    grad_scaled = grad.float()

                U_gpu  = inf["U"].to(grad.device)
                Vh_gpu = inf["Vh"].to(grad.device)
                gbar = spectral_grad_from_weight_grad(
                    grad_scaled.detach(), U_gpu, Vh_gpu, inf["full_k"])
                Hbars[n] += torch.outer(gbar, gbar).cpu()

                del gbar, grad_scaled, U_gpu, Vh_gpu
                return None
            return hook

        hooks.append(module.weight.register_hook(make_hook(name, info)))

    for i, batch in enumerate(calib_loader):
        if i < start_batch:
            continue
        if i >= end_batch:
            break
        # print(f"Fisher batch {i}/{num_batches}, "
        #           f"GPU mem: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        model.zero_grad(set_to_none=True)
        batch = {k: v.to(dev) for k, v in batch.items()}
        loss = model(**batch, labels=batch["input_ids"]).loss
        loss.backward()
        del loss
        for k in batch:
            batch[k] = batch[k].cpu()
        torch.cuda.empty_cache()

    for h in hooks:
        h.remove()

    if use_grad_ckpt:
        model.gradient_checkpointing_disable()
    model.eval()
    ## model = model.float()
    model = model.cpu().float()
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    print(f"After Fisher collection: {torch.cuda.memory_allocated()/1024**3:.2f} GB GPU, "
         f"{torch.cuda.memory_reserved()/1024**3:.2f} GB reserved")

    # save this chunk
    if hbar_save_path is not None:
        chunk_path = Path(str(hbar_save_path) + f"_chunk{chunk_id}")
        torch.save(Hbars, chunk_path)
        print(f"[Hbar] Saved chunk {chunk_id} to {chunk_path}")

    return Hbars

# ============================================================

def whitening_obs(model_name, model, calib_loader, profiling_mat, ratio, dev, obs_damping=1e-5, obs_hdamping=1e-1, obs_scale=1.0, alpha=0.1,
                  select_by_loss=True, obs_batches=16, hbar_save_path=None, reuse_hbars=False):
    """
    Drop-in companion to whitening().

    Identical to whitening() except that after the truncated SVD the
    kept singular values are corrected with the OBS update before the
    low-rank factors are assembled.

    Extra arguments vs whitening():
      obs_damping     : diagonal damping for H_SS solve
      obs_scale       : step-size scalar for the correction delta
                        (pass -1 to use automatic dynamic scaling)
      select_by_loss  : if True, select kept indices by OBS sensitivity
                        score rather than by magnitude
      obs_batches     : number of backward passes for Fisher estimation
    """
    model.eval()
    if 'opt' in model_name:
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers

    print("Starting OBS-SVD compression (whitening + OBS correction)...")

    # ------------------------------------------------------------------
    # Step 1: collect SVD info for every layer
    # ------------------------------------------------------------------
    layer_svd_info = {}
    with torch.no_grad():
        for i in range(len(layers)):
            subset = find_layers(layers[i])
            for name, module in subset.items():
                W     = module.weight.data.float().to(dev)
                full_name = f"{'model.decoder.layers' if 'opt' in model_name else 'model.layers'}.{i}.{name}"
                if profiling_mat is not None:
                    # scaling_diag_matrix = profiling_mat[i][name].to(dev).float()
                    scaling_diag_matrix = profiling_mat[i][name].to(dev)
                    try:
                        scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
                    except Exception as e:
                        print("Warning: scaling_diag_matrix is not full rank!")
                        scaling_diag_matrix += 1e-6 * torch.eye(scaling_diag_matrix.shape[0]).to(dev)
                        scaling_matrix_inv  = torch.linalg.inv(scaling_diag_matrix)
                    scaling_diag_matrix = scaling_diag_matrix.float()
                    scaling_matrix_inv = scaling_matrix_inv.float()
                        

                    W_scale = torch.matmul(W, scaling_diag_matrix)
                else:
                    W_scale            = W
                    scaling_matrix_inv = None

                U, S, Vh = torch.linalg.svd(W_scale, full_matrices=False)
                num_s_after_trunc = int(W.shape[0] * W.shape[1] * ratio /
                                        (W.shape[0] + W.shape[1]))
                rank   = max(1, num_s_after_trunc)

                full_k = rank + int((S.numel() - rank) * alpha)

                layer_svd_info[full_name] = {
                    "U":                  U[:, :full_k].detach().cpu(),
                    "S":                  S.detach().cpu(),
                    "Vh":                 Vh[:full_k, :].detach().cpu(),
                    "rank":               rank,
                    "full_k":             full_k,
                    "scaling_matrix_inv": scaling_matrix_inv.cpu() if scaling_matrix_inv is not None else None,
                    "layer_idx":          i,
                    "layer_name":         name,
                }
                W = W_scale = U = S = Vh = None
                torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Step 2: collect spectral Fisher
    # ------------------------------------------------------------------
    print("Collecting spectral Fisher for OBS correction ...")

    # enable gradient checkpointing for large models
    model_params = sum(p.numel() for p in model.parameters())
    use_collect_fisher_large = model_params > 1e9

    if use_collect_fisher_large:
        print(f"Using memory-efficient Fisher collection for large model ({model_params/1e9:.1f}B params)")
        Hbars = collect_spectral_fisher_large(
            model, calib_loader, dev, layer_svd_info, num_batches=obs_batches,
            hbar_save_path=hbar_save_path, reuse_hbars=reuse_hbars)
    else:
        print(f"Using standard Fisher collection for smaller model ({model_params/1e9:.1f}B params)")
        Hbars = collect_spectral_fisher_small(
            model, calib_loader, dev, layer_svd_info, num_batches=obs_batches, 
            hbar_save_path=hbar_save_path, reuse_hbars=reuse_hbars)

    # ------------------------------------------------------------------
    # Step 3: apply OBS-corrected SVD layer by layer
    # ------------------------------------------------------------------
    print("Applying OBS-corrected SVD decomposition ...")
    with torch.no_grad():
        for i in tqdm(range(len(layers))):
            layer  = layers[i]
            subset = find_layers(layer)

            if "llama" in model_name or "vicuna" in model_name:
                svd_attn = SVD_LlamaAttention(config=model.config, ratio=ratio)
                svd_mlp  = SVD_LlamaMLP(
                    hidden_size=layer.hidden_size,
                    intermediate_size=model.config.intermediate_size,
                    hidden_act=model.config.hidden_act, ratio=ratio)
            elif "mistral" in model_name:
                svd_attn = SVD_MistralAttention(config=model.config, ratio=ratio)
                svd_mlp  = SVD_MistralMLP(config=model.config, ratio=ratio)
            elif 'opt' in model_name:
                svd_decoder = SVDOPTDecoderLayer(model.config, ratio=ratio)

            for name in subset:
                dtype     = subset[name].weight.dtype
                full_name = f"{'model.decoder.layers' if 'opt' in model_name else 'model.layers'}.{i}.{name}"
                info      = layer_svd_info[full_name]
                Hbar      = Hbars[full_name].to(dev)

                S    = info["S"].to(dev)
                U    = info["U"].to(dev)          # (out, full_k)
                Vh   = info["Vh"].to(dev)         # (full_k, in_scaled)
                rank = info["rank"]


                # --- OBS correction -------------------------------------------
                # This is the only change vs whitening():
                # instead of plain truncation  truc_s = S[:rank],  we call the
                # OBS update to get corrected singular values and (optionally
                # reordered) keep indices.
                sigma_new, keep_idx, _ = obs_update_singular_values_with_selection(
                    S=S, rank=rank, Hbar=Hbar,
                    damping=obs_damping, hdamping=obs_hdamping, do_update=True,
                    obs_scale=obs_scale, alpha=alpha, select_by_loss=select_by_loss, ratio=ratio)
                # --------------------------------------------------------------

                truc_s     = sigma_new                       # corrected, length rank
                truc_u     = U[:, keep_idx]                  # (out, rank)
                truc_v_raw = Vh[keep_idx, :]                 # (rank, in_scaled)

                if info["scaling_matrix_inv"] is not None:
                    scaling_matrix_inv = info["scaling_matrix_inv"].to(dev).float()
                    truc_v = torch.matmul(truc_v_raw, scaling_matrix_inv)
                else:
                    truc_v = truc_v_raw

                truc_sigma = torch.diag(truc_s)
                sqrtSigma  = torch.sqrt(truc_sigma)
                svd_u      = torch.matmul(truc_u,  sqrtSigma).cpu().to(dtype)
                svd_v      = torch.matmul(sqrtSigma, truc_v).cpu().to(dtype)

                # --- assign factors (identical to whitening()) ----------------
                if 'opt' in model_name:
                    if "q_proj" in name:
                        svd_decoder.self_attn.q_u_proj.weight.data = svd_u
                        svd_decoder.self_attn.q_v_proj.weight.data = svd_v
                        svd_decoder.self_attn.q_u_proj.bias.data   = layer.self_attn.q_proj.bias.data
                    elif "k_proj" in name:
                        svd_decoder.self_attn.k_u_proj.weight.data = svd_u
                        svd_decoder.self_attn.k_v_proj.weight.data = svd_v
                        svd_decoder.self_attn.k_u_proj.bias.data   = layer.self_attn.k_proj.bias.data
                    elif "v_proj" in name:
                        svd_decoder.self_attn.v_u_proj.weight.data = svd_u
                        svd_decoder.self_attn.v_v_proj.weight.data = svd_v
                        svd_decoder.self_attn.v_u_proj.bias.data   = layer.self_attn.v_proj.bias.data
                    elif "out_proj" in name:
                        svd_decoder.self_attn.out_u_proj.weight.data = svd_u
                        svd_decoder.self_attn.out_v_proj.weight.data = svd_v
                        svd_decoder.self_attn.out_u_proj.bias.data   = layer.self_attn.out_proj.bias.data
                    elif "fc1" in name:
                        svd_decoder.fc1_u_proj.weight.data = svd_u
                        svd_decoder.fc1_v_proj.weight.data = svd_v
                        svd_decoder.fc1_u_proj.bias.data   = layer.fc1.bias.data
                    elif "fc2" in name:
                        svd_decoder.fc2_u_proj.weight.data      = svd_u
                        svd_decoder.fc2_v_proj.weight.data      = svd_v
                        svd_decoder.fc2_u_proj.bias.data        = layer.fc2.bias.data
                        svd_decoder.self_attn_layer_norm        = layer.self_attn_layer_norm
                        svd_decoder.final_layer_norm            = layer.final_layer_norm
                        layers[i]                               = svd_decoder
                else:
                    if "q_proj" in name:
                        svd_attn.q_u_proj.weight.data = svd_u
                        svd_attn.q_v_proj.weight.data = svd_v
                    elif "k_proj" in name:
                        svd_attn.k_u_proj.weight.data = svd_u
                        svd_attn.k_v_proj.weight.data = svd_v
                    elif "v_proj" in name:
                        svd_attn.v_u_proj.weight.data = svd_u
                        svd_attn.v_v_proj.weight.data = svd_v
                    elif "o_proj" in name:
                        svd_attn.o_u_proj.weight.data = svd_u
                        svd_attn.o_v_proj.weight.data = svd_v
                        layer.self_attn               = svd_attn
                    elif "gate_proj" in name:
                        svd_mlp.gate_u_proj.weight.data = svd_u
                        svd_mlp.gate_v_proj.weight.data = svd_v
                    elif "down_proj" in name:
                        svd_mlp.down_u_proj.weight.data = svd_u
                        svd_mlp.down_v_proj.weight.data = svd_v
                    elif "up_proj" in name:
                        svd_mlp.up_u_proj.weight.data = svd_u
                        svd_mlp.up_v_proj.weight.data = svd_v
                        layer.mlp                     = svd_mlp

                Hbar = svd_u = svd_v = truc_u = truc_v = truc_s = truc_sigma = sqrtSigma = None
                torch.cuda.empty_cache()

            del layer
            torch.cuda.empty_cache()

        del Hbars, layer_svd_info
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        print(f"Compression complete: {torch.cuda.memory_allocated()/1024**3:.2f} GB GPU")



def collect_hbars_only(model_name, model, calib_loader, profiling_mat, ratio, 
                       dev, obs_batches=16, alpha=0.1,
                       hbar_save_path=None, reuse_hbars=False,
                       chunk_id=0, num_chunks=1):
    """
    Only performs Steps 1 and 2 of whitening_obs:
    - Step 1: collect SVD info for every layer
    - Step 2: collect spectral Fisher (Hbar)
    Saves Hbars to hbar_save_path and returns without applying compression.
    Used for parallel/chunked Fisher collection before compression.
    """
    model.eval()
    if 'opt' in model_name:
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers

    print("Starting Hbar collection only (Steps 1 and 2 of whitening_obs)...")

    # ------------------------------------------------------------------
    # Step 1: collect SVD info
    # ------------------------------------------------------------------
    layer_svd_info = {}
    with torch.no_grad():
        for i in range(len(layers)):
            subset = find_layers(layers[i])
            for name, module in subset.items():
                W = module.weight.data.float().to(dev)
                full_name = f"{'model.decoder.layers' if 'opt' in model_name else 'model.layers'}.{i}.{name}"
                if profiling_mat is not None:
                    scaling_diag_matrix = profiling_mat[i][name].to(dev)
                    try:
                        scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
                    except Exception:
                        scaling_diag_matrix += 1e-6 * torch.eye(
                            scaling_diag_matrix.shape[0], device=dev,
                            dtype=scaling_diag_matrix.dtype)
                        scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
                    scaling_diag_matrix = scaling_diag_matrix.float()
                    scaling_matrix_inv  = scaling_matrix_inv.float()
                    W_scale = torch.matmul(W, scaling_diag_matrix)
                else:
                    W_scale            = W
                    scaling_matrix_inv = None

                U, S, Vh = torch.linalg.svd(W_scale, full_matrices=False)
                num_s_after_trunc = int(W.shape[0] * W.shape[1] * ratio /
                                        (W.shape[0] + W.shape[1]))
                rank  = max(1, num_s_after_trunc)
                full_k = rank + int((S.numel() - rank) * alpha)

                print(f"rank {rank} / {full_k} / {S.numel()} for layer {full_name}")

                layer_svd_info[full_name] = {
                    "U":                  U[:, :full_k].detach().cpu(),
                    "S":                  S.detach().cpu(),
                    "Vh":                 Vh[:full_k, :].detach().cpu(),
                    "rank":               rank,
                    "full_k":             full_k,
                    "scaling_matrix_inv": scaling_matrix_inv.cpu() if scaling_matrix_inv is not None else None,
                    "layer_idx":          i,
                    "layer_name":         name,
                }
                W = W_scale = U = S = Vh = None
                torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Step 2: collect spectral Fisher
    # ------------------------------------------------------------------
    print("Collecting spectral Fisher for OBS correction ...")
    collect_spectral_fisher_in_chunks(
        model, calib_loader, dev, layer_svd_info,
        num_batches=obs_batches,
        hbar_save_path=hbar_save_path,
        chunk_id=chunk_id, num_chunks=num_chunks) 

    print("Hbar collection complete. Run with --reuse_hbars to use saved Hbars.")


# ============================================================

@torch.no_grad()
def profle_svdllm(name, model, calib_loader, dev):
    if "llama" in name or "mistral" in name or "vicuna" in name:
        layers = model.model.layers
    elif "opt" in name:
        layers = model.model.decoder.layers
    model = model.to(dev)
    print("Start obtaining the whitening matrix...")
    def hook(module, input, output):
        inp = input[0].detach().float()
        if inp.dim() == 2:
            inp = inp.unsqueeze(0)
        adds = torch.matmul(inp.transpose(1,2), inp)
        adds_sum = torch.sum(adds, dim=0)
        module.raw_scaling_diag_matrix += adds_sum
        del inp, adds, adds_sum
        torch.cuda.empty_cache()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.raw_scaling_diag_matrix = 0
            module.register_forward_hook(hook)
    for batch in tqdm(calib_loader):
        batch = {k: v.to(dev) for k, v in batch.items()}
        model(**batch)
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module._forward_hooks.clear()
    torch.cuda.empty_cache()
    model = model.cpu()
    for i in range(len(layers)):
        subset = find_layers(layers[i])
        for name in subset:
            subset[name].raw_scaling_diag_matrix = subset[name].raw_scaling_diag_matrix.cpu()
    profiling_mat = {}
    print("Start Cholesky Decomposition...")
    for i in tqdm(range(len(layers))):
        layer_profile = {}
        subset = find_layers(layers[i])
        for name in subset:
            raw_scaling_diag_matrix = subset[name].raw_scaling_diag_matrix.double().to(dev)
            try:
                scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix)
            except Exception as e:
                print("Warning: eigen scaling_diag_matrix is not positive!")
                eigenvalues = torch.linalg.eigvalsh(raw_scaling_diag_matrix)
                raw_scaling_diag_matrix += (- eigenvalues[0] + 1e-6) * torch.eye(raw_scaling_diag_matrix.shape[0]).to(dev)
                scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix)
                eigenvalues = None
                del eigenvalues
            layer_profile[name] = scaling_diag_matrix.cpu()
            scaling_diag_matrix = raw_scaling_diag_matrix = subset[name].raw_scaling_diag_matrix = None
            del scaling_diag_matrix, raw_scaling_diag_matrix, subset[name].raw_scaling_diag_matrix
            torch.cuda.empty_cache()
        profiling_mat[i] = layer_profile
    return profiling_mat


@torch.no_grad()
def profle_svdllm_low_resource(model_name, model, calib_loader, dev):
    if "opt" in model_name:
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        # model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
        if model.model.decoder.final_layer_norm is not None:
            model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
        if model.model.decoder.project_in is not None:
            model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
        if model.model.decoder.project_out is not None:
            model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    else:
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    print("len calib_loader:", len(calib_loader))

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (len(calib_loader), model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp.cpu()
            cache['i'] += 1
            if cache['attention_mask'] is None:
                cache['attention_mask'] = kwargs['attention_mask'].cpu()
                if "opt" not in model_name:
                    cache['position_ids'] = kwargs['position_ids'].cpu()
            else:
                cache['attention_mask'] = torch.cat((cache['attention_mask'], kwargs['attention_mask'].cpu()), dim=0)
                if "opt" not in model_name:
                    cache['position_ids'] = torch.cat((cache['position_ids'], kwargs['position_ids'].cpu()), dim=0)
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in calib_loader:
        try:
            batch = {k: v.to(dev) for k, v in batch.items()}
            model(**batch)
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    if "opt" in model_name:
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        # model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.cpu()

        if model.model.decoder.final_layer_norm is not None:
            model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.cpu()
        if model.model.decoder.project_in is not None:
            model.model.decoder.project_in = model.model.decoder.project_in.cpu()
        if model.model.decoder.project_out is not None:
            model.model.decoder.project_out = model.model.decoder.project_out.cpu()

        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    else:
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()
    outs = torch.zeros_like(inps)
    attention_masks = cache['attention_mask']
    if "opt" not in model_name:
        position_ids = cache['position_ids']
    profiling_mat = {}
    for i in tqdm(range(len(layers))):
        layer_profile = {}
        layer = layers[i].to(dev)
        subset = find_layers(layer)
        def hook(module, input, output):
            inp = input[0].detach().float()
            if inp.dim() == 2:
                inp = inp.unsqueeze(0)
            adds = torch.matmul(inp.transpose(1,2), inp)
            adds_sum = torch.sum(adds, dim=0)
            module.scaling_diag_matrix += adds_sum
            del inp, adds, adds_sum, output
            torch.cuda.empty_cache()
        handles = []
        for name in subset:
            subset[name].scaling_diag_matrix = 0
            handles.append(subset[name].register_forward_hook(hook))
        for j in range(inps.shape[0]):
            if "opt" not in model_name:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_masks[j].unsqueeze(0).to(dev), position_ids=position_ids[j].unsqueeze(0).to(dev))[0]
            else:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_masks[j].unsqueeze(0).to(dev))[0]
        for h in handles:
            h.remove()
        layer = layer.cpu()
        for name in subset:
            subset[name].scaling_diag_matrix = subset[name].scaling_diag_matrix.cpu()
        torch.cuda.empty_cache()
        for name in subset:
            raw_scaling_diag_matrix = subset[name].scaling_diag_matrix.double().to(dev)
            try:
                scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix)
            except Exception as e:
                print("Warning: eigen scaling_diag_matrix is not positive!")
                eigenvalues = torch.linalg.eigvalsh(raw_scaling_diag_matrix)
                raw_scaling_diag_matrix += (- eigenvalues[0] + 1e-6) * torch.eye(raw_scaling_diag_matrix.shape[0]).to(dev)
                scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix)
                eigenvalues = None
                del eigenvalues
            layer_profile[name] = scaling_diag_matrix.cpu()
            scaling_diag_matrix = raw_scaling_diag_matrix = subset[name].raw_scaling_diag_matrix = None
            del scaling_diag_matrix, raw_scaling_diag_matrix, subset[name].raw_scaling_diag_matrix
            torch.cuda.empty_cache()
        layers[i] = layer.cpu()
        profiling_mat[i] = layer_profile
        inps = outs
        torch.cuda.empty_cache()
    return profiling_mat


@torch.no_grad()
def whitening(model_name, model, profiling_mat, ratio, dev):
    model.eval()
    if 'opt' in model_name:
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers
    print("Start SVD decomposition after whitening...")
    for i in tqdm(range(len(layers))):
        layer = layers[i]
        subset = find_layers(layer)
        if "llama" in model_name or "vicuna" in model_name:
            svd_attn = SVD_LlamaAttention(config=model.config, ratio=ratio)
            svd_mlp = SVD_LlamaMLP(hidden_size=layer.hidden_size, intermediate_size=model.config.intermediate_size, hidden_act=model.config.hidden_act, ratio=ratio)
        elif "mistral" in model_name:
            svd_attn = SVD_MistralAttention(config=model.config, ratio=ratio)
            svd_mlp = SVD_MistralMLP(config=model.config, ratio=ratio)
        elif 'opt' in model_name:
            svd_decoder = SVDOPTDecoderLayer(model.config, ratio=ratio)
        for name in subset:
            W = subset[name].weight.data.float().to(dev)
            dtype = W.dtype
            if profiling_mat is not None:
                scaling_diag_matrix = profiling_mat[i][name].to(dev)
                try:
                    scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
                except Exception as e:
                    print("Warning: scaling_diag_matrix is not full rank!")
                    scaling_diag_matrix += 1e-6 * torch.eye(scaling_diag_matrix.shape[0]).to(dev)
                    scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
                scaling_diag_matrix = scaling_diag_matrix.float()
                scaling_matrix_inv = scaling_matrix_inv.float()
                W_scale = torch.matmul(W, scaling_diag_matrix)
            else: # if profiling_mat is None, skip scaling and set scaling_matrix_inv to None
                W_scale = W
                scaling_matrix_inv = None
            U, S, VT = torch.linalg.svd(W_scale, full_matrices=False)
            num_s_after_trunc = int(W.shape[0] * W.shape[1] * ratio / (W.shape[0] + W.shape[1]))
            truc_s = S[:num_s_after_trunc]
            truc_u = U[:, :num_s_after_trunc]
            if scaling_matrix_inv is not None:
                truc_v = torch.matmul(VT[:num_s_after_trunc, :], scaling_matrix_inv)
            else:
                truc_v = VT[:num_s_after_trunc, :]
            truc_sigma = torch.diag(truc_s)
            sqrtSigma = torch.sqrt(truc_sigma)
            svd_u = torch.matmul(truc_u, sqrtSigma).cpu().to(dtype)
            svd_v = torch.matmul(sqrtSigma, truc_v).cpu().to(dtype)

            if 'opt' in model_name:
                if "q_proj" in name:
                    svd_decoder.self_attn.q_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.q_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.q_u_proj.bias.data = layer.self_attn.q_proj.bias.data
                elif "k_proj" in name:
                    svd_decoder.self_attn.k_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.k_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.k_u_proj.bias.data = layer.self_attn.k_proj.bias.data
                elif "v_proj" in name:
                    svd_decoder.self_attn.v_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.v_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.v_u_proj.bias.data = layer.self_attn.v_proj.bias.data
                elif "out_proj" in name:
                    svd_decoder.self_attn.out_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.out_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.out_u_proj.bias.data = layer.self_attn.out_proj.bias.data
                elif "fc1" in name:
                    svd_decoder.fc1_u_proj.weight.data = svd_u
                    svd_decoder.fc1_v_proj.weight.data = svd_v
                    svd_decoder.fc1_u_proj.bias.data = layer.fc1.bias.data
                elif "fc2" in name:
                    svd_decoder.fc2_u_proj.weight.data = svd_u
                    svd_decoder.fc2_v_proj.weight.data = svd_v
                    svd_decoder.fc2_u_proj.bias.data = layer.fc2.bias.data
                    svd_decoder.self_attn_layer_norm = layer.self_attn_layer_norm
                    svd_decoder.final_layer_norm = layer.final_layer_norm
                    layers[i] = svd_decoder
            else:
                if "q_proj" in name:
                    svd_attn.q_u_proj.weight.data = svd_u
                    svd_attn.q_v_proj.weight.data = svd_v
                elif "k_proj" in name:
                    svd_attn.k_u_proj.weight.data = svd_u
                    svd_attn.k_v_proj.weight.data = svd_v
                elif "v_proj" in name:
                    svd_attn.v_u_proj.weight.data = svd_u
                    svd_attn.v_v_proj.weight.data = svd_v
                elif "o_proj" in name:
                    svd_attn.o_u_proj.weight.data = svd_u
                    svd_attn.o_v_proj.weight.data = svd_v
                    layer.self_attn = svd_attn
                elif "gate_proj" in name:
                    svd_mlp.gate_u_proj.weight.data = svd_u
                    svd_mlp.gate_v_proj.weight.data = svd_v
                elif "down_proj" in name:
                    svd_mlp.down_u_proj.weight.data = svd_u
                    svd_mlp.down_v_proj.weight.data = svd_v
                elif "up_proj" in name:
                    svd_mlp.up_u_proj.weight.data = svd_u
                    svd_mlp.up_v_proj.weight.data = svd_v
                    layer.mlp = svd_mlp
            W = W_scale = scaling_matrix_inv = scaling_diag_matrix = U = S = VT = truc_s = truc_u = truc_v = sqrtSigma = None
            del W, W_scale, scaling_matrix_inv, scaling_diag_matrix, U, S, VT, truc_s, truc_u, truc_v, sqrtSigma
        del layer
        torch.cuda.empty_cache()


@torch.no_grad()
def whitening_local_update(model_name, model, dataloader, profiling_mat, ratio, dev, direct_update=False):
    print("Start SVD decomposition then update...")
    use_cache = model.config.use_cache
    model.config.use_cache = False
    if "opt" in model_name:
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    else:
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
    # model.model.norm = model.model.norm.to(dev)
    # layers[0] = layers[0].to(dev)
    if "opt" not in model_name:
        model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (len(dataloader), model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            if cache['attention_mask'] is None:
                cache['attention_mask'] = kwargs['attention_mask']
                if "opt" not in model_name:
                    cache['position_ids'] = kwargs['position_ids']
            else:
                cache['attention_mask'] = torch.cat((cache['attention_mask'], kwargs['attention_mask']), dim=0)
                if "opt" not in model_name:
                    cache['position_ids'] = torch.cat((cache['position_ids'], kwargs['position_ids']), dim=0)
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    # model.model.embed_tokens = model.model.embed_tokens.cpu()
    # model.model.norm = model.model.norm.cpu()
    if "opt" in model_name: 
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    else:
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()
    outs = torch.zeros_like(inps)
    attention_masks = cache['attention_mask']
    if "opt" not in model_name:
        position_ids = cache['position_ids']
    for i in tqdm(range(len(layers))):
        layer = layers[i].to(dev)
        subset = find_layers(layer)
        gpts = {}
        if "llama" in model_name or "vicuna" in model_name:
            svd_attn = SVD_LlamaAttention(config=model.config, ratio=ratio)
            svd_mlp = SVD_LlamaMLP(hidden_size=layer.hidden_size, intermediate_size=model.config.intermediate_size, hidden_act=model.config.hidden_act, ratio=ratio)
        elif "mistral" in model_name:
            svd_attn = SVD_MistralAttention(config=model.config, ratio=ratio)
            svd_mlp = SVD_MistralMLP(config=model.config, ratio=ratio)
        elif 'opt' in model_name:
            svd_decoder = SVDOPTDecoderLayer(model.config, ratio=ratio)
        for name in subset:
            if profiling_mat is not None:
                scaling_diag_matrix = profiling_mat[i][name].to(dev)
            else:
                scaling_diag_matrix = None
            gpts[name] = local_update(subset[name], scaling_diag_matrix=scaling_diag_matrix, ratio=ratio, name=name, direct_update=direct_update)

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch_update_u(inp[0].data, out.data)
            return tmp
        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        if "opt" not in model_name:
            outs = layer(inps, attention_mask=attention_masks, position_ids=position_ids)[0]
        else:
            outs = layer(inps, attention_mask=attention_masks)[0]
        for h in handles:
            h.remove()
        for name in gpts:
            svd_u, svd_v = gpts[name].fasterprune()
            svd_u, svd_v = svd_u.to(dtype), svd_v.to(dtype)
            if 'opt' in model_name:
                if "q_proj" in name:
                    svd_decoder.self_attn.q_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.q_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.q_u_proj.bias.data = layer.self_attn.q_proj.bias.data
                elif "k_proj" in name:
                    svd_decoder.self_attn.k_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.k_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.k_u_proj.bias.data = layer.self_attn.k_proj.bias.data
                elif "v_proj" in name:
                    svd_decoder.self_attn.v_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.v_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.v_u_proj.bias.data = layer.self_attn.v_proj.bias.data
                elif "out_proj" in name:
                    svd_decoder.self_attn.out_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.out_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.out_u_proj.bias.data = layer.self_attn.out_proj.bias.data
                elif "fc1" in name:
                    svd_decoder.fc1_u_proj.weight.data = svd_u
                    svd_decoder.fc1_v_proj.weight.data = svd_v
                    svd_decoder.fc1_u_proj.bias.data = layer.fc1.bias.data
                elif "fc2" in name:
                    svd_decoder.fc2_u_proj.weight.data = svd_u
                    svd_decoder.fc2_v_proj.weight.data = svd_v
                    svd_decoder.fc2_u_proj.bias.data = layer.fc2.bias.data
                    svd_decoder.self_attn_layer_norm = layer.self_attn_layer_norm
                    svd_decoder.final_layer_norm = layer.final_layer_norm
                    layers[i] = svd_decoder
            else:
                if "q_proj" in name:
                    svd_attn.q_u_proj.weight.data = svd_u
                    svd_attn.q_v_proj.weight.data = svd_v
                elif "k_proj" in name:
                    svd_attn.k_u_proj.weight.data = svd_u
                    svd_attn.k_v_proj.weight.data = svd_v
                elif "v_proj" in name:
                    svd_attn.v_u_proj.weight.data = svd_u
                    svd_attn.v_v_proj.weight.data = svd_v
                elif "o_proj" in name:
                    svd_attn.o_u_proj.weight.data = svd_u
                    svd_attn.o_v_proj.weight.data = svd_v
                    layer.self_attn = svd_attn
                elif "gate_proj" in name:
                    svd_mlp.gate_u_proj.weight.data = svd_u
                    svd_mlp.gate_v_proj.weight.data = svd_v
                elif "down_proj" in name:
                    svd_mlp.down_u_proj.weight.data = svd_u
                    svd_mlp.down_v_proj.weight.data = svd_v
                elif "up_proj" in name:
                    svd_mlp.up_u_proj.weight.data = svd_u
                    svd_mlp.up_v_proj.weight.data = svd_v
                    layer.mlp = svd_mlp
        layer = layer.to(dev)
        if "opt" not in model_name:
            outs = layer(inps, attention_mask=attention_masks, position_ids=position_ids)[0]
        else:
            outs = layer(inps, attention_mask=attention_masks)[0]
        layers[i] = layer.cpu()
        del gpts
        torch.cuda.empty_cache()
        inps = outs
        outs = None
        del outs
    model.config.use_cache = use_cache


class local_update:
    def __init__(self, layer, scaling_diag_matrix, ratio, name, direct_update=False):
        self.layer = layer
        self.name = name
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        if direct_update:
            self.U, self.S, self.VT = torch.linalg.svd(W.data, full_matrices=False)
        else:
            try:
                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
            except Exception as e:
                print("Warning: scaling_diag_matrix is not full rank!")
                scaling_diag_matrix += 1e-6 * torch.eye(scaling_diag_matrix.shape[0])
                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
            scaling_diag_matrix = scaling_diag_matrix.float()
            scaling_matrix_inv = scaling_matrix_inv.float()
            W_scale = torch.matmul(W, scaling_diag_matrix)
            self.U, self.S, self.VT = torch.linalg.svd(W_scale, full_matrices=False)
        num_s_after_trunc = int(W.shape[0] * W.shape[1] * ratio / (W.shape[0] + W.shape[1]))
        self.truc_s = self.S[:num_s_after_trunc].cuda()
        self.truc_u = self.U[:, :num_s_after_trunc].cuda()
        if direct_update:
            self.truc_v = self.VT[:num_s_after_trunc, :].cuda()
        else:
            self.truc_v = torch.matmul(self.VT[:num_s_after_trunc, :].cuda(), scaling_matrix_inv)
        self.truc_sigma = torch.diag(self.truc_s)
        self.new_w = torch.matmul(self.truc_u, torch.matmul(self.truc_sigma, self.truc_v[:num_s_after_trunc, :]))
        self.updated_err = self.error = 0

    def add_batch_update_u(self, inp, out):
        print(f"Start local update for {self.name}...")
        # handle both 2D (batch*seq, hidden) and 3D (batch, seq, hidden)
        if inp.dim() == 2:
            inps = inp
            outs = out
        else:
            inps = inp.view(inp.shape[0] * inp.shape[1], inp.shape[2])
            outs = out.view(out.shape[0] * out.shape[1], out.shape[2])
        
        new_w = torch.matmul(self.truc_u, torch.matmul(self.truc_sigma, self.truc_v))
        new_output = inps.matmul(new_w.t())
        self.error = torch.sqrt(torch.sum((outs - new_output)**2)).item() / torch.norm(outs, p='fro').item()
        x = torch.matmul(torch.matmul(inps, self.truc_v.T), self.truc_sigma)
        self.updated_uT = torch.linalg.lstsq(x, outs).solution
        updated_output = torch.matmul(torch.matmul(torch.matmul(inps, self.truc_v.T), self.truc_sigma), self.updated_uT)
        self.updated_error = torch.sqrt(torch.sum((outs - updated_output)**2)).item() / torch.norm(outs, p='fro').item()
        inps = outs = new_output = updated_output = x = new_w = None
        del inps, outs, new_output, updated_output, x, new_w
        torch.cuda.empty_cache()

    def fasterprune(self):
        sqrtSigma = torch.sqrt(self.truc_sigma)
        self.appendU = self.updated_uT.t().matmul(sqrtSigma)
        self.appendV = sqrtSigma.matmul(self.truc_v)
        return self.appendU, self.appendV


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='jeffwan/llama-7b-hf')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--ratio', type=float, default=0.2)
    parser.add_argument('--run_low_resource', action='store_true')
    parser.add_argument('--dataset', type=str, default='wikitext2')
    parser.add_argument('--whitening_nsamples', type=int, default=256)
    parser.add_argument('--updating_nsamples', type=int, default=16)
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--profiling_mat_path', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--DEV', type=str, default="cuda")
    parser.add_argument('--model_seq_len', type=int, default=2048)
    parser.add_argument('--eval_batch_size', type=int, default=4)
    parser.add_argument('--gen_seq_len', type=int, default=1024)
    parser.add_argument('--step', type=int, default=4)
    parser.add_argument('--lora', type=str, default=None)

    parser.add_argument('--obs_damping',     type=float, default=1e-5)
    parser.add_argument('--obs_scale',       type=float, default=1.0,
                        help='Step-size for OBS delta. Pass -1 for automatic scaling.')
    parser.add_argument('--obs_batches',     type=int,   default=512)
    parser.add_argument('--select_by_loss',  action='store_true')
    parser.add_argument('--obs_nsamples',    type=int,   default=512)
    parser.add_argument('--max_nsamples',    type=int,   default=512)
    parser.add_argument('--obs_hdamping',    type=float, default=1e-1)
    parser.add_argument('--hbar_save_path', type=str, default='hbars.pt')
    parser.add_argument('--reuse_hbars',  action='store_true')
    parser.add_argument('--hbar_chunk_id',    type=int, default=0,
                    help='Which chunk of batches to process (0-indexed)')
    parser.add_argument('--hbar_num_chunks',  type=int, default=1,
                        help='Total number of parallel chunks')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='Parameter for controlling the amount of singular values to drop')


    args = parser.parse_args()

    torch.manual_seed(args.seed)


    if args.step == 1:
        t_start_prune = time.time()
        model, tokenizer = get_model_from_huggingface(model_id=args.model)
        model = model.eval()
        if args.profiling_mat_path is None:
            cali_white_data = get_calib_train_data(args.dataset, tokenizer, args.whitening_nsamples, seqlen=args.model_seq_len)
            profiling_mat = profle_svdllm_low_resource(args.model, model, cali_white_data, args.DEV)
            if args.save_path is not None:
                torch.save(profiling_mat, args.save_path + "/" + args.model.replace("/", "_").replace("-", "_") + '_profiling_' + args.dataset + '_' + str(args.whitening_nsamples) + '_' + str(args.seed) + '.pt')
        elif args.profiling_mat_path == 'vanilla':
            profiling_mat = None
        else:
            profiling_mat = torch.load(args.profiling_mat_path)

        t_start_alg = time.time()
        whitening(args.model, model, profiling_mat, args.ratio, args.DEV)

        t_end = time.time()
        pruning_time = t_end - t_start_prune
        algorithm_time = t_end - t_start_alg
        if args.save_path is not None:

            model_save_path = get_svd_llm_save_path(args)
            torch.save({'model': model, 'tokenizer': tokenizer}, model_save_path)

            # save timing next to the model with matching name
            timing_path = model_save_path.replace('.pt', '_timing.json')
            with open(timing_path, 'w') as f:
                json.dump({
                    'pruning_time_seconds': pruning_time,
                    'algorithm_time_seconds': algorithm_time,
                     }, f)
            print(f"Pruning time: {pruning_time:.1f}s, Training time: {algorithm_time:.1f}s")

    elif args.step == 2:
        model, tokenizer = get_model_from_huggingface(model_id=args.model)
        dataloader, _ = get_loaders(args.dataset, nsamples=args.updating_nsamples, seed=args.seed, tokenizer=tokenizer, seqlen=args.model_seq_len)
        model = model.eval()
        model = model.float()
        if args.profiling_mat_path is None:
            cali_white_data = get_calib_train_data(args.dataset, tokenizer, args.whitening_nsamples, seqlen=args.model_seq_len)
            profiling_mat = profle_svdllm_low_resource(args.model, model, cali_white_data, args.DEV)
            if args.save_path is not None:
                torch.save(profiling_mat, args.save_path + "/" + args.model.replace("/", "_").replace("-", "_") + '_profiling_' + args.dataset + '_' + str(args.whitening_nsamples) + '_' + str(args.seed) + '.pt')
        else:
            profiling_mat = torch.load(args.profiling_mat_path)
        whitening_local_update(args.model, model, dataloader, profiling_mat, args.ratio, args.DEV)
        if args.save_path is not None:
            torch.save({'model': model, 'tokenizer': tokenizer}, args.save_path + "/" + args.model.replace("/", "_").replace("-", "_") + '_whitening_then_update_' + str(args.ratio) + '.pt')

    elif args.step == 6:
        t_start_prune = time.time()
        model, tokenizer = get_model_from_huggingface(model_id=args.model)
        model = model.eval()

        # whitening uses the first whitening_nsamples batches
        cali_white_data = list(get_calib_train_data(args.dataset, tokenizer, args.whitening_nsamples, seqlen=args.model_seq_len))

        # Fisher uses the first obs_nsamples batches (superset of whitening data)
        if args.reuse_hbars:
            cali_obs_data = None
        else:
            cali_obs_data = list(get_calib_train_data(args.dataset, tokenizer, args.obs_nsamples, seqlen=args.model_seq_len))

        if args.profiling_mat_path is None:
            profiling_mat = profle_svdllm_low_resource(args.model, model, cali_white_data, args.DEV)
            if args.save_path is not None:
                torch.save(profiling_mat, args.save_path + "/" + args.model.replace("/", "_").replace("-", "_") + '_profiling_' + args.dataset + '_' + str(args.whitening_nsamples) + '_' + str(args.seed) + '.pt')
        elif args.profiling_mat_path == 'vanilla':
            profiling_mat = None
        else:
            profiling_mat = torch.load(args.profiling_mat_path)

        t_start_alg = time.time()
        whitening_obs(
            args.model, model, cali_obs_data, profiling_mat, args.ratio, args.DEV, # profiling_mat
            obs_damping=args.obs_damping,
            obs_hdamping=args.obs_hdamping,
            obs_scale=args.obs_scale,
            alpha=args.alpha,
            select_by_loss=args.select_by_loss,
            obs_batches=args.obs_batches,
            hbar_save_path=args.hbar_save_path,
            reuse_hbars=args.reuse_hbars
        )

        t_end = time.time()
        pruning_time = t_end - t_start_prune
        algorithm_time = t_end - t_start_alg

        if args.save_path is not None:

            model_save_path = get_model_save_path(args)
            torch.save({'model': model, 'tokenizer': tokenizer}, model_save_path)

            # save timing next to the model with matching name
            timing_path = model_save_path.replace('.pt', '_timing.json')
            with open(timing_path, 'w') as f:
                json.dump({
                    'pruning_time_seconds': pruning_time,
                    'algorithm_time_seconds': algorithm_time,
                     }, f)
            print(f"Pruning time: {pruning_time:.1f}s, Training time: {algorithm_time:.1f}s")

    elif args.step == 7:
        # t_start = time.time()
        model, tokenizer = get_model_from_huggingface(model_id=args.model)
        model = model.eval()

        # whitening uses the first whitening_nsamples batches
        cali_white_data = list(get_calib_train_data(args.dataset, tokenizer, args.whitening_nsamples, seqlen=args.model_seq_len))

        # Fisher uses the first obs_nsamples batches (superset of whitening data)
        cali_obs_data = list(get_calib_train_data(args.dataset, tokenizer, args.obs_nsamples, seqlen=args.model_seq_len))

        if args.profiling_mat_path is None:
            profiling_mat = profle_svdllm_low_resource(args.model, model, cali_white_data, args.DEV)
            if args.save_path is not None:
                torch.save(profiling_mat, args.save_path + "/" + args.model.replace("/", "_").replace("-", "_") + '_profiling_' + args.dataset + '_' + str(args.whitening_nsamples) + '_' + str(args.seed) + '.pt')
        elif args.profiling_mat_path == 'vanilla':
            profiling_mat = None
        else:
            profiling_mat = torch.load(args.profiling_mat_path)

        t_start = time.time()

        collect_hbars_only(
            args.model, model, cali_obs_data, profiling_mat, args.ratio, args.DEV, # profiling_mat
            obs_batches=args.obs_batches,
            alpha=args.alpha,
            hbar_save_path=args.hbar_save_path,
            reuse_hbars=args.reuse_hbars,
            chunk_id=args.hbar_chunk_id, 
            num_chunks=args.hbar_num_chunks)

        t_end = time.time()

        hbar_compute_time = t_end - t_start

        model_save_path = get_model_save_path(args) 
        timing_path = model_save_path.replace('.pt', '_timing_hbar.json')
        with open(timing_path, 'w') as f:
            json.dump({'hbar_compute_time': hbar_compute_time}, f)
            
        print(f"Hbar collection complete in {t_end - t_start:.1f}s")

    elif args.step == 3:
        model, tokenizer = get_model_from_huggingface(args.model)
        model = model.eval().float()
        dataloader, _ = get_loaders(args.dataset, nsamples=args.updating_nsamples, seed=args.seed, tokenizer=tokenizer, seqlen=args.model_seq_len)
        whitening_local_update(model_name=args.model, model=model, dataloader=dataloader, profiling_mat=None, ratio=args.ratio, dev=args.DEV, direct_update=True)
        if args.save_path is not None:
            torch.save({'model': model, 'tokenizer': tokenizer}, args.save_path + "/" + args.model.replace("/", "_").replace("-", "_") + '_update_only_' + str(args.ratio) + '.pt')

    elif args.step >= 4:
        print(f"evaluating {args.model_path}...")

        if args.model_path == "original":
            model, tokenizer = get_model_from_huggingface(args.model)
        elif args.model_path == "svd_llm":
            model, tokenizer = get_model_from_local(get_svd_llm_save_path(args))
        elif args.model_path == "svd_surgeon":
            model, tokenizer = get_model_from_local(get_model_save_path(args))
        else:
            model, tokenizer = get_model_from_local(args.model_path)  # explicit path fallback
            if args.lora is not None:
                from utils.peft import PeftModel
                model = PeftModel.from_pretrained(model, args.lora, torch_dtype=torch.float16)
                model = model.merge_and_unload()
                torch.save({'model': model, 'tokenizer': tokenizer}, args.lora + '/merge.pt')


        model.eval()
        model = model.float().to(args.DEV)
        if args.step == 4:
            ppls = ppl_eval(model, tokenizer, datasets=[args.dataset], model_seq_len=args.model_seq_len, batch_size=args.eval_batch_size, device=args.DEV)
            save_experiment(args, ppls) 
        elif args.step == 5:
            eff_eval(model, tokenizer, datasets=args.dataset, generated_len=args.gen_seq_len, batch_size=args.eval_batch_size, device=args.DEV)