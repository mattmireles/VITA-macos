"""
VLMEvalKit Inference Engine - Distributed inference for vision-language models.

This module orchestrates large-scale inference across multiple vision-language
models and datasets. It provides distributed processing capabilities for efficient
evaluation of VITA and other VLM models on benchmark datasets.

Core Functionality:
- Distributed inference across multiple GPUs and processes
- API-based model inference with rate limiting and error handling
- Dataset batching and progress tracking for large evaluations
- Model result caching and resumption of interrupted evaluations

Called by:
- Benchmark evaluation scripts for systematic model assessment
- Research pipelines for comparative model analysis
- Production evaluation systems for model validation
- Academic benchmark submission workflows

This inference engine supports:
- VITA multimodal models with vision, audio, and text processing
- API-based models with rate limiting and timeout handling
- Local GPU-based models with distributed processing
- Custom prompt formatting for dataset-specific evaluation

Flow continues to:
- Model-specific inference implementations
- Result aggregation and metric calculation
- Performance analysis and comparison reports
- Benchmark leaderboard submissions

Optimization Features:
- Parallel processing across multiple workers
- Intelligent batching for memory optimization
- Progress tracking and resumption capabilities
- Error handling and retry mechanisms
"""

import torch
import torch.distributed as dist
from vlmeval.config import supported_VLM
from vlmeval.utils import track_progress_rich
from vlmeval.smp import *

# Error message constant for API failures
FAIL_MSG = 'Failed to obtain answer via API.'


def parse_args():
    """
    Parse command line arguments for distributed inference configuration.
    
    Configures inference parameters including datasets, models, and
    processing settings for large-scale VLM evaluation workflows.
    
    Called by:
    - Main inference script initialization
    - Distributed evaluation launch scripts
    - Benchmark evaluation workflows
    
    Returns:
        argparse.Namespace: Configuration containing:
            - data: List of datasets to evaluate
            - model: List of models to test
            - nproc: Number of processes for distributed inference
            - verbose: Enable detailed logging output
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, nargs='+', required=True)
    parser.add_argument('--model', type=str, nargs='+', required=True)
    parser.add_argument('--nproc', type=int, default=4, required=True)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    return args


def infer_data_api(work_dir, model_name, dataset, index_set=None, api_nproc=4, ignore_failed=False):
    """
    Execute API-based inference for vision-language model evaluation.
    
    This function handles API-based model inference with proper error handling,
    rate limiting, and result caching. Used for evaluating models that provide
    API endpoints rather than direct model access.
    
    Called by:
    - Main evaluation workflows for API-based models
    - Distributed inference systems for remote model access
    - Benchmark evaluation pipelines with API model support
    
    Args:
        work_dir (str): Working directory for result storage
        model_name (str): Name of API model to evaluate
        dataset: Dataset object containing evaluation samples
        index_set (list, optional): Subset of indices to process
        api_nproc (int): Number of concurrent API requests
        ignore_failed (bool): Continue on API failures
    
    Returns:
        Results from API-based model inference
        Includes proper error handling and retry mechanisms
    """
    rank, world_size = get_rank_and_world_size()
    assert rank == 0 and world_size == 1
    dataset_name = dataset.dataset_name
    data = dataset.data
    if index_set is not None:
        data = data[data['index'].isin(index_set)]

    model = supported_VLM[model_name]() if isinstance(model_name, str) else model_name
    # Ensure model supports API-based inference
    assert getattr(model, 'is_api', False), f"Model {model_name} does not support API inference"

    lt, indices = len(data), list(data['index'])

    structs = []
    for i in range(lt):
        item = data.iloc[i]
        if hasattr(model, 'use_custom_prompt') and model.use_custom_prompt(dataset_name):
            assert hasattr(model, 'build_prompt')
            struct = model.build_prompt(item, dataset=dataset_name)
        else:
            struct = dataset.build_prompt(item)
        structs.append(struct)

    # structs = [dataset.build_prompt(data.iloc[i]) for i in range(lt)]

    out_file = f'{work_dir}/{model_name}_{dataset_name}_supp.pkl'
    res = {}
    if osp.exists(out_file):
        res = load(out_file)
        if ignore_failed:
            res = {k: v for k, v in res.items() if FAIL_MSG not in v}

    structs = [s for i, s in zip(indices, structs) if i not in res]
    indices = [i for i in indices if i not in res]

    gen_func = model.generate
    structs = [dict(message=struct, dataset=dataset_name) for struct in structs]

    if len(structs):
        track_progress_rich(gen_func, structs, nproc=api_nproc, chunksize=api_nproc, save=out_file, keys=indices)

    res = load(out_file)
    if index_set is not None:
        res = {k: v for k, v in res.items() if k in index_set}
    os.remove(out_file)
    return res


def infer_data(model_name, work_dir, dataset, out_file, verbose=False, api_nproc=4):
    dataset_name = dataset.dataset_name
    prev_file = f'{work_dir}/{model_name}_{dataset_name}_PREV.pkl'
    res = load(prev_file) if osp.exists(prev_file) else {}
    if osp.exists(out_file):
        res.update(load(out_file))

    rank, world_size = get_rank_and_world_size()
    sheet_indices = list(range(rank, len(dataset), world_size))
    lt = len(sheet_indices)
    data = dataset.data.iloc[sheet_indices]
    data_indices = [i for i in data['index']]

    # If finished, will exit without building the model
    all_finished = True
    for i in range(lt):
        idx = data.iloc[i]['index']
        if idx not in res:
            all_finished = False
    if all_finished:
        res = {k: res[k] for k in data_indices}
        dump(res, out_file)
        return

    # Data need to be inferred
    data = data[~data['index'].isin(res)]
    lt = len(data)

    model = supported_VLM[model_name]() if isinstance(model_name, str) else model_name

    is_api = getattr(model, 'is_api', False)
    if is_api:
        lt, indices = len(data), list(data['index'])
        supp = infer_data_api(
            work_dir=work_dir,
            model_name=model_name,
            dataset=dataset,
            index_set=set(indices),
            api_nproc=api_nproc)
        for idx in indices:
            assert idx in supp
        res.update(supp)
        res = {k: res[k] for k in data_indices}
        dump(res, out_file)
        return model_name
    else:
        model.set_dump_image(dataset.dump_image)

    for i in tqdm(range(lt)):
        idx = data.iloc[i]['index']
        if idx in res:
            continue

        if hasattr(model, 'use_custom_prompt') and model.use_custom_prompt(dataset_name):
            struct = model.build_prompt(data.iloc[i], dataset=dataset_name)
        else:
            struct = dataset.build_prompt(data.iloc[i])

        response = model.generate(message=struct, dataset=dataset_name)
        torch.cuda.empty_cache()

        if verbose:
            print(response, flush=True)

        res[idx] = response
        if (i + 1) % 20 == 0:
            dump(res, out_file)

    res = {k: res[k] for k in data_indices}
    dump(res, out_file)
    return model


# A wrapper for infer_data, do the pre & post processing
def infer_data_job(model, work_dir, model_name, dataset, verbose=False, api_nproc=4, ignore_failed=False):
    rank, world_size = get_rank_and_world_size()
    dataset_name = dataset.dataset_name
    result_file = osp.join(work_dir, f'{model_name}_{dataset_name}.xlsx')

    prev_file = f'{work_dir}/{model_name}_{dataset_name}_PREV.pkl'
    if osp.exists(result_file):
        if rank == 0:
            data = load(result_file)
            results = {k: v for k, v in zip(data['index'], data['prediction'])}
            if not ignore_failed:
                results = {k: v for k, v in results.items() if FAIL_MSG not in str(v)}
            dump(results, prev_file)
        if world_size > 1:
            dist.barrier()

    tmpl = osp.join(work_dir, '{}' + f'{world_size}_{dataset_name}.pkl')
    out_file = tmpl.format(rank)

    model = infer_data(
        model, work_dir=work_dir, dataset=dataset, out_file=out_file, verbose=verbose, api_nproc=api_nproc)
    if world_size > 1:
        dist.barrier()

    if rank == 0:
        data_all = {}
        for i in range(world_size):
            data_all.update(load(tmpl.format(i)))

        data = dataset.data
        for x in data['index']:
            assert x in data_all
        data['prediction'] = [str(data_all[x]) for x in data['index']]
        if 'image' in data:
            data.pop('image')

        dump(data, result_file)
        for i in range(world_size):
            os.remove(tmpl.format(i))
    return model
