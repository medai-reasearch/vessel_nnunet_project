from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, StreamingResponse
import numpy as np
import cupy as cp
import tempfile, os
from tqdm import tqdm
import sys
from .gpu_paths import get_labeled_endpoints_gpu, gpu_parallel_flood_fill_limited, build_cugraph_optimized, find_grouped_paths
from .cv import region_grow_cupy
from scipy.ndimage import label
import copy
import json
import io

app = FastAPI(title="GPU‑Centerline Service")

@app.post("/region-grow")
async def region_grow(
    file: UploadFile = File(...),
    seeds: str = Form(...),
    threshold: float = Form(...)
):
    np_bytes = await file.read()
    np_image = np.load(io.BytesIO(np_bytes))
    cp_image = cp.array(np_image)
    seeds_list = json.loads(seeds)

    cp_result = region_grow_cupy(cp_image, seeds_list, threshold)
    result_np = cp.asnumpy(cp_result)

    buffer = io.BytesIO()
    np.save(buffer, result_np)
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="application/octet-stream")

@app.post("/centerline")
async def calc_centerline(
    skel_file: UploadFile = File(...),         # .npy  (Z,Y,X)  uint8/bool
    ep_file:   UploadFile = File(...),         # .npy  (N,3)    int64
):
    # --- 1. 파일을 임시 저장 후 로드 ---
    with tempfile.TemporaryDirectory() as td:
        skel_path = os.path.join(td, "skel.npy")
        ep_path   = os.path.join(td, "ep.npy")
        with open(skel_path, "wb") as f:
            f.write(await skel_file.read())
        with open(ep_path, "wb") as f:
            f.write(await ep_file.read())

        skel = np.load(skel_path)
        background = np.load(ep_path)

    # --- 2. GPU 연산 ---
    cp_skel = cp.asarray(skel)
    costmap = background
    costmap = costmap.astype(np.uint8)

    cp_costmap = cp.asarray(costmap)
    group_dict, endpoints = get_labeled_endpoints_gpu(cp_skel)
    result_connections = []
    for key, seeds in tqdm(group_dict.items(), file=sys.stderr, ascii=True):
        tqdm.write(f"[{key}] processing...")
        # 같은 그룹 seed 는 target 에서 제외
        seeds_set = set(seeds)
        targets   = [pt for pt in endpoints if pt not in seeds_set]

        # 최단-연결
        connections = gpu_parallel_flood_fill_limited(cp_costmap, seeds, targets, 32)
        result_connections.extend(connections)
    
    return JSONResponse(content={"paths": result_connections})

# def add_connections_to_skeleton(skeleton: np.ndarray, connections: list) -> np.ndarray:
#     connected_skeleton = skeleton.copy()
#     all_coords = [pt for path in connections for pt in path]

#     if all_coords:
#         coords = np.array(all_coords, dtype=np.int32)  # (N, 3) shape
#         z, y, x = coords[:, 0], coords[:, 1], coords[:, 2]
#         connected_skeleton[z, y, x] = 1


#     # labeling
#     skeleton_labels, _ = label(connected_skeleton > 0, structure=np.ones((3, 3, 3)))

#     # target_value가 있는 위치의 label 값만 추출
#     label_ids = np.unique(skeleton_labels[connected_skeleton == 2])
#     label_ids = label_ids[label_ids > 0]  # label 0 제외

#     change_mask = np.isin(skeleton_labels, [label_ids])
#     connected_skeleton[change_mask == 1] = 2

#     return connected_skeleton
def add_connections_to_skeleton(skeleton: np.ndarray, connections: list) -> np.ndarray:
    connected_skeleton = skeleton.copy()
    all_coords = [pt for path in connections for pt in path]
    # 1. 연결 좌표에 값 2 할당
    if all_coords:
        coords = np.array(all_coords, dtype=np.int32)
        z, y, x = coords[:, 0], coords[:, 1], coords[:, 2]
        connected_skeleton[z, y, x] = 2
    
        # 2. 라벨링 (0은 제외)
        skeleton_labels, _ = label(connected_skeleton > 0, structure=np.ones((3, 3, 3)))
        
        # 3. 딱 한번만 탐색해서 label ID 추출
        coords = np.array(all_coords, dtype=np.int32)
        labeled_values = skeleton_labels[coords[:, 0], coords[:, 1], coords[:, 2]]
        label_ids = np.unique(labeled_values[labeled_values > 0])

        # 4. label ID를 가진 영역만 2로 변경
        mask = np.isin(skeleton_labels, label_ids)
        connected_skeleton[(mask) & (skeleton_labels > 0)] = 2

    return connected_skeleton

@app.post("/cugraph")
async def calc_centerline(
    skel_file: UploadFile = File(...),         # .npy  (Z,Y,X)  uint8/bool
    image_file: UploadFile = File(...),         # .npy  (Z,Y,X)  int32
    bg_file:   UploadFile = File(...),         # .npy  (N,3)    int64
    skel_threshold: int = Form(...)
):
    # --- 1. 파일을 임시 저장 후 로드 ---
    with tempfile.TemporaryDirectory() as td:
        skel_path = os.path.join(td, "skel.npy")
        image_path = os.path.join(td, "image.npy")
        bg_path   = os.path.join(td, "bg.npy")
        with open(skel_path, "wb") as f:
            f.write(await skel_file.read())
        with open(image_path, "wb") as f:
            f.write(await image_file.read())
        with open(bg_path, "wb") as f:
            f.write(await bg_file.read())

        skel = np.load(skel_path)
        image = np.load(image_path)
        background = np.load(bg_path)


    COST_THRESHOLD = 30 * int(np.mean(skel.shape))
    graph = build_cugraph_optimized(image, background, skel_threshold)

    target_skel = np.isin(skel, [2])
    target_skel = target_skel.astype(np.int32)
    
    candidate_skel = np.isin(skel, [1, 2])
    candidate_skel = candidate_skel.astype(np.int32)

    cp_target_skel = cp.asarray(target_skel)
    
    input_skel = skel.copy()
    

    group_dict, endpoints = get_labeled_endpoints_gpu(cp_target_skel)
    previous_endpoints = copy.deepcopy(endpoints)
    result_connections = find_grouped_paths(graph, candidate_skel, group_dict, COST_THRESHOLD)
    
    output_skel = add_connections_to_skeleton(input_skel, result_connections)
    
    count = 0
    while not np.array_equal(input_skel, output_skel):
        count += 1
        print(f"확장 가능한 좌표를 확인했습니다, 현재 반복횟수 = {count}")
        input_skel = output_skel.copy()

        target_skel = np.isin(input_skel, [2])
        target_skel = target_skel.astype(np.int32)
        cp_target_skel = cp.asarray(target_skel)

        candidate_skel = np.isin(input_skel, [1, 2])
        candidate_skel = candidate_skel.astype(np.int32)
        
        group_dict, endpoints = get_labeled_endpoints_gpu(cp_target_skel)
        
        if len(endpoints) == 0:
            group_dict[1] = []
        group_dict[1].extend(previous_endpoints)
        previous_endpoints = copy.deepcopy(endpoints)
        result_connections = find_grouped_paths(graph, candidate_skel, group_dict, COST_THRESHOLD)
        output_skel = add_connections_to_skeleton(input_skel, result_connections)


    result_list = output_skel.tolist()
    print(np.unique(output_skel))
    return JSONResponse(content={"paths": result_list})