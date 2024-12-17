# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# utilities needed for the inference
# --------------------------------------------------------
import torch
import torch.nn.functional as F
import numpy as np

from .utils.device import to_cpu, collate_with_cat
from .utils.misc import invalid_to_nans, invalid_to_zeros
from .utils.geometry import depthmap_to_pts3d, geotrf, inv
from .utils.device import MyNvtxRange


def loss_of_one_batch_multiview(batch, model, criterion, device, 
                                symmetrize_batch=False, use_amp=False, 
                                ret=None, ref_id=-1):
    views = batch
    for view in views:
        for name in 'depthmap_img pointmap_img img pts3d pts3d_cam valid_mask camera_pose camera_intrinsics F_matrix corres'.split():  # pseudo_focal
            if name not in view:
                continue
            view[name] = view[name].to(device, non_blocking=True)
    if ref_id == -1:
        ref_id = (len(views)-1)//2

    with torch.cuda.amp.autocast(enabled=bool(use_amp)):
        preds = model(views, ref_id=ref_id)
        assert len(preds) == len(views)
        
        with torch.cuda.amp.autocast(enabled=False):
            if criterion is None:
                loss = None
            else:
                loss = criterion(views, preds, ref_id=ref_id)
    
    result = dict(views=views, preds=preds, loss=loss)
    for i in range(len(preds)):
        result[f'pred{i+1}'] = preds[i]
        result[f'view{i+1}'] = views[i]
    return result[ret] if ret else result


def loss_of_one_batch_multiview_pseudo_conf_sigmoid(batch, model, criterion, device, 
                                symmetrize_batch=False, use_amp=False, 
                                ret=None, ref_id=-1):
    views = batch
    for view in views:
        for name in 'depthmap_img pointmap_img img pts3d pts3d_cam valid_mask camera_pose camera_intrinsics F_matrix corres'.split():  # pseudo_focal
            if name not in view:
                continue
            view[name] = view[name].to(device, non_blocking=True)
    if ref_id == -1:
        ref_id = (len(views)-1)//2
    # if symmetrize_batch:
    #     view1, view2 = make_batch_symmetric([view1,view2])
    #     view3, _ = make_batch_symmetric([view3, view3])
    all_loss = [0, {}]
    with torch.cuda.amp.autocast(enabled=bool(use_amp)):
        preds = model(views, ref_id=ref_id, pseudo_conf=True)
        assert len(preds) == len(views)
        for i,pred in enumerate(preds):
            if i == ref_id:
                continue
            patch_pseudo_conf = pred['pseudo_conf']  # (B,S)
            # print(patch_pseudo_conf)
            true_conf = (pred['conf']-1.).mean(dim=(1,2))   # (B,)  mean(exp(x))
            # print(patch_pseudo_conf[0][:10],patch_true_conf[0][:10])
            pseudo_conf = torch.exp(patch_pseudo_conf).mean(dim=1) # (B,)  mean(exp(batch(x)))
            # print(pseudo_conf, true_conf)
            pseudo_conf = pseudo_conf / (1+pseudo_conf)
            true_conf = true_conf / (1+true_conf)
            # dis = (pseudo_conf-true_conf)**2
            dis = torch.abs(pseudo_conf-true_conf)
            loss = dis.mean()
            #如何判断loss是否为inf
            # if loss.isinf():
            #     print(((patch_pseudo_conf-patch_true_conf)**2).max())
            all_loss[0] += loss
            all_loss[1][f'pseudo_conf_loss_{i}'] = loss
    
    result = dict(views=views, preds=preds, loss=all_loss)
    for i in range(len(preds)):
        result[f'pred{i+1}'] = preds[i]
        result[f'view{i+1}'] = views[i]
    return result[ret] if ret else result

def check_if_same_size(pairs):
    shapes1 = [img1['img'].shape[-2:] for img1, img2 in pairs]
    shapes2 = [img2['img'].shape[-2:] for img1, img2 in pairs]
    return all(shapes1[0] == s for s in shapes1) and all(shapes2[0] == s for s in shapes2)


def get_pred_pts3d(gt, pred, use_pose=False):
    if 'depth' in pred and 'pseudo_focal' in pred:
        try:
            pp = gt['camera_intrinsics'][..., :2, 2]
        except KeyError:
            pp = None
        pts3d = depthmap_to_pts3d(**pred, pp=pp)

    elif 'pts3d' in pred:
        # pts3d from my camera
        pts3d = pred['pts3d']

    elif 'pts3d_in_other_view' in pred:
        # pts3d from the other camera, already transformed
        assert use_pose is True
        return pred['pts3d_in_other_view']  # return!

    if use_pose:
        camera_pose = pred.get('camera_pose')
        assert camera_pose is not None
        pts3d = geotrf(camera_pose, pts3d)

    return pts3d



def get_multiview_scale(pts:list, valid:list, norm_mode='avg_dis'):
    # adpat from DUSt3R
    for i in range(len(pts)):
        assert pts[i].ndim >= 3 and pts[i].shape[-1] == 3
    assert len(pts) == len(valid)
    norm_mode, dis_mode = norm_mode.split('_')

    if norm_mode == 'avg':
        # gather all points together (joint normalization)
        all_pts = []
        all_nnz = 0
        for i in range(len(pts)):
            nan_pts, nnz = invalid_to_zeros(pts[i], valid[i], ndim=3) 
            # print(nnz,nan_pts.shape) #(B,) (B,H*W,3)
            all_pts.append(nan_pts)
            all_nnz += nnz
        all_pts = torch.cat(all_pts, dim=1)
        # compute distance to origin
        all_dis = all_pts.norm(dim=-1) 
        if dis_mode == 'dis':
            pass  # do nothing
        elif dis_mode == 'log1p':
            all_dis = torch.log1p(all_dis)
        else:
            raise ValueError(f'bad {dis_mode=}')

        norm_factor = all_dis.sum(dim=1) / (all_nnz + 1e-8)    
    else: 
        raise ValueError(f'bad {norm_mode=}')

    norm_factor = norm_factor.clip(min=1e-8)
    while norm_factor.ndim < pts[0].ndim:
        norm_factor.unsqueeze_(-1)
    # print(norm_factor.shape)
    # print('norm factor:', norm_factor)
    return norm_factor


def get_multiview_shift_scale(pts:list, valid:list):
    #pts: list of tensor, (B, H, W, 3)
    #valid: list of tensor, (B, H, W)
    for i in range(len(pts)):
        assert pts[i].ndim >= 3 and pts[i].shape[-1] == 3
    assert len(pts) == len(valid)
    
    # gather all points together (joint normalization)
    all_pts = []
    all_valid_mask = []
    for i in range(len(pts)):
        all_pts.append(pts[i])
        all_valid_mask.append(valid[i])
    all_pts = torch.stack(all_pts, dim=1)  # (B, V, H, W, 3)
    all_valid_mask = torch.stack(all_valid_mask, dim=1)  # (B, V, H, W)
    all_shift = []
    all_scale = [] 
    for i in range(all_pts.shape[0]):
        valid_pts = all_pts[i][all_valid_mask[i]]  # (N, 3)
        # in case of no valid points
        if valid_pts.shape[0] == 0:
            all_shift.append(torch.zeros(3, device=all_pts.device))
            all_scale.append(torch.tensor(1, device=all_pts.device))
            continue
        #make median x,y,z as the center
        shift = torch.median(valid_pts, dim=0).values # (3,)
        shifted_pts = valid_pts - shift 
        
        #make the median distance as the scale
        dis = shifted_pts.norm(dim=-1) # (N,)
        # scale = torch.median(dis, dim=0).values # (,)
        scale = torch.mean(dis) # (,)
        all_shift.append(shift)
        all_scale.append(scale)
    
    all_shift = torch.stack(all_shift, dim=0)[:,None,None]  # (B, 1, 1, 3)
    all_scale = torch.stack(all_scale, dim=0)[:,None,None,None]  # (B, 1, 1, 1)
    return all_shift, all_scale


def generate_mask(B, H, W, mask_ratio, patch_size, valid_mask=None, valid_mask_ratio=0.15, device='cuda'):
    pH, pW = H//patch_size, W//patch_size
    num_patches = pH*pW
    num_patch_masks = int(num_patches * mask_ratio)
    scores = torch.rand((B, num_patches), dtype=torch.float, device=device)
    indices = torch.argsort(scores, dim=1) # B, pH*pW
    masks = indices < num_patch_masks
    
    if valid_mask is not None:
        assert valid_mask.shape[1] == H and valid_mask.shape[2] == W
        invalid_mask = ~valid_mask  # B,H,W
        invalid_mask = invalid_mask.unsqueeze(1).float()  # B,1,H,W
        patch_invalid_ratio = F.avg_pool2d(invalid_mask, 
                                           kernel_size=patch_size, stride=patch_size)
        patch_invalid_mask = (patch_invalid_ratio > valid_mask_ratio).reshape(B, -1) # B, pH*pW  
        masks = masks | patch_invalid_mask
    
    masks = masks.reshape(B, pH, pW)
    
    return masks

def loss_of_one_batch_masked_multiref_multisrc(batch, model, criterion, device, 
                                symmetrize_batch=False, use_amp=False, 
                                ret=None, ref_id=-1,cam_id=0, exclude_cam=True,
                                mask_ratio=0., mask_invalid_ratio=1., to_zero=False):
    views = batch
    for view in views:
        for name in 'img pts3d pts3d_cam valid_mask camera_pose'.split():  # pseudo_focal
            if name not in view:
                continue
            view[name] = view[name].to(device, non_blocking=True)
    
    with MyNvtxRange("preprocess"):
        if cam_id == -1:
            # ramdomly select a camera as the target camera
            cam_id = np.random.randint(0, len(views))
        # print(cam_id)
        c2w = views[cam_id]['camera_pose']  # not so resonable..
        w2c = inv(c2w) 
        if exclude_cam:
            views.pop(cam_id)
        
        if ref_id == -1:
            ref_id = [i for i in range(len(views)-1)] # all views except the last one
        elif -2 in ref_id:
            #随机选择一半的视角作为参考视角
            ref_id = np.random.choice(len(views), len(views)//2, replace=False).tolist()
        # print(ref_id)
        for id in ref_id:
            views[id]['pts3d_world'] = geotrf(w2c, views[id]['pts3d'])  #转移到目标坐标系
        norm_factor_world = get_multiview_scale([views[id]['pts3d_world'] for id in ref_id],
                                                [views[id]['valid_mask'] for id in ref_id], 
                                                norm_mode='avg_dis')
        model_wo_ddp = model.module if hasattr(model, 'module') else model
        assert model_wo_ddp.patch_size[0] == model_wo_ddp.patch_size[1]
        patch_size = model_wo_ddp.patch_size[0]
        for id,view in enumerate(views):            
            if id in ref_id:
                view['pts3d_world'] = view['pts3d_world'].permute(0,3,1,2) / norm_factor_world
                # view['patch_mask'] = generate_mask(*view['valid_mask'].shape, mask_ratio=0.2, 
                                            #    patch_size=patch_size, valid_mask=view['valid_mask'])
            else:
                norm_factor_src = get_multiview_scale([view['pts3d_cam']],
                                                    [view['valid_mask']], 
                                                    norm_mode='avg_dis')
                view['pts3d_cam'] = view['pts3d_cam'].permute(0,3,1,2) / norm_factor_src
                # view['patch_mask'] = generate_mask(*view['valid_mask'].shape, mask_ratio=1, 
                #                                patch_size=patch_size, valid_mask=view['valid_mask'])
            if mask_ratio > 0 or mask_invalid_ratio < 1:
                view['patch_mask'] = generate_mask(*view['valid_mask'].shape, mask_ratio=mask_ratio, 
                                                patch_size=patch_size, valid_mask=view['valid_mask'],
                                                valid_mask_ratio=mask_invalid_ratio)
    if to_zero:    
        for id,view in enumerate(views):
            valid_mask = view['valid_mask'].unsqueeze(1).float() # B,1,H,W
            if id in ref_id:
                # print(view['pts3d_world'].shape, valid_mask.shape, (-valid_mask+1).sum())
                view['pts3d_world'] = view['pts3d_world'] * valid_mask
            else:
                view['pts3d_cam'] = view['pts3d_cam'] * valid_mask

    with MyNvtxRange("forward"):
        with torch.cuda.amp.autocast(enabled=bool(use_amp)):
            preds = model(views, ref_ids=ref_id)
            assert len(preds) == len(views)
            with torch.cuda.amp.autocast(enabled=False):
                if criterion is None:
                    loss = None
                else:
                    loss = criterion(views, preds, ref_id=ref_id, ref_camera=w2c, norm_scale=norm_factor_world)   
                    # loss = criterion(views, preds, ref_id=src_id, ref_camera=w2c)   

    result = dict(views=views, preds=preds, loss=loss)
    for i in range(len(preds)):
        result[f'pred{i+1}'] = preds[i]
        result[f'view{i+1}'] = views[i]
    return result[ret] if ret else result





