import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import random
import scipy
import numpy as np
import math
from tqdm import tqdm
from copy import deepcopy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch.nn.parameter import Parameter
from sklearn.metrics import confusion_matrix
from seqeval.metrics import f1_score # 序列标注评估工具
from transformers import AutoTokenizer
from src.selective_ffn_distill import SelectiveFFNDistiller
from src.dataloader import *
from src.utils import *

logger = logging.getLogger()
params = get_params()
auto_tokenizer = AutoTokenizer.from_pretrained(params.model_name)
pad_token_label_id = nn.CrossEntropyLoss().ignore_index  # -100



class BaseTrainer(object):
    def __init__(self, params, model, label_list):
        # parameters
        self.params = params # 配置
        self.model = model 
        self.label_list = label_list
        self.past_tasks = []  # 新增：存储过去任务信息 [{'task_id': int, 'age': int, 'entity_embs': list of tensors}]
        
        
        # training
        self.lr = float(params.lr)
        self.mu = 0.9
        self.weight_decay = 5e-4
        self.layer_f1_scores = {}  # 记录每层跳过的 F1 分数
        self.layer_losses = {}    # 记录每层跳过的蒸馏损失
        self.memory_dataloader = None
        self.classifier_pool = []
        self.ffn_distiller = None  # 先占位
    

    def batch_forward(self, inputs):    
        # Compute features
        self.inputs = inputs # # (bsz, seq_len)
        self.all_features = self.model.encoder(self.inputs)
        # Compute logits 常规logits
        self.logits = self.model.forward_classifier(self.all_features[1][-1])   # (bsz, seq_len, output_dim) 


    def batch_loss(self, labels):
        '''
            Cross-Entropy Loss
        '''
        self.loss = 0
        assert self.logits!=None, "logits is none!"

        # classification loss
        ce_loss = nn.CrossEntropyLoss()(self.logits.view(-1, self.logits.shape[-1]), 
                                labels.flatten().long()) # bs*seq_len, out_dim 默认自动忽略-100 label （pad、cls、sep、第二子词对应的索引）
        self.loss = ce_loss
        return ce_loss.item() 

    def find_median(self, train_loader):
        
        bg_entropy_values_total = []
        bg_pseudo_labels_total = []
        for X, labels in train_loader: # 928*8*30
            X = X.cuda()
            labels = labels.cuda()

            with torch.no_grad():
                self.refer_model.eval()
                refer_features = self.refer_model.forward_encoder(X)
                refer_logits = self.refer_model.forward_classifier(refer_features)# (bsz,seq_len,refer_dims)


            mask_bg = labels == 0  # (bsz, seq_len)  找出背景位置
            probas = torch.softmax(refer_logits, dim=-1)  # (bsz,seq_len,refer_dims)

            _, pseudo_labels = probas.max(dim=-1) # 最大值 以及 最大值所在位置

            bg_entropy_values = entropy(probas)[mask_bg].view(-1)
            bg_entropy_values_total.extend(bg_entropy_values.detach().cpu().numpy().tolist())

            bg_pseudo_labels = pseudo_labels[mask_bg].view(-1) # bsz*seq_len
            bg_pseudo_labels_total.extend(bg_pseudo_labels.detach().cpu().numpy().tolist())

    
        bg_entropy_values_total = np.array(bg_entropy_values_total,dtype=np.float32)
        bg_pseudo_labels_total = np.array(bg_pseudo_labels_total, dtype=np.int32)
        thresholds = np.zeros(self.old_classes, dtype=np.float32) #old_classes
        base_threshold = self.params.threshold #0.001
        for c in range(len(thresholds)):
            thresholds[c] = np.median(bg_entropy_values_total[bg_pseudo_labels_total==c])
            thresholds[c] = max(thresholds[c], base_threshold)

        return torch.from_numpy(thresholds).cuda() 


    def before(self, train_loader):
        self.thresholds = self.find_median(train_loader)

    def calculate_sample_weight(self, labels): # labels (bsz, seq_len)
        background = labels == 0
        old_token = (labels < self.old_classes) & (labels != pad_token_label_id)
        old_token = old_token & (~background)
        new_token = labels >= self.old_classes
        old_token = torch.sum(old_token, 1)
        new_token = torch.sum(new_token, 1)
        new_token[new_token==0] = 1 # 某个样本没有新label 防止除0异常
        weight = 0.5 + F.sigmoid(old_token/new_token)
        return weight
    
    def enable_selective_distillation(self, old_dataloader):#选择神经元
        self.ffn_distiller = SelectiveFFNDistiller(
            model=self.model,
            topk_ratio=0.1,        # 保留 top 10% 神经元
            norm_type="l1"
        )
        self.ffn_distiller.assess_importance(old_dataloader, steps=100)  # 用100个batch评估
        logger.info("Selective FFN Distillation 已激活！")

#    def compute_ffn_distillation_loss(self, teacher_hidden_states, student_hidden_states, skip_layer=None):
#            mse_loss = nn.MSELoss(reduction='mean')
#            distill_ffn_loss = 0.0
#            layer_losses = {}
#            num_layers = 0
#            for layer in range(1, 13):  # 覆盖 12 个 Transformer 层
#                if layer == skip_layer:  # 跳过指定层
#                    continue
#                
#                teacher_hidden = teacher_hidden_states[layer]
#                student_hidden = student_hidden_states[layer]
##                if attention_mask is not None:
##                    mask = attention_mask.unsqueeze(-1).expand_as(teacher_hidden)
##                    teacher_hidden = teacher_hidden * mask
##                    student_hidden = student_hidden * mask
#                layer_loss = mse_loss(student_hidden, teacher_hidden)
#                distill_ffn_loss += layer_loss
#                layer_losses[layer] = layer_loss.item()
#                num_layers += 1
#            distill_ffn_loss = distill_ffn_loss / num_layers if num_layers > 0 else 0.0
#            return distill_ffn_loss, layer_losses
    def compute_entity_embeddings(self, old_entity_list, old_dataloader, ner_dataloader):
        """
        从旧任务数据中计算实体嵌入，存储到self.past_tasks。
        - old_entity_list: 旧任务实体列表 (e.g., ['PER', 'ORG'])
        - old_dataloader: 旧任务的DataLoader
        - ner_dataloader: NER_dataloader实例，用于访问label_list
        返回: None (结果存储在self.past_tasks)
        """
        entity_embs = {}
        self.model.eval()
        device = next(self.model.parameters()).device
        hidden_dim = getattr(self.model, 'hidden_dim', 768)
        with torch.no_grad():
            for entity in old_entity_list:
                print(000000000000000000000)
                print(old_entity_list)
                entity_embs[entity] = []
                for X_batch, y_batch in old_dataloader:
                    X_batch, y_batch = X_batch.cuda(), y_batch.cuda()
                    # 获取最后一层特征
                    features = self.model.forward_encoder(X_batch)  # (bsz, seq_len, 768)
                    # 筛选实体token（B-ENT或I-ENT）
                    entity_mask = (y_batch == ner_dataloader.label_list.index(f'B-{entity}')) | \
                                  (y_batch == ner_dataloader.label_list.index(f'I-{entity}'))
                    if entity_mask.sum() > 0:
                        # 平均实体token的嵌入
                        entity_features = features[entity_mask].mean(dim=0)  # (768,)
                        entity_embs[entity].append(entity_features)
                # 平均所有样本的实体嵌入
                if entity_embs[entity]:
                    entity_embs[entity] = torch.stack(entity_embs[entity]).mean(dim=0)  # (768,)
                    logger.info(f"Entity {entity} embedding norm: {entity_embs[entity].norm()}")
                else:
                    # 若无样本，使用随机嵌入
                    entity_embs[entity] = torch.randn(768).cuda()
                    logger.warning(f"No samples found for entity {entity}, using random embedding")
        
        # 存储到past_tasks
        task_info = {
            'task_id': len(self.past_tasks),
            'age': len(self.past_tasks),  # 当前任务年龄
            'entity_embs': [entity_embs[entity] for entity in old_entity_list]
        }
        self.past_tasks.append(task_info)
        logger.info(f"Stored task {task_info['task_id']} with {len(old_entity_list)} entity embeddings")

    def batch_loss_cpfd(self, labels,dataloader_test_cumul,all_seen_entity_list,old_classes_list,new_classes_list,new_entity_list):
     
        original_labels = labels.clone()
        self.loss = 0
        refer_dims = self.refer_model.classifier.output_dim # old model 输出维度
        all_dims = self.model.classifier.output_dim
            
        # Check input
        assert self.logits!=None, "logits is none!"
        assert self.refer_model!=None, "refer_model is none!"
        assert self.inputs!=None, "inputs is none!"
        assert self.inputs.shape[:2]==labels.shape[:2], "inputs and labels are not matched!"  

        with torch.no_grad():
            self.refer_model.eval()
            refer_all_features= self.refer_model.encoder(self.inputs)
            refer_features = refer_all_features[1][-1]
            refer_logits = self.refer_model.forward_classifier(refer_features)# (bsz,seq_len,refer_dims)
            assert refer_logits.shape[:2] == self.logits.shape[:2], \
                    "the first 2 dims of refer_logits and logits are not equal!!!"
        
            refer_all_attention_features = refer_all_features[2]
        

        classif_adaptive_factor = 1.0
        mask_background = (labels < self.old_classes) & (labels != pad_token_label_id) # 0 的位置

  
        probs = torch.softmax(refer_logits, dim=-1) # (bsz,seq_len,refer_dims)
        _, pseudo_labels = probs.max(dim=-1) # 最大概率 以及 最大概率所在位置

        mask_valid_pseudo = entropy(probs) < self.thresholds[pseudo_labels] # (bsz, seq_len)

        # All old labels that are NOT confident enough to be used as pseudo labels:
        labels[~mask_valid_pseudo & mask_background] = pad_token_label_id

        # All old labels that are confident enough to be used as pseudo labels:
        labels[mask_valid_pseudo & mask_background] = pseudo_labels[mask_valid_pseudo &
                                                                    mask_background]
 
        if self.params.classif_adaptive_factor:
            # Number of old/bg tokens that are certain
            num = (mask_valid_pseudo & mask_background).float().sum(dim=-1)
            # Number of old/bg tokens
            den =  mask_background.float().sum(dim=-1)
            # If all old/bg tokens are certain the factor is 1 (loss not changed)
            # Else the factor is < 1, i.e. the loss is reduced to avoid
            # giving too much importance to new tokens
            classif_adaptive_factor = num / (den + 1e-6)
            classif_adaptive_factor = classif_adaptive_factor[:, None]

            if self.params.classif_adaptive_min_factor:
                classif_adaptive_factor = classif_adaptive_factor.clamp(min=self.params.classif_adaptive_min_factor)

   
        loss = nn.CrossEntropyLoss(reduction='none')(self.logits.permute(0,2,1), labels) # 0 新类 旧类伪标签 -100(计算的loss为0)    (bsz,seq_len)
        loss = classif_adaptive_factor * loss

        # type balance

        pre_sample_weights = self.calculate_sample_weight(labels)
        sample_weights = torch.ones(loss.size()).cuda() 
        for i in range(pre_sample_weights.size(0)): 
            sample_weights[i][(labels[i] > 0) & (labels[i] < self.old_classes)] = pre_sample_weights[i] # 样本中 被伪标注出来的旧类token 赋予 计算的权重， 其他（新类，真正的背景）token对应权重1

  
        loss = sample_weights * loss


        ignore_mask = (labels!=pad_token_label_id) 
            
        if torch.sum(ignore_mask.float())==0: 
            ce_loss = torch.tensor(0., requires_grad=True).cuda()
        else:
            ce_loss = loss[ignore_mask].mean()  # scalar
        

        all_attention_features = self.all_features[2]
        
        # FNN提取FFN输出（隐藏状态）
        teacher_hidden_states = self.all_features[1]  # 13个张量（嵌入层+12个Transformer层）
        student_hidden_states = refer_all_features[1]
        
        distill_mask = torch.logical_and(original_labels==0, original_labels!=pad_token_label_id) # other class token (non-entity)
        if torch.sum(distill_mask.float())==0:
            distill_loss = torch.tensor(0., requires_grad=True).cuda()
        else:   
            # distill logits loss
            old_logits_score = F.log_softmax(
                                self.logits[distill_mask]/self.params.temperature,
                                dim=-1)[:,:refer_dims].view(-1, refer_dims) #(bsz*seq_len(select out), refer_dims)
     
            ref_old_logits_score = F.softmax(
                                refer_logits[distill_mask]/self.params.ref_temperature, 
                                dim=-1).view(-1, refer_dims)

            distill_logits_loss = nn.KLDivLoss(reduction='batchmean')(old_logits_score, ref_old_logits_score)

        # if torch.sum(distill_mask.float())==0:
        #     distill_loss = torch.tensor(0., requires_grad=True).cuda()
        # else:   
        #         # 时序注意力动态加权（仅当有旧任务时）
        #     if len(self.past_tasks) > 0:
        #         # 新任务实体嵌入（简化示例）
        #         new_entity_embs = [torch.randn(768).cuda() for _ in new_entity_list]  # 实际从当前任务数据获取
        #         print(new_entity_list)
        #         print(new_classes_list)
            
        #         # 计算每个旧任务的相似性和衰减
        #         similarities = []
        #         task_ages = []
        #         for past_task in self.past_tasks:
        #             sim = compute_task_similarity(past_task['entity_embs'], new_entity_embs)
        #             similarities.append(sim)
        #             task_ages.append(past_task['age'])
            
        #         # 结合相似性和衰减生成注意力权重
        #         sim_scores = torch.tensor(similarities, dtype=torch.float32).cuda()
        #         decay_weights = compute_temporal_decay(task_ages, decay_factor=params.temporal_decay_factor)  # 新增params.temporal_decay_factor=0.5
        #         attention_scores = sim_scores * decay_weights  # 乘法结合
        #         attention_weights = F.softmax(attention_scores, dim=0)  # softmax归一化
            
        #         # 假设refer_logits已包含所有旧任务知识（如果有多个旧模型，可扩展为加权多个refer_logits）
        #         # 这里简化：直接用注意力权重缩放refer_logits（实际中若有多个refer_model，可加权融合）
        #         weighted_refer_logits = refer_logits * attention_weights.unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # 广播到batch维度
            
        #         # 修改蒸馏损失使用加权logits
        #         old_logits_score = F.log_softmax(self.logits[distill_mask] / self.params.temperature, dim=-1)[:, :refer_dims].view(-1, refer_dims)
        #         ref_old_logits_score = F.softmax(weighted_refer_logits[distill_mask] / self.params.ref_temperature, dim=-1).view(-1, refer_dims)
        #         distill_logits_loss = nn.KLDivLoss(reduction='batchmean')(old_logits_score, ref_old_logits_score)
        #     else:
        #         # distill logits loss
        #         old_logits_score = F.log_softmax(
        #                             self.logits[distill_mask]/self.params.temperature,
        #                             dim=-1)[:,:refer_dims].view(-1, refer_dims) #(bsz*seq_len(select out), refer_dims)
         
        #         ref_old_logits_score = F.softmax(
        #                             refer_logits[distill_mask]/self.params.ref_temperature, 
        #                             dim=-1).view(-1, refer_dims)
    
        #       distill_logits_loss = nn.KLDivLoss(reduction='batchmean')(old_logits_score, ref_old_logits_score)
# 逐层跳过实验
#            for skip_layer in range(1, 13):  # 测试跳过每个 FFN 层
#                distill_ffn_loss, layer_losses = self.compute_ffn_distillation_loss(
#                    teacher_hidden_states, student_hidden_states, skip_layer=skip_layer
#                )
#                distill_loss = self.params.distill_weight * (distill_logits_loss + distill_ffn_loss)
#                total_loss = ce_loss + distill_loss
#                self.optimizer.zero_grad()
#                total_loss.backward()
#                self.optimizer.step()
#                f1, ma_f1, f1_score_dict = self.evaluate(dataloader_test_cumul, each_class=True, entity_order=all_seen_entity_list,is_plot_hist=False)
#                self.layer_f1_scores[skip_layer] = {
#                    'old_f1': np.mean([f1_score_dict.get(e, 0.0) for e in old_classes_list]),
#                    'new_f1': np.mean([f1_score_dict.get(e, 0.0) for e in new_classes_list]),
#                    'overall_f1': f1
#                }
#                self.layer_losses[skip_layer] = distill_ffn_loss.item()
#                logger.info(f"Skip Layer {skip_layer}: Old F1={self.layer_f1_scores[skip_layer]['old_f1']:.2f}, New F1={self.layer_f1_scores[skip_layer]['new_f1']:.2f}, Overall F1={f1:.2f}, Distill Loss=                                {distill_ffn_loss.item():.4f}")
#            self.plot_ffn_layer_contribution()





            #     distill_ffn_loss = 0.0
            #     for layer in range(1, 13):  # 跳过嵌入层，只看12个Transformer层
            #     # if layer in [2, 4, 6, 8, 10]:
            #         # 检查当前层是否在跳过的层列表中
                    
            #     #   continue
                
                    
            #         teacher_hidden = teacher_hidden_states[layer]  # 形状：(batch_size, seq_len, 768)
            #         student_hidden = student_hidden_states[layer]
                    
                    
                    
                    
            #         # 计算该层的MSE损失
            #         layer_loss = nn.MSELoss(reduction='mean')(student_hidden, teacher_hidden)
            #         distill_ffn_loss += layer_loss
            
            # # 平均损失（12个FFN层）
            #     distill_ffn_loss = distill_ffn_loss / 6
            distill_loss = params.distill_weight * distill_logits_loss
              #  distill_loss = params.distill_weight*(distill_logits_loss + distill_ffn_loss)

       
            # distill attention feature loss
        #    '''
         #       all_attention_features(12 layer attention map, 12*(bsz, att_heads=12, seq_len, seq_len))
         #   '''
#            distill_attention_features_loss = torch.tensor(0., requires_grad=True).cuda()
#            for attention_features, refer_attention_features in zip(all_attention_features, refer_all_attention_features):
#                assert attention_features.shape == refer_attention_features.shape, (attention_features.shape, refer_attention_features.shape)  # (bsz, heads=12, seq_len, seq_len)
#                
#                bsz, heads, seq_len, seq_len = attention_features.shape
#
#                attention_features = torch.where(attention_features <= -1e2, torch.zeros_like(attention_features).cuda(),
#                                              attention_features)
#                refer_attention_features = torch.where(refer_attention_features <= -1e2, torch.zeros_like(refer_attention_features).cuda(),
#                                              refer_attention_features)
#
#                # pooled feature distillation
#                pfd1 = torch.mean(attention_features, dim=1)
#                rpfd1 = torch.mean(refer_attention_features, dim=1)
#
#                pfd2 = torch.mean(attention_features, dim=2)
#                rpfd2 = torch.mean(refer_attention_features, dim=2)
#
#                pfd3 = torch.mean(attention_features, dim=3)
#                rpfd3 = torch.mean(refer_attention_features, dim=3)
#
#                layer_loss1 = nn.MSELoss(reduction='mean')(pfd1,rpfd1)
#                layer_loss2 = nn.MSELoss(reduction='mean')(pfd2,rpfd2)
#                layer_loss3 = nn.MSELoss(reduction='mean')(pfd3,rpfd3)
#
#                layer_loss = layer_loss1 + layer_loss2 + layer_loss3
#
#                distill_attention_features_loss += layer_loss
#
#            distill_attention_features_loss = distill_attention_features_loss/len(all_attention_features)
#
#
#            distill_loss = params.distill_weight*(distill_logits_loss + distill_attention_features_loss)
          

        if not params.adaptive_distill_weight: 
            distill_loss_coefficient = 1 
        elif params.adaptive_schedule=='root': 
            distill_loss_coefficient = np.power((refer_dims-1)/(all_dims-refer_dims),0.5) 
        elif params.adaptive_schedule=='linear':
            distill_loss_coefficient = np.power((refer_dims-1)/(all_dims-refer_dims),1)
        elif params.adaptive_schedule=='square':
            distill_loss_coefficient = np.power((refer_dims-1)/(all_dims-refer_dims),2)
        else:
            raise Exception('Invalid %s'%(params.adaptive_schedule))
        
        # 在 batch_loss_cpfd 末尾，添加：
#        replay_loss = torch.tensor(0., requires_grad=True).cuda()
#        if self.memory_dataloader is not None:
#            for X_replay, y_replay in self.memory_dataloader:  # 重放整个 buffer（或采样 batch）
#                X_replay, y_replay = X_replay.cuda(), y_replay.cuda()
#                with torch.no_grad():
#                    features_replay = self.model.forward_encoder(X_replay)  # 当前编码器
#                
#                # 通过分类器池 ensemble
#                ensemble_logits = []
#                for past_classifier in self.classifier_pool:
#                    past_logits = past_classifier(features_replay)  # (bsz, seq_len, past_dim) 注意：dim 可能不同，需 pad 或 slice
#                    ensemble_logits.append(past_logits[:, :, :min(past_logits.size(-1), self.model.classifier.output_dim)])  # 兼容 dim 增长
#                
#                # 平均 logits 或 softmax
#                ensemble_logits = torch.mean(torch.stack(ensemble_logits), dim=0)  # (bsz, seq_len, dim)
#                ensemble_probs = F.softmax(ensemble_logits / self.params.temperature, dim=-1)
#                
#                # CE 损失（参考的分类损失）
#                replay_loss += nn.CrossEntropyLoss()(ensemble_logits.view(-1, ensemble_logits.size(-1)), y_replay.flatten().long())
#            
#            replay_loss /= len(self.memory_dataloader)  # 平均
#            if self.params.replay_adaptive_factor:
#                # 自适应权重（类似于 classif_adaptive_factor，根据旧/新比例）
#                adaptive_replay_factor = (self.old_classes / self.nb_current_classes)  # 旧类比例
#                replay_loss *= adaptive_replay_factor
        replay_loss = torch.tensor(0., requires_grad=True).cuda()
        if self.memory_dataloader is not None:
            for X_replay, y_replay in self.memory_dataloader:  # 重放整个 buffer（或采样 batch）
                X_replay, y_replay = X_replay.cuda(), y_replay.cuda()
                with torch.no_grad():
                    features_replay = self.model.forward_encoder(X_replay)  # 当前编码器
                
                # 通过分类器池 ensemble
                ensemble_logits = []
                target_dim = self.model.classifier.output_dim  # 当前模型的输出维度（例如，len(all_seen_entity_list) * 2 + 1）
                for past_classifier in self.classifier_pool:
                    past_logits = past_classifier(features_replay)  # (bsz, seq_len, past_dim)
                    past_dim = past_logits.size(-1)
                    if past_dim != target_dim:
                        # 创建目标形状的零张量
                        padded_logits = torch.zeros(X_replay.size(0), X_replay.size(1), target_dim, device=X_replay.device)
                        # 复制旧分类器的 logits 到对应位置
                        padded_logits[:, :, :min(past_dim, target_dim)] = past_logits[:, :, :min(past_dim, target_dim)]
                        ensemble_logits.append(padded_logits)
                    else:
                        ensemble_logits.append(past_logits)
                
                # 调试：检查每个 logits 的形状
                #logger.debug(f"ensemble_logits shapes: {[logit.shape for logit in ensemble_logits]}")
                
                # 平均 logits
                ensemble_logits = torch.mean(torch.stack(ensemble_logits), dim=0)  # (bsz, seq_len, target_dim)
                ensemble_probs = F.softmax(ensemble_logits / self.params.temperature, dim=-1)
                
                # CE 损失（参考的分类损失）
                replay_loss += nn.CrossEntropyLoss()(ensemble_logits.view(-1, ensemble_logits.size(-1)), y_replay.flatten().long())
            
            replay_loss /= len(self.memory_dataloader)  # 平均
            if self.params.replay_adaptive_factor:
                # 自适应权重（类似于 classif_adaptive_factor，根据旧/新比例）
                adaptive_replay_factor = (self.old_classes / self.nb_current_classes)  # 旧类比例
        
        ffn_loss = torch.tensor(0.0, device=self.inputs.device)#神经元蒸馏损失初始化
        # 在 batch_loss_cpfd 中
        if self.ffn_distiller is not None and self.refer_model is not None:
            with torch.no_grad():
                #teacher_hidden = self.refer_model.forward_encoder(self.inputs, output_hidden_states=True)
                # 如果你的 forward_encoder 不支持 output_hidden_states，就改造成支持，或者手动收集
                # 简单做法：直接用 encoder 的所有层输出（你原来就用了 features[1] 是所有层）
                teacher_hidden = self.refer_model.encoder(self.inputs)[1]  # tuple of 13 layers

            student_hidden = self.model.encoder(self.inputs)[1]  # 同上

            ffn_loss = self.ffn_distiller.compute_loss(student_hidden, teacher_hidden)
            
        # 更新总损失
        self.loss = ce_loss + distill_loss_coefficient * distill_loss + ffn_loss #+ self.params.replay_weight * replay_loss
        return ce_loss.item(), distill_loss_coefficient * distill_loss.item(), self.params.replay_weight * replay_loss.item() , ffn_loss.item()# 更新返回
        

            
    def batch_backward(self):
        self.model.train()
        self.optimizer.zero_grad()        
        self.loss.backward()
        self.optimizer.step()
        
        return self.loss.item()

    def evaluate(self, dataloader, each_class=False, entity_order=[], is_plot_hist=False, is_plot_cm=False):
        with torch.no_grad():
            self.model.eval()

            y_list = []
            x_list = []
            logits_list = []

            for x, y in dataloader: 
                x, y = x.cuda(), y.cuda()
                self.batch_forward(x)
                _logits = self.logits.view(-1, self.logits.shape[-1]).detach().cpu()
                logits_list.append(_logits)
                x = x.view(x.size(0)*x.size(1)).detach().cpu() # bs*seq_len
                x_list.append(x) 
                y = y.view(y.size(0)*y.size(1)).detach().cpu()
                y_list.append(y)

            
            y_list = torch.cat(y_list)
            x_list = torch.cat(x_list)
            logits_list = torch.cat(logits_list)   
            pred_list = torch.argmax(logits_list, dim=-1)

            ### Plot the (logits) prob distribution for each class
            if is_plot_hist: # False
                plot_prob_hist_each_class(deepcopy(y_list), 
                                        deepcopy(logits_list),
                                        ignore_label_lst=[
                                            self.label_list.index('O'),
                                            pad_token_label_id
                                        ])


            ### for confusion matrix visualization
            if is_plot_cm: # False
                plot_confusion_matrix(deepcopy(pred_list),
                                deepcopy(y_list), 
                                label_list=self.label_list,
                                pad_token_label_id=pad_token_label_id)

            ### calcuate f1 score
            pred_line = []
            gold_line = []
            for pred_index, gold_index in zip(pred_list, y_list):
                gold_index = int(gold_index)
                if gold_index != pad_token_label_id: # !=-100
                    pred_token = self.label_list[pred_index] # label索引转label
                    gold_token = self.label_list[gold_index]
                    # lines.append("w" + " " + pred_token + " " + gold_token)
                    pred_line.append(pred_token) 
                    gold_line.append(gold_token) 

            # Check whether the label set are the same,
            # ensure that the predict label set is the subset of the gold label set
            gold_label_set, pred_label_set = np.unique(gold_line), np.unique(pred_line)
            if set(gold_label_set)!=set(pred_label_set):
                O_label_set = []
                for e in pred_label_set:
                    if e not in gold_label_set:
                        O_label_set.append(e)
                if len(O_label_set)>0:
                    # map the predicted labels which are not seen in gold label set to 'O'
                    for i, pred in enumerate(pred_line):
                        if pred in O_label_set:
                            pred_line[i] = 'O'

            self.model.train()

            # compute overall f1 score
            # micro f1 (default)
            f1 = f1_score([gold_line], [pred_line])*100
            # macro f1 (average of each class f1)
            ma_f1 = f1_score([gold_line], [pred_line], average='macro')*100
            if not each_class: # 不打印每个类别的f1
                return f1, ma_f1

            # compute f1 score for each class
            f1_list = f1_score([gold_line], [pred_line], average=None)
            f1_list = list(np.array(f1_list)*100)
            gold_entity_set = set()
            for l in gold_label_set:
                if 'B-' in l or 'I-' in l or 'E-' in l or 'S-' in l:
                    gold_entity_set.add(l[2:])
            gold_entity_list = sorted(list(gold_entity_set))
            f1_score_dict = dict()
            for e, s in zip(gold_entity_list,f1_list):
                f1_score_dict[e] = round(s,2)
            # using the default order for f1_score_dict
            if entity_order==[]:
                return f1, ma_f1, f1_score_dict
            # using the pre-defined order for f1_score_dict
            assert set(entity_order)==set(gold_entity_list),\
                "gold_entity_list and entity_order has different entity set!"
            ordered_f1_score_dict = dict()
            for e in entity_order:
                ordered_f1_score_dict[e] = f1_score_dict[e]
            return f1, ma_f1, ordered_f1_score_dict

    def save_model(self, save_model_name, path=''):
        """
        save the best model
        """
        if len(path)>0:
            saved_path = os.path.join(path, str(save_model_name))
        else:
            saved_path = os.path.join(self.params.dump_path, str(save_model_name))
        torch.save({
            "hidden_dim": self.model.hidden_dim,
            "output_dim": self.model.output_dim,
            "encoder": self.model.encoder.state_dict(),
            "classifier": self.model.classifier
        }, saved_path)
        logger.info("Best model has been saved to %s" % saved_path)

    def load_model(self, load_model_name, path=''):
        """
        load the checkpoint
        """
        if len(path)>0:
            load_path = os.path.join(path, str(load_model_name))
        else:
            load_path = os.path.join(self.params.dump_path, str(load_model_name))
        ckpt = torch.load(load_path)

        self.model.hidden_dim = ckpt['hidden_dim']
        self.model.output_dim = ckpt['output_dim']
        self.model.encoder.load_state_dict(ckpt['encoder'])
        self.model.classifier = ckpt['classifier']
        logger.info("Model has been load from %s" % load_path)