import torch
import numpy as np

class OmniDataCollator:
    def __init__(self, processor):
        self.processor = processor
        # 适配 Qwen2.5 标准 Tokenizer 接口
        # self.im_start_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
        # self.assistant_id = self.processor.tokenizer.convert_tokens_to_ids("assistant")
        self.query_id = self.processor.tokenizer.convert_tokens_to_ids("<|action_query|>")
        self.pad_token_id = self.processor.tokenizer.pad_token_id
        # 确保 pad_token_id 存在
        if self.pad_token_id is None:
            self.pad_token_id = self.processor.tokenizer.eos_token_id

        print(f"DEBUG Collator: Action Query ID = {self.query_id}")

    def __call__(self, features):
        batch = {}
        
        # 1. 基础文本序列对齐
        batch['input_ids'] = torch.nn.utils.rnn.pad_sequence(
            [f['input_ids'] for f in features], batch_first=True, padding_value=self.pad_token_id
        )
        batch['attention_mask'] = torch.nn.utils.rnn.pad_sequence(
            [f['attention_mask'] for f in features], batch_first=True, padding_value=0
        )
        
        # 2. 标签掩码处理
        # labels_list = []
        # for f in features:
        #     input_ids = f['input_ids']
        #     # 初始化 mask 全为 -100 (默认不计算 loss)
        #     # labels 的形状必须与 input_ids 一致
        #     target_labels = torch.full_like(input_ids, -100)
            
        #     # A. 寻找 Assistant 的起始位置
        #     # Qwen2.5 格式: ... <|im_start|> assistant \n ...
        #     # 我们寻找 [im_start_id, assistant_id] 的组合
        #     # 注意：这里简化查找逻辑，找到最后一个 assistant 标签（通常在一轮对话最后）
            
        #     # 找到所有 <|im_start|> 的位置
        #     start_indices = (input_ids == self.im_start_id).nonzero(as_tuple=True)[0]
            
        #     answer_start_idx = -1
            
        #     # 倒序查找，找到最后一个 'assistant' 对应的块
        #     for idx in torch.flip(start_indices, dims=[0]):
        #         if idx + 1 < len(input_ids) and input_ids[idx + 1] == self.assistant_id:
        #             # 找到了 <|im_start|> assistant
        #             # 通常后面紧跟一个 \n (id 可能会变，所以我们往后找非特殊 token)
        #             # 假设结构是: [im_start, assistant, \n] -> 内容从 idx + 3 开始
        #             # 但为了安全，我们结合 Query 逻辑判断
        #             base_start = idx + 2 # 指向 assistant 后面那个 token
                    
        #             # --- B. 判断是 Stage 1 还是 Stage 2 ---
        #             # 检查在这个 assistant 之后是否存在 action_query_id
        #             # 截取后续片段
        #             remainder = input_ids[base_start:]
        #             query_positions = (remainder == self.query_id).nonzero(as_tuple=True)[0]
                    
        #             if len(query_positions) > 0:
        #                 # === Stage 2: 包含 Action Queries ===
        #                 # Answer 应该从 最后一个 Query 之后 开始
        #                 # query_positions 是相对于 remainder 的索引
        #                 last_query_idx_in_remainder = query_positions[-1]
        #                 answer_start_idx = base_start + last_query_idx_in_remainder + 1
        #             else:
        #                 # === Stage 1: 纯文本 QA (无 Action Queries) ===
        #                 # Answer 直接从 assistant 后面的 \n 之后开始
        #                 # 这里稍微宽松一点，通常是 base_start + 1 (\n)
        #                 # 如果没有 \n，就是 base_start
        #                 answer_start_idx = base_start + 1 # 跳过换行符
                    
        #             break # 找到最后一个 assistant 就停止
            
        #     # C. 填充 Labels
        #     if answer_start_idx != -1 and answer_start_idx < len(input_ids):
        #         # 将 Answer 部分的 ID 复制到 Labels
        #         target_labels[answer_start_idx:] = input_ids[answer_start_idx:]
                
        #         # [再次保险] 无论如何，强制把所有的 <|action_query|> 设为 -100
        #         # 即使上面逻辑算对了，这步也是双重保险，防止 LLM 学习预测占位符
        #         target_labels[input_ids == self.query_id] = -100
                
        #         # [Pad 处理] 确保 Pad token 也是 -100
        #         target_labels[input_ids == self.pad_token_id] = -100
                
        #     labels_list.append(target_labels)

        # 堆叠 Labels
        # batch['labels'] = torch.nn.utils.rnn.pad_sequence(
        #     labels_list, batch_first=True, padding_value=-100
        # )
        batch['labels'] = torch.full_like(batch['input_ids'], -100)
        # 3. 视频/视觉输入处理
        if 'pixel_values' in features[0]:
            batch['pixel_values'] = torch.cat([f['pixel_values'] for f in features], dim=0)
            batch['image_grid_thw'] = torch.cat([f['image_grid_thw'] for f in features], dim=0)
        if 'sample_token' in features[0]:
            batch['sample_token'] = [f['sample_token'] for f in features]
        # 4. 轨迹与状态堆叠
        batch['future_traj'] = torch.stack([f['future_traj'] for f in features])
        batch['ego_status'] = torch.stack([f['ego_status'] for f in features])
        batch['target_images'] = torch.stack([f['target_images'] for f in features])
        

        # 5. Simple-BEV sidecar 数据
        if 'bev_imgs' in features[0]:
            batch['bev_imgs'] = torch.stack([f['bev_imgs'] for f in features], dim=0)          # [B, 3, 6, 3, H, W]

        if 'bev_rots' in features[0]:
            batch['bev_rots'] = torch.stack([f['bev_rots'] for f in features], dim=0)          # [B, 3, 6, 3, 3]

        if 'bev_trans' in features[0]:
            batch['bev_trans'] = torch.stack([f['bev_trans'] for f in features], dim=0)        # [B, 3, 6, 3]

        if 'bev_intrins' in features[0]:
            batch['bev_intrins'] = torch.stack([f['bev_intrins'] for f in features], dim=0)    # [B, 3, 6, 4, 4]

        if 'bev_ego_pose' in features[0]:
            batch['bev_ego_pose'] = torch.stack([f['bev_ego_pose'] for f in features], dim=0)  # [B, 3, 4, 4]
        
        return batch