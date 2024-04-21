import dataclasses
import json
import os
from tqdm import tqdm

import torch
import torch.nn as nn


class MoDLlamaTrainer():
    def __init__(self, params, model, tokenizer, dataloader):
        self.params = params
        self.model = model
        self.tokenizer = tokenizer
        self.dataloader = dataloader

    def save_params(self, model_dir):
        param_dict = dataclasses.asdict(self.params)
        param_dict.pop('max_batch_size')
        param_dict.pop('max_seq_len')

        for k, v in list(param_dict.items()):
            if v is None:
                param_dict.pop(k)
            elif isinstance(v, bool):
                param_dict[k] = int(v)
        
        with open(model_dir + "/params.json", 'w') as f:
            json.dump(param_dict, f)
    
    def compute_causal_loss(self, weights, topk_indices):
        criterion = nn.BCEWithLogitsLoss()
        token_weights = torch.stack(weights).flatten(0,1)
        aux_targets = torch.zeros_like(token_weights)
        batches = torch.arange(token_weights.size(0)).unsqueeze(-1)
        aux_targets[batches, torch.stack(topk_indices).flatten(0,1)] = 1.0
        aux_targets = aux_targets.flatten()
        return criterion(token_weights.flatten().to("cuda"), aux_targets.to("cuda")), aux_targets

    def train(
        self,
        epochs=10,
        lr=1e-5,
        model_dir="./models/MoDLlama/",
        log_path="./logs/MoDLlama_log.txt",
        use_aux_loss=False,
        use_aux_predictor=False,
        aux_epochs=1,
        log_steps=1000,
    ):  
        self.model.train()
        self.save_params(model_dir)

        min_loss = float("inf")
        criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_id)

        if use_aux_predictor:
            for name, param in self.model.named_parameters():
                if name.startswith("aux_router"):
                    param.requires_grad = False

        optimizer = torch.optim.AdamW(self.model.parameters(), lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs*len(self.dataloader))
        
        prev_ckpt_path = None
        for epoch in range(epochs):
            running_loss = 0.0
            running_causal_loss = 0.0

            for i, (inputs, targets) in enumerate(tqdm(self.dataloader, desc=f"Epoch: {epoch+1}")):
                optimizer.zero_grad()

                outputs = self.model(inputs, start_pos=0)

                loss = criterion(outputs['output'].permute(0, 2, 1), targets)

                causal_loss = 0.0
                if use_aux_loss:
                    # compute auxiliary loss
                    causal_loss, _ = self.compute_causal_loss(outputs['token_weights'], outputs['topk_indices'])
                    loss += causal_loss
                    running_causal_loss += causal_loss.detach().cpu().item()
                        
                loss.backward()
                optimizer.step()
                scheduler.step()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)

                running_loss += loss.detach().cpu().item()

                if (i+1) % log_steps == 0:
                    avg_loss = running_loss / (i+1)
                    print(f"Loss at step {i+1}: {avg_loss}")
                    if use_aux_loss:
                        avg_causal_loss = running_causal_loss / (i+1)
                        print(f"Causal Loss at step {i+1}: {avg_causal_loss}")

            epoch_loss = running_loss / len(self.dataloader)

            if min_loss > epoch_loss:
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir, exist_ok=True)
                elif prev_ckpt_path:
                    os.remove(prev_ckpt_path)
                
                prev_ckpt_path = os.path.join(model_dir, f"checkpoint_epoch-{epoch}.pth")
                torch.save(self.model.state_dict(), prev_ckpt_path)
                min_loss = epoch_loss
            
            print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss}")
            if epoch == 0 and os.path.exists(log_path):
                os.remove(log_path)
            with open(log_path, 'a') as f:
                f.write(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss}")
                f.write("\n")
    
        # train auxiliary router
        # freeze all parameters but aux router
        for name, param in self.model.named_parameters():
            if not name.startswith("aux_router"):
                param.requires_grad = False
            else:
                param.requires_grad = True
        optimizer = torch.optim.AdamW(self.model.parameters(), 1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, aux_epochs*len(self.dataloader))

        if use_aux_predictor:
            print("-"*10, "Training Auxiliary Router", "-"*10)
            for epoch in range(aux_epochs):
                running_causal_loss = 0.0

                correct = 0
                total = 0
                for i, (inputs, targets) in enumerate(tqdm(self.dataloader, desc=f"Epoch: {epoch+1}")):
                    optimizer.zero_grad()
                    
                    outputs = self.model(inputs, start_pos=0)

                    # compute auxiliary loss
                    causal_loss = 0.0
                    causal_loss, aux_targets = self.compute_causal_loss(outputs['aux_weights'], outputs['topk_indices'])
                    running_causal_loss += causal_loss.detach().cpu().item()
                
                    # measure accuracy during training
                    token_weights = torch.stack(outputs['aux_weights']).flatten(0,1)
                    batches = torch.arange(token_weights.size(0)).unsqueeze(-1)
                    k = min(token_weights.size(-1), self.params.capacity)
                    pred_indices = torch.topk(token_weights, k=k, sorted=False).indices
                    preds = torch.zeros_like(token_weights)
                    preds[batches, pred_indices] = 1.0
                    correct += (preds.flatten() == aux_targets).sum()  
                    total += len(aux_targets)

                    causal_loss.backward()
                    optimizer.step()
                    scheduler.step()

                    if (i+1) % log_steps == 0:
                        accuracy = correct / total
                        correct, total = 0, 0
                        print(f"Token Predictor Accuracy at step {i+1}: {accuracy}")

                epoch_loss = running_causal_loss / len(self.dataloader)

                torch.save(self.model.state_dict(), prev_ckpt_path)
                
                print(f"Epoch {epoch+1}/{aux_epochs} for aux router - Causal Loss: {epoch_loss}")
                with open(log_path, 'a') as f:
                    f.write(f"Epoch {epoch+1}/{aux_epochs} for aux router - Causal Loss: {epoch_loss}")
                    f.write("\n")
