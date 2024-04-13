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

    def train(
        self,
        epochs=10,
        lr=1e-5,
        model_dir="./models/MoDLlama/",
        log_path="./logs/MoDLlama_log.txt",
        use_aux_loss=False,
        use_aux_predictor=False,
        log_steps=1000,
    ):  
        self.model.train()

        min_loss = float("inf")
        criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_id)
        bce_criterion = nn.BCEWithLogitsLoss()
        
        if use_aux_predictor:
            parameters = []
            aux_parameters = []
            for name, param in self.model.named_parameters():
                if name.startswith("aux_router"):
                    aux_parameters.append(param)
                else:
                    parameters.append(param)
            optimizer = torch.optim.AdamW(parameters, lr)
            aux_optimizer = torch.optim.AdamW(aux_parameters, 1e-3)
        else:
            optimizer = torch.optim.AdamW(self.model.parameters(), lr)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs*len(self.dataloader))

        for epoch in range(epochs):
            running_loss = 0.0
            running_causal_loss = 0.0

            correct = 0
            total = 0
            for i, (inputs, targets) in enumerate(tqdm(self.dataloader, desc=f"Epoch: {epoch}")):
                optimizer.zero_grad()

                outputs = self.model(inputs, start_pos=0)

                loss = criterion(outputs['output'].permute(0, 2, 1), targets)

                causal_loss = 0.0
                if use_aux_loss or use_aux_predictor:
                    # compute auxiliary loss
                    weights = outputs['aux_weights'] if use_aux_predictor else outputs['token_weights']
                    token_weights = torch.stack(weights).flatten(0,1)
                    aux_targets = torch.zeros_like(token_weights)
                    batches = torch.arange(token_weights.size(0)).unsqueeze(-1)
                    aux_targets[batches, torch.stack(outputs['topk_indices']).flatten(0,1)] = 1.0
                    aux_targets = aux_targets.flatten()
                    causal_loss = bce_criterion(token_weights.flatten().to("cuda"), aux_targets.to("cuda"))
                    running_causal_loss += causal_loss.detach().cpu().item()
                    
                    if use_aux_predictor:
                        # measure accuracy during training
                        k = min(token_weights.size(-1), self.params.capacity)
                        pred_indices = torch.topk(token_weights, k=k, sorted=False).indices
                        preds = torch.zeros_like(token_weights)
                        preds[batches, pred_indices] = 1.0
                        correct += (preds.flatten() == aux_targets).sum()  
                        total += len(aux_targets)

                loss += causal_loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)

                if use_aux_predictor:
                    aux_optimizer.step()

                running_loss += loss.detach().cpu().item()

                if (i+1) % log_steps == 0:
                    avg_loss = running_loss / (i+1)
                    print(f"Loss at step {i+1}: {avg_loss}")
                    if use_aux_loss:
                        avg_causal_loss = running_causal_loss / (i+1)
                        print(f"Causal Loss at step {i+1}: {avg_causal_loss}")
                    if use_aux_predictor:
                        accuracy = correct / total
                        correct, total = 0, 0
                        print(f"Token Predictor Accuracy at step {i+1}: {accuracy}")

            epoch_loss = running_loss / len(self.dataloader)

            prev_ckpt_path = None
            if min_loss > epoch_loss:
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir, exist_ok=True)
                elif prev_ckpt_path:
                    os.remove(prev_ckpt_path)
                    prev_ckpt_path = os.path.join(model_dir + f"checkpoint_epoch-{epoch}.pth")
                torch.save(self.model.state_dict(), os.path.join(model_dir + f"checkpoint_epoch-{epoch}.pth"))
                self.save_params(model_dir)
                min_loss = epoch_loss
            
            print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss}")
            with open(log_path, 'w') as f:
                f.write(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss}")
                f.write("\n")
