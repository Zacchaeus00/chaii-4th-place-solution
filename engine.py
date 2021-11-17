import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR

class Engine:
    def __init__(self, model, optimizer, scheduler, device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.scaler = torch.cuda.amp.GradScaler()
        self.input_columns = ['attention_mask', 'end_positions', 'input_ids', 'start_positions']
    
    def train(self, dataloader, accumulation_steps=1, grad_clip=1):
        self.model.train()
        final_loss = 0
        for i, batch in enumerate(tqdm(dataloader)):
            batch = {k: v.to(self.device) for k, v in batch.items() if k in self.input_columns}
            with torch.cuda.amp.autocast():
                outputs = self.model(**batch)
                loss = outputs.loss
                loss /= accumulation_steps
                nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            self.scaler.scale(loss).backward()

            if (i+1) % accumulation_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.model.zero_grad()
            if self.scheduler is not None:
                self.scheduler.step()
            final_loss += loss.item()        
        return final_loss / len(dataloader)
    
    def evaluate(self, dataloader):
        with torch.no_grad():
            self.model.eval()
            final_loss = 0
            for batch in tqdm(dataloader):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                final_loss += loss.item()
            
            return final_loss / len(dataloader)

    def predict(self, dataloader):
        with torch.no_grad():
            start_logits = []
            end_logits = []
            self.model.eval()
            for batch in tqdm(dataloader):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                start_logits.append(outputs.start_logits)
                end_logits.append(outputs.end_logits)
            start_logits = torch.cat(start_logits, dim=0)
            end_logits = torch.cat(end_logits, dim=0)
            start_logits = start_logits.cpu().numpy()
            end_logits = end_logits.cpu().numpy()
            return [start_logits, end_logits]

    def train_evaluate(self, train_dataloader, predict_dataloader, data_retriever, eval_steps, best_metric, save_path, metric='mean_jaccard', accumulation_steps=1, grad_clip=1):
        eval_steps //= data_retriever.batch_size
        self.model.train()
        tloss = 0
        for i, batch in enumerate(tqdm(train_dataloader)):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.cuda.amp.autocast():
                outputs = self.model(**batch)
                loss = outputs.loss
                loss /= accumulation_steps
                nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            self.scaler.scale(loss).backward()

            if (i+1) % accumulation_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.model.zero_grad()
            if self.scheduler is not None:
                self.scheduler.step()
            tloss += loss.item()
            if (i+1) % eval_steps == 0:
                raw_predictions = self.predict(predict_dataloader)
                score, lang_scores, df = data_retriever.evaluate_jaccard(raw_predictions, return_predictions=True)
                nonzero_jaccard_per = len(df[df['jaccard']!=0]) / len(df)
                cur_metric = score if metric == 'mean_jaccard' else nonzero_jaccard_per
                if cur_metric > best_metric:
                    best_metric = cur_metric
                    if save_path is not None:
                        self.save(save_path)
                print(f'batch {i+1}, tloss {tloss / eval_steps}, vscore {score}, nonzero_jaccard_per {nonzero_jaccard_per} best {metric} {best_metric}')
                tloss = 0
        return best_metric

    def save(self, path):
        path = Path(path)
        path.parents[0].mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)

class CustomModelEngine:
    def __init__(self, model, optimizer, scheduler, device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.scaler = torch.cuda.amp.GradScaler()
    
    def train(self, dataloader, accumulation_steps=1, grad_clip=1):
        self.model.train()
        final_loss = 0
        for i, batch in enumerate(tqdm(dataloader)):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.cuda.amp.autocast():
                outputs = self.model(**batch)
                loss = outputs['loss']
                loss /= accumulation_steps
                nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            self.scaler.scale(loss).backward()

            if (i+1) % accumulation_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.model.zero_grad()
            if self.scheduler is not None:
                self.scheduler.step()
            final_loss += loss.item()        
        return final_loss / len(dataloader)
    
    def evaluate(self, dataloader):
        with torch.no_grad():
            self.model.eval()
            final_loss = 0
            for batch in tqdm(dataloader):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs['loss']
                final_loss += loss.item()
            
            return final_loss / len(dataloader)

    def predict(self, dataloader):
        with torch.no_grad():
            start_logits = []
            end_logits = []
            self.model.eval()
            for batch in tqdm(dataloader):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                start_logits.append(outputs['start_logits'])
                end_logits.append(outputs['end_logits'])
            start_logits = torch.cat(start_logits, dim=0)
            end_logits = torch.cat(end_logits, dim=0)
            start_logits = start_logits.cpu().numpy()
            end_logits = end_logits.cpu().numpy()
            return [start_logits, end_logits]

    def save(self, path):
        path = Path(path)
        path.parents[0].mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)

class SWAEngine:
    def __init__(self, model, optimizer, scheduler, device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.scaler = torch.cuda.amp.GradScaler()
        self.swa_model = AveragedModel(model).to(device)
    
    def train(self, dataloader, accumulation_steps=1, grad_clip=1):
        self.model.train()
        final_loss = 0
        for i, batch in enumerate(tqdm(dataloader)):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.cuda.amp.autocast():
                outputs = self.model(**batch)
                loss = outputs.loss
                loss /= accumulation_steps
                nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            self.scaler.scale(loss).backward()

            if (i+1) % accumulation_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.model.zero_grad()
            if self.scheduler is not None:
                self.scheduler.step()
            final_loss += loss.item()        
        return final_loss / len(dataloader)
    
    def evaluate(self, dataloader):
        with torch.no_grad():
            self.model.eval()
            final_loss = 0
            for batch in tqdm(dataloader):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                final_loss += loss.item()
            
            return final_loss / len(dataloader)

    def predict(self, dataloader):
        with torch.no_grad():
            start_logits = []
            end_logits = []
            self.swa_model.eval()
            for batch in tqdm(dataloader):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.swa_model(**batch)
                start_logits.append(outputs.start_logits)
                end_logits.append(outputs.end_logits)
            start_logits = torch.cat(start_logits, dim=0)
            end_logits = torch.cat(end_logits, dim=0)
            start_logits = start_logits.cpu().numpy()
            end_logits = end_logits.cpu().numpy()
            return [start_logits, end_logits]

    def train_evaluate(self, train_dataloader, predict_dataloader, data_retriever, eval_steps, best_score, save_path, accumulation_steps=1, grad_clip=1):
        eval_steps //= data_retriever.batch_size
        self.model.train()
        self.swa_model.train()
        tloss = 0
        for i, batch in enumerate(tqdm(train_dataloader)):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.cuda.amp.autocast():
                outputs = self.model(**batch)
                loss = outputs.loss
                loss /= accumulation_steps
                nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            self.scaler.scale(loss).backward()

            if (i+1) % accumulation_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.model.zero_grad()
            if self.scheduler is not None:
                self.scheduler.step()
            tloss += loss.item()
            if (i+1) % eval_steps == 0:
                self.swa_model.update_parameters(self.model)
                raw_predictions = self.predict(predict_dataloader)
                score, lang_scores = data_retriever.evaluate_jaccard(raw_predictions)
                if score > best_score:
                    best_score = score
                    if save_path is not None:
                        self.save(save_path)
                print(f'batch {i+1}, tloss {tloss / eval_steps}, vscore {score}, best score {best_score}')
                tloss = 0
        torch.optim.swa_utils.update_bn(train_dataloader, self.swa_model)
        return best_score

    def save(self, path):
        path = Path(path)
        path.parents[0].mkdir(parents=True, exist_ok=True)
        # torch.save(self.model.state_dict(), path)
        torch.save(self.swa_model.state_dict(), path)