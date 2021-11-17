import transformers
from transformers import AutoModel, AutoConfig, AutoModelForQuestionAnswering
import torch
import torch.nn as nn
import numpy as np
from transformers.modeling_outputs import QuestionAnsweringModelOutput

class Output:
    pass

class ChaiiModel(nn.Module):
    def __init__(self, model_name, config):
        super(ChaiiModel, self).__init__()
        self.transformer = AutoModel.from_pretrained(model_name, config=config)
        self.output = nn.Linear(config.hidden_size, 2)
    def forward(self, input_ids, attention_mask, start_positions=None, end_positions=None):
        transformer_out = self.transformer(input_ids, attention_mask)
        sequence_output = transformer_out[0]
        logits = self.output(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        if start_positions is not None and end_positions is not None:
            loss_fct = nn.CrossEntropyLoss()
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            # total_loss = (start_loss + end_loss) / 2
            total_loss = (start_loss*end_loss) ** 0.5
        else:
            total_loss = None

        output = Output()
        output.loss = total_loss
        output.start_logits = start_logits
        output.end_logits = end_logits
        return output

class ChaiiModelLoadHead(nn.Module):
    def __init__(self, model_name, config):
        super(ChaiiModelLoadHead, self).__init__()
        self.transformer = AutoModelForQuestionAnswering.from_pretrained(model_name, config=config)
    def forward(self, **inputs):
        output = self.transformer(**inputs)
        start_positions = inputs.get('start_positions', None)
        end_positions = inputs.get('end_positions', None)
        if start_positions is not None and end_positions is not None:
            loss_fct = nn.CrossEntropyLoss()
            start_loss = loss_fct(output.start_logits, start_positions)
            end_loss = loss_fct(output.end_logits, end_positions)
            total_loss = (start_loss*end_loss) ** 0.5
        else:
            total_loss = None
        myoutput = Output()
        myoutput.loss = total_loss
        myoutput.start_logits = output.start_logits
        myoutput.end_logits = output.end_logits
        return myoutput

# https://github.com/Danielhuxc/CLRP-solution/blob/main/components/model.py
def init_params(module_lst):
    for module in module_lst:
        for param in module.parameters():
            if param.dim() > 1:
                torch.nn.init.xavier_uniform_(param)
    return

class ChaiiModel1008(nn.Module):
    def __init__(self,model_dir, dropout=0.2, hdropout=0.5):
        super().__init__()

        #load base model
        config = AutoConfig.from_pretrained(model_dir)
        config.update({"output_hidden_states":True, 
                       "hidden_dropout_prob": 0.0,
                       "layer_norm_eps": 1e-7})                       
        
        self.base = AutoModel.from_pretrained(model_dir, config=config)  
        
        dim = self.base.encoder.layer[0].output.dense.bias.shape[0]
        
        self.dropout = nn.Dropout(p=dropout)
        self.high_dropout = nn.Dropout(p=hdropout)
        
        #weights for weighted layer average
        n_weights = 24
        weights_init = torch.zeros(n_weights).float()
        weights_init.data[:-1] = -3
        self.layer_weights = torch.nn.Parameter(weights_init)
        
#         #attention head
#         self.attention = nn.Sequential(
#             nn.Linear(1024, 1024),            
#             nn.Tanh(),
#             nn.Linear(1024, 1),
#             nn.Softmax(dim=1)
#         ) 
#         self.cls = nn.Sequential(
#             nn.Linear(dim,1)
#         )
#         init_params([self.cls,self.attention])
        self.output = nn.Linear(config.hidden_size, 2)
        init_params([self.output])
        
    def reini_head(self):
        init_params([self.cls,self.attention])
        return 
        
    def forward(self, input_ids, attention_mask, start_positions=None, end_positions=None):
        base_output = self.base(input_ids=input_ids,
                                      attention_mask=attention_mask)

        #weighted average of all encoder outputs
        cls_outputs = torch.stack(
            [self.dropout(layer) for layer in base_output['hidden_states'][-24:]], dim=0
        )
        # print('cls_outputs', cls_outputs.shape) # nlayers * bs * seqlen * hiddim
        cls_output = (torch.softmax(self.layer_weights, dim=0).unsqueeze(1).unsqueeze(1).unsqueeze(1) * cls_outputs).sum(0)
        # print('cls_outputs', cls_output.shape) # bs * seqlen * hiddim
        #multisample dropout
        logits = torch.mean(
            torch.stack(
                [self.output(self.high_dropout(cls_output)) for _ in range(5)],
                dim=0,
            ),
            dim=0,
        )
        # print('logits', logits.shape) # bs * seqlen * 2
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
        
        if start_positions is not None and end_positions is not None:
            loss_fct = nn.CrossEntropyLoss()
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
#             total_loss = (start_loss*end_loss) ** 0.5
        else:
            total_loss = None

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
        )

class ChaiiRemBert(nn.Module):
    def __init__(self, model_dir, dropout=0.2, hdropout=0.5, nlast=2):
        super(ChaiiRemBert, self).__init__()
        #load base model
        self.config = AutoConfig.from_pretrained(model_dir)
        self.config.update({
            'hidden_dropout_prob': dropout,
            'attention_probs_dropout_prob': dropout,
        })
        self.nlast = nlast
        self.base = AutoModel.from_pretrained(model_dir, config=self.config)
        self.high_dropout = nn.Dropout(p=hdropout)
        self.output = nn.Linear(self.nlast * self.config.hidden_size, 2)
        torch.nn.init.normal_(self.output.weight, std=0.02)

    def forward(self, input_ids, attention_mask, token_type_ids, start_positions=None, end_positions=None):
        out = self.base(input_ids, attention_mask, token_type_ids, output_hidden_states=True)
        if self.nlast == 1:
            out = out.last_hidden_state
        else:
            out = torch.cat(out.hidden_states[-self.nlast:], dim=-1)
        
        # Multisample Dropout: https://arxiv.org/abs/1905.09788
        logits = torch.mean(torch.stack([self.output(self.high_dropout(out)) for _ in range(5)], dim=0), dim=0)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        if start_positions is not None and end_positions is not None:
            loss_fct = nn.CrossEntropyLoss()
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            # total_loss = (start_loss + end_loss) / 2
            total_loss = (start_loss*end_loss) ** 0.5
        else:
            total_loss = None

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
        )