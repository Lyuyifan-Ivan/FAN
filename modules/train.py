import torch
from torch import optim
from tqdm import tqdm
from seqeval.metrics import classification_report
from transformers.optimization import get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
class BaseTrainer(object):
    def train(self):
        raise NotImplementedError()

    def evaluate(self):
        raise NotImplementedError()

    def test(self):
        raise NotImplementedError()


class NERTrainer(BaseTrainer):
    def __init__(self, train_data=None, dev_data=None, test_data=None, model=None, processor=None, label_map=None, args=None, logger=None,  writer=None) -> None:
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data
        self.model = model
        self.processor = processor
        self.logger = logger
        self.label_map = label_map
        self.writer = writer
        self.refresh_step = 2
        self.best_dev_metric = 0
        self.best_test_metric = 0
        self.best_train_metric = 0
        self.best_dev_epoch = None
        self.best_test_epoch = None
        self.best_train_epoch = None
        self.optimizer = None
        if self.train_data is not None:
            self.train_num_steps = len(self.train_data) * args.num_epochs
        self.step = 0
        self.args = args

    def train(self):
        self.multiModal_before_train()
        self.step = 0
        self.model.train()
        self.logger.info("***** Running training *****")
        self.logger.info("  Num instance = %d", len(self.train_data)*self.args.batch_size)
        self.logger.info("  Num epoch = %d", self.args.num_epochs)
        self.logger.info("  Batch size = %d", self.args.batch_size)
        self.logger.info("  Bert learning rate = {}".format(self.args.bert_lr))
        self.logger.info("  CRF learning rate = {}".format(self.args.crf_lr))
        self.logger.info("  Other learning rate = {}".format(self.args.other_lr))
        self.logger.info("  Evaluate begin = %d", self.args.eval_begin_epoch)

        if self.args.load_path is not None:  # load model from load_path
            self.logger.info("Loading model from {}".format(self.args.load_path))
            self.model.load_state_dict(torch.load(self.args.load_path))
            self.logger.info("Load model successful!")

        with tqdm(total=self.train_num_steps, postfix='loss:{0:<6.5f}', leave=False, dynamic_ncols=True, initial=self.step) as pbar:
            self.pbar = pbar
            avg_loss = 0
            for epoch in range(1, self.args.num_epochs+1):
                y_true, y_pred = [], []
                pbar.set_description_str(desc="Epoch {}/{}".format(epoch, self.args.num_epochs))
                for batch in self.train_data:
                    self.step += 1
                    batch = (tup.to(self.args.device)  if isinstance(tup, torch.Tensor) else tup for tup in batch)
                    attention_mask, labels, logits, loss = self._step(batch, mode="train")
                    avg_loss += loss.detach().cpu().item()
 
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    if isinstance(logits, torch.Tensor):    # CRF return lists
                        logits = logits.argmax(-1).detach().cpu().numpy()  # batch, seq, 1
                    label_ids = labels.to('cpu').numpy()
                    input_mask = attention_mask.to('cpu').numpy()
                    label_map = {idx:label for label, idx in self.label_map.items()}
                    
                    for row, mask_line in enumerate(input_mask):
                        true_label = []
                        true_predict = []
                        for column, mask in enumerate(mask_line):
                            if column == 0:
                                continue
                            if mask:
                                if label_map[label_ids[row][column]] != "X" and label_map[label_ids[row][column]] != "[SEP]":
                                    true_label.append(label_map[label_ids[row][column]])
                                    true_predict.append(label_map[logits[row][column]])
                            else:
                                break
                        y_true.append(true_label)
                        y_pred.append(true_predict)

                    if self.step % self.refresh_step == 0:
                        avg_loss = float(avg_loss) / self.refresh_step
                        print_output = "loss:{:<6.5f}".format(avg_loss)
                        pbar.update(self.refresh_step)
                        pbar.set_postfix_str(print_output)
                        if self.writer:
                            self.writer.add_scalar(tag='train_loss', scalar_value=avg_loss, global_step=self.step)    # tensorbordx
                        avg_loss = 0

                results = classification_report(y_true, y_pred, digits=4) 
                self.logger.info("***** Train Eval results *****")
                self.logger.info("\n%s", results)
                f1_score = float(results.split('\n')[-4].split('      ')[0].split('    ')[3])
                if self.writer:
                    self.writer.add_scalar(tag='train_f1', scalar_value=f1_score, global_step=epoch)    # tensorbordx
                self.logger.info("Epoch {}/{}, best train f1: {}, best epoch: {}, current train f1 score: {}."\
                            .format(epoch, self.args.num_epochs, self.best_train_metric, self.best_train_epoch, f1_score))
                if f1_score > self.best_train_metric:
                    self.best_train_metric = f1_score
                    self.best_train_epoch = epoch

                if epoch >= self.args.eval_begin_epoch:
                    self.evaluate(epoch)   # generator to dev.

            torch.cuda.empty_cache()
            pbar.close()
            self.pbar = None
            self.logger.info("Get best dev performance at epoch {}, best dev f1 score is {}".format(self.best_dev_epoch, self.best_dev_metric))
            self.logger.info("Get best test performance at epoch {}, best test f1 score is {}".format(self.best_test_epoch, self.best_test_metric))

    def evaluate(self, epoch):
        self.model.eval()
        self.logger.info("***** Running evaluate *****")
        self.logger.info("  Num instance = %d", len(self.dev_data)*self.args.batch_size)
        self.logger.info("  Batch size = %d", self.args.batch_size)

        y_true, y_pred = [], []
        step = 0
        with torch.no_grad():
            with tqdm(total=len(self.dev_data), leave=False, dynamic_ncols=True) as pbar:
                pbar.set_description_str(desc="Dev")
                total_loss = 0
                for batch in self.dev_data:
                    step += 1
                    batch = (tup.to(self.args.device)  if isinstance(tup, torch.Tensor) else tup for tup in batch)  # to cpu/cuda device
                    attention_mask, labels, logits, loss = self._step(batch, mode="dev")    # logits: batch, seq, num_labels
                    total_loss += loss.detach().cpu().item()

                    if isinstance(logits, torch.Tensor):
                        logits = logits.argmax(-1).detach().cpu().numpy()  # batch, seq, 1
                    label_ids = labels.detach().cpu().numpy()
                    input_mask = attention_mask.detach().cpu().numpy()
                    label_map = {idx:label for label, idx in self.label_map.items()}
                    for row, mask_line in enumerate(input_mask):
                        true_label = []
                        true_predict = []
                        for column, mask in enumerate(mask_line):
                            if column == 0: continue
                            if mask:
                                if label_map[label_ids[row][column]] != "X" and label_map[label_ids[row][column]] != "[SEP]":
                                    true_label.append(label_map[label_ids[row][column]])
                                    true_predict.append(label_map[logits[row][column]])
                            else: break
                        y_true.append(true_label)
                        y_pred.append(true_predict)

                    pbar.update()
                pbar.close()
                results = classification_report(y_true, y_pred, digits=4)  
                self.logger.info("***** Dev Eval results *****")
                self.logger.info("\n%s", results)
                f1_score = float(results.split('\n')[-4].split('      ')[-2].split('    ')[-1]) 
                if self.writer:
                    self.writer.add_scalar(tag='dev_f1', scalar_value=f1_score, global_step=epoch)    # tensorbordx
                    self.writer.add_scalar(tag='dev_loss', scalar_value=total_loss/step, global_step=epoch)    # tensorbordx

                self.logger.info("Epoch {}/{}, best dev f1: {}, best epoch: {}, current dev f1 score: {}."\
                            .format(epoch, self.args.num_epochs, self.best_dev_metric, self.best_dev_epoch, f1_score))
                if f1_score >= self.best_dev_metric:  # this epoch get best performance
                    self.logger.info("Get better performance at epoch {}".format(epoch))
                    self.best_dev_epoch = epoch
                    self.best_dev_metric = f1_score # update best metric(f1 score)
                    if self.args.save_path is not None:
                        torch.save(self.model.state_dict(), self.args.save_path+"/best_model.pth")
                        self.logger.info("Save best model at {}".format(self.args.save_path))
        self.model.train()

    def test(self):
        self.model.eval()
        self.logger.info("\n***** Running testing *****")
        self.logger.info("  Num instance = %d", len(self.test_data)*self.args.batch_size)
        self.logger.info("  Batch size = %d", self.args.batch_size)

        if self.args.load_path is not None:  # load model from load_path
            self.logger.info("Loading model from {}".format(self.args.load_path))
            self.model.load_state_dict(torch.load(self.args.load_path))
            self.logger.info("Load model successful!")
        y_true, y_pred = [], []
        with torch.no_grad():
            with tqdm(total=len(self.test_data), leave=False, dynamic_ncols=True) as pbar:
                pbar.set_description_str(desc="Testing")
                total_loss = 0
                for batch in self.test_data:
                    batch = (tup.to(self.args.device)  if isinstance(tup, torch.Tensor) else tup for tup in batch)  # to cpu/cuda device
                    attention_mask, labels, logits, loss = self._step(batch, mode="dev")    # logits: batch, seq, num_labels
                    total_loss += loss.detach().cpu().item()

                    if isinstance(logits, torch.Tensor): 
                        logits = logits.argmax(-1).detach().cpu().tolist()  # batch, seq, 1
                    label_ids = labels.detach().cpu().numpy()
                    input_mask = attention_mask.detach().cpu().numpy()
                    label_map = {idx:label for label, idx in self.label_map.items()}
                    for row, mask_line in enumerate(input_mask):
                        true_label = []
                        true_predict = []
                        for column, mask in enumerate(mask_line):
                            if column == 0:
                                continue
                            if mask:
                                if label_map[label_ids[row][column]] != "X" and label_map[label_ids[row][column]] != "[SEP]":
                                    true_label.append(label_map[label_ids[row][column]])
                                    true_predict.append(label_map[logits[row][column]])
                            else:
                                break
                        y_true.append(true_label)
                        y_pred.append(true_predict)
                    pbar.update()
                # evaluate done
                pbar.close()
                results = classification_report(y_true, y_pred, digits=4) 

                self.logger.info("***** Test Eval results *****")
                self.logger.info("\n%s", results)
                f1_score = float(results.split('\n')[-4].split('      ')[-2].split('    ')[-1])
                if self.writer:
                    self.writer.add_scalar(tag='test_f1', scalar_value=f1_score)    # tensorbordx
                    self.writer.add_scalar(tag='test_loss', scalar_value=total_loss/len(self.test_data))    # tensorbordx
                total_loss = 0
                self.logger.info("Test f1 score: {}.".format(f1_score))
                    
        self.model.train()
        
    def _step(self, batch, mode="train"):
        input_ids, token_type_ids, attention_mask, labels, images = batch
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels, images=images)
        logits, loss = output.logits, output.loss
        return attention_mask, labels, logits, loss


    def multiModal_before_train(self):
        parameters = []
        params = {'lr': self.args.bert_lr, 'weight_decay': 1e-2}
        params['params'] = []
        for name, param in self.model.named_parameters():
            if 'bert' in name or 'Cross_atten' in name:
                params['params'].append(param)
        parameters.append(params)
    
        params = {'lr': self.args.crf_lr, 'weight_decay': 1e-2}
        params['params'] = []
        for name, param in self.model.named_parameters():
            if 'crf' in name:
                params['params'].append(param)
        parameters.append(params)
    
        params = {'lr': self.args.other_lr, 'weight_decay': 1e-2}
        params['params'] = []
        for name, param in self.model.named_parameters():
            if 'cross_atten' in name or 'classifier' in name or 'image_model' in name:
                params['params'].append(param)
        parameters.append(params)

        self.optimizer = optim.AdamW(parameters, amsgrad=True)

        self.scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer, 
                                                            num_warmup_steps=self.args.warmup_ratio*self.train_num_steps, 
                                                            num_training_steps=self.train_num_steps)
        self.model.to(self.args.device)
