import logging
import os
import sys

import pandas as pd
import torch
import torch.nn as nn
from data.dataset import ABSADataset
from scipy.sparse.construct import random
from sklearn import metrics
from torch.utils.data import DataLoader, random_split
from transformers import BertModel, BertTokenizer, AdamW

from models.bert_spc import BERT_SPC

logging.basicConfig(
    filename='train-logs.txt',
    filemode='a',
    level=logging.INFO
)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def train(model, criterion, log_step, optimizer,
		  train_data_loader, val_data_loader,
		  device, num_epoch=4, patience=5):
	max_val_acc = 0
	max_val_f1 = 0
	max_val_epoch = 0
	global_step = 0
	path = None
	for i_epoch in range(num_epoch):
		logger.info('>' * 100)
		logger.info('epoch: {}'.format(i_epoch))
		n_correct, n_total, loss_total = 0, 0, 0
		# switch model to training mode
		model.train()
		for i_batch, batch in enumerate(train_data_loader):
			global_step += 1
			# clear gradient accumulators
			optimizer.zero_grad()

			inputs = [batch['text_indices'].to(
				device), batch['segment_indices'].to(device)]
			outputs = model(inputs)
			targets = batch['polarity'].to(device)

			loss = criterion(outputs, targets)
			loss.backward()
			optimizer.step()

			n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
			n_total += len(outputs)
			loss_total += loss.item() * len(outputs)
			if global_step % log_step == 0:
				train_acc = n_correct / n_total
				train_loss = loss_total / n_total
				logger.info('loss: {:.4f}, acc: {:.4f}'.format(
					train_loss, train_acc))

		val_acc, val_f1 = evaluate_acc_f1(
			model, val_data_loader, device)
		logger.info(
			'> val_acc: {:.4f}, val_f1: {:.4f}'.format(val_acc, val_f1))
		if val_acc > max_val_acc:
			max_val_acc = val_acc
			max_val_epoch = i_epoch
			if not os.path.exists('state_dict'):
				os.mkdir('state_dict')
			path = 'models/bert-spc_val_acc_{0}.pt'.format(
				round(val_acc, 4))
			torch.save(model.state_dict(), path)
			logger.info('>> saved: {}'.format(path))
		if val_f1 > max_val_f1:
			max_val_f1 = val_f1
		if i_epoch - max_val_epoch >= patience:
			print('>> early stop.')
			break

	return path


def evaluate_acc_f1(model, data_loader, device):
	n_correct, n_total = 0, 0
	t_targets_all, t_outputs_all = None, None
	# switch model to evaluation mode
	model.eval()
	with torch.no_grad():
		for i_batch, t_batch in enumerate(data_loader):
			t_inputs = [t_batch['text_indices'].to(
				device), t_batch['segment_indices'].to(device)]

			t_targets = t_batch['polarity'].to(device)
			t_outputs = model(t_inputs)

			n_correct += (torch.argmax(t_outputs, -1)
						  == t_targets).sum().item()
			n_total += len(t_outputs)

			if t_targets_all is None:
				t_targets_all = t_targets
				t_outputs_all = t_outputs
			else:
				t_targets_all = torch.cat(
					(t_targets_all, t_targets), dim=0)
				t_outputs_all = torch.cat(
					(t_outputs_all, t_outputs), dim=0)

		acc = n_correct / n_total
		f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(
			t_outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro')
		return acc, f1


if __name__ == "__main__":
	bert_model = BertModel.from_pretrained('bert-base-uncased')
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
	data_path = '/srv/home/ahuja/Enterpret/data/train.csv'
	train_df = pd.read_csv(data_path)
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	dataset = ABSADataset(train_df, tokenizer)
	val_set_len = int(0.2*len(dataset))
	train_data, val_data = random_split(dataset, (len(dataset) - val_set_len, val_set_len))
	model = BERT_SPC(bert_model, dropout_prob=0.1).to(device)  # Model defined
	criterion = nn.CrossEntropyLoss()
	_params = filter(lambda p: p.requires_grad, model.parameters())
	optimizer = AdamW(_params, lr=3e-5, weight_decay=0.01)
	train_data_loader = DataLoader(
		dataset=train_data, batch_size=16, shuffle=True)
	val_data_loader = DataLoader(
		dataset=val_data, batch_size=16, shuffle=False)
	path_model = train(model, criterion, 10, optimizer,
					   train_data_loader, val_data_loader, device, num_epoch = 6)