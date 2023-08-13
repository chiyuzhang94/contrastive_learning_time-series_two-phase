from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from exp.exp_basic import Exp_Basic
from models.model import Informer, InformerStack
from models.tcn import TCN
from models.lstm import LSTM
from models.tcn_moco import TCN_MoCo_Pretrain, TCN_MoCo_Task
from models.dtcn_moco import DTCN_MoCo
from models.cost_e2e import COST_E2E
from models.dtcn import DTCN
from models.losses import hierarchical_contrastive_loss
from einops import rearrange, repeat, reduce

from data.utils import *

from utils.tools import EarlyStopping, adjust_learning_rate, adjust_learning_rate_cos
from utils.metrics import metric

import numpy as np

import torch, math
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from data.datautils import *

import os
import time, json

import warnings
warnings.filterwarnings('ignore')

class Exp_Informer(Exp_Basic):
    def __init__(self, args):
        super(Exp_Informer, self).__init__(args)
    
    def _build_model(self):
        model_dict = {
            'informer':Informer,
            'informerstack':InformerStack,
        }

        if self.args.model == "tcn-moco" or self.args.model=='dtcn-moco' or self.args.model=='lstm-moco' or self.args.model=='informer-moco':
            model = TCN_MoCo_Pretrain(
                             model_name=self.args.model,
                             input_size = self.args.enc_in,
                             hidden_size = self.args.d_model, 
                             output_size = self.args.c_out, 
                             e_layer = self.args.e_layers, 
                             pred_leng = self.args.pred_len,
                             freq = self.args.freq,
                             kernel_size = self.args.kernel_size,
                             dropout=self.args.dropout,
                             K=512,
                             T=1.00,
                             mask_rate = self.args.mask_rate,
                             l2norm=self.args.l2norm,
                             average_pool = self.args.moco_average_pool,
                             data_aug = self.args.data_aug,
                             mare = self.args.mare
                        )

        # elif self.args.model == "tcn-moco-hcl" or self.args.model=='dtcn-moco-hcl':
        #     model = TCN_MoCo_Pretrain(self.args.model,
        #                      input_size = self.args.enc_in,
        #                      hidden_size = self.args.d_model, 
        #                      output_size = self.args.c_out, 
        #                      e_layer = self.args.e_layers, 
        #                      pred_leng = self.args.pred_len,
        #                      freq = self.args.freq,
        #                      kernel_size = self.args.kernel_size,
        #                      dropout=self.args.dropout,
        #                      K=512,
        #                      T=1.00,
        #                      l2norm=self.args.l2norm,
        #                      average_pool = self.args.moco_average_pool,
        #                      data_aug = self.args.data_aug,
        #                      tempral_cl = True,
        #                      mask_rate = self.args.mask_rate,
        #                      moco_cl_weight = 0.5,
        #                      mare = self.args.mare
        #                 )

        # elif self.args.model=='dtcn':
        #     model = DTCN(input_size = self.args.enc_in,
        #                  hidden_size = self.args.d_model, 
        #                  output_size = self.args.c_out, 
        #                  e_layer = self.args.e_layers, 
        #                  pred_leng = self.args.pred_len,
        #                  mask_rate = self.args.mask_rate,
        #                  freq = self.args.freq,
        #                  kernel_size = self.args.kernel_size,
        #                  contrastive_loss=self.args.loss_lambda,
        #                  dropout=self.args.dropout
        #                  )

        # elif self.args.model == "dtcn-moco":
        #     model = DTCN_MoCo(input_size = self.args.enc_in,
        #                      hidden_size = self.args.d_model, 
        #                      output_size = self.args.c_out, 
        #                      e_layer = self.args.e_layers, 
        #                      pred_leng = self.args.pred_len,
        #                      freq = self.args.freq,
        #                      kernel_size = self.args.kernel_size,
        #                      dropout=self.args.dropout,
        #                      K=512,
        #                      T=1.00,
        #                      l2norm=self.args.l2norm,
        #                      average_pool = self.args.moco_average_pool,
        #                      data_aug = self.args.data_aug
        #                 )

        # elif self.args.model == "cost-e2e":
        #     model = COST_E2E(input_size = self.args.enc_in,
        #                      hidden_size = self.args.d_model, 
        #                      output_size = self.args.c_out, 
        #                      e_layer = self.args.e_layers, 
        #                      pred_leng = self.args.pred_len,
        #                      input_length= self.args.seq_len,
        #                      dropout=self.args.dropout
        #                 )
            
        else:
            print("Wrong model name")
        

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        print(model)
        
        return model

    def _get_data(self, flag):
        args = self.args

        data_dict = {
            'ETTh1':Dataset_ETT_hour,
            'ETTh2':Dataset_ETT_hour,
            'ETTm1':Dataset_ETT_minute,
            'ETTm2':Dataset_ETT_minute,
            'WTH':Dataset_Custom,
            'ECL':Dataset_Custom,
            'Solar':Dataset_Custom,
            'custom':Dataset_Custom,
        }
        Data = data_dict[self.args.data]
        timeenc = 0 if args.embed!='timeF' else 1

        if flag == 'test':
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size; freq=args.freq
        elif flag=='pred':
            shuffle_flag = False; drop_last = False; batch_size = 1; freq=args.detail_freq
            Data = Dataset_Pred
        else:
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size; freq=args.freq
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def _select_criterion(self):
        criterion =  nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(vali_loader):
            pred, true, _ = self._process_one_batch_task(
                vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            loss = criterion(pred.detach().cpu(), true.detach().cpu())
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train_task(self, setting):
        # smaller learning rate for fine-tuning
        # self.args.learning_rate *= 0.5

        print("learning_rate", self.args.learning_rate)

        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'val')
        test_data, test_loader = self._get_data(flag = 'test')

        model = TCN_MoCo_Task(hidden_size = self.args.d_model, 
                               output_size = self.args.c_out,
                               pred_leng=self.args.pred_len,
                               encoder=self.model,
                               dropout = self.args.dropout,
                               freeze_encoder = self.args.freeze_encoder)

        if self.args.use_multi_gpu and self.args.use_gpu:
            self.model = nn.DataParallel(model, device_ids=self.args.device_ids)
        elif self.args.use_gpu:
            self.model = model.to(self.device)

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        epoch_train_loss = []

        epo_eval_loss = []
        epo_test_loss = []
        itera_losses = []

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                
                model_optim.zero_grad()
                pred, true, loss = self._process_one_batch_task(
                    train_data, batch_x, batch_y, batch_x_mark, batch_y_mark)

                train_loss.append(loss.item())
                
                if i % 100==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                    # print("backward")

            itera_losses.extend(train_loss)
            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            epoch_train_loss.append(train_loss)
            epo_eval_loss.append(vali_loss)
            epo_test_loss.append(test_loss)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.cos_lr:
                adjust_learning_rate_cos(model_optim, epoch+1, self.args)
            else:
                adjust_learning_rate(model_optim, epoch+1, self.args)
            
        best_model_path = path+'/'+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        loss_record = {"itera_train_loss": itera_losses,
                        "epoch_train_loss": epoch_train_loss, "epo_eval_loss": epo_eval_loss, "epo_test_loss": epo_test_loss, "num_step": train_steps}

        # result save
        folder_path = self.args.des_path + '/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        torch.save(loss_record, folder_path + "task_training_log.pt")

        return self.model

    def train_pretrain(self, setting):

        def adjust_learning_rate(optimizer, lr, itera, all_iteras):
            """Decay the learning rate based on schedule"""
            lr *= 0.5 * (1. + math.cos(math.pi * itera / all_iteras))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        print(self.args.data)
        if "ETT" in self.args.data:
            dataset = "ETT/" + self.args.data
        else:
            dataset = self.args.data

        print(dataset)
        if self.args.features == "M":
            data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = load_forecast_csv(dataset, svm_evaluate = self.args.svm_evaluate)
        elif self.args.features == "S":
            data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = load_forecast_csv(dataset, univar=True, svm_evaluate = self.args.svm_evaluate)
        train_data = data[:, train_slice]

        # train_data, train_loader = self._get_data(flag = 'train')
        print("data size:", train_data.size)
        n_iters = 200 if train_data.size <= 100000 else 600

        print("n_iters: ", n_iters)

        if self.args.svm_evaluate:
            self.max_train_length = 201
            sections = train_data.shape[1] // self.max_train_length
        else:
            sections = train_data.shape[1] // self.args.seq_len #??? CoST use 201

        print("sections: ", sections)

        if sections >= 2:
            train_data = np.concatenate(split_with_nan(train_data, sections, axis=1), axis=0)

        temporal_missing = np.isnan(train_data).all(axis=-1).any(axis=0)
        if temporal_missing[0] or temporal_missing[-1]:
            train_data = centerize_vary_length_series(train_data)

        train_data = train_data[~np.isnan(train_data).all(axis=2).all(axis=1)]

        print("train_data", train_data.shape)
        
        multiplier = 1 if train_data.shape[0] >= self.args.batch_size else math.ceil(self.args.batch_size / train_data.shape[0])

        print("multiplier", multiplier)

        train_dataset = PretrainDataset(torch.from_numpy(train_data).to(torch.float), multiplier=multiplier)
        # train_dataset = torch.from_numpy(train_data).to(torch.float)
        train_loader = DataLoader(train_dataset, batch_size=min(self.args.batch_size, len(train_dataset)), shuffle=True, drop_last=True)

        print("data loader:", len(train_loader))

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        
        # model_optim = self._select_optimizer()

        model_optim = torch.optim.SGD([p for p in self.model.parameters() if p.requires_grad],
                            lr=self.args.learning_rate,
                            momentum=0.9,
                            weight_decay=1e-4)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        itera_train_loss_cl = []
        iter_count = 0

        while True:
            interrupted = False

            for batch_x in train_loader:
                
                if n_iters is not None and iter_count >= n_iters:
                    interrupted = True
                    break
                
                iter_count += 1
                model_optim.zero_grad()

                x_q, x_k = batch_x

                if self.args.svm_evaluate and x_q.size(1) > self.max_train_length:
                    window_offset = np.random.randint(x_q.size(1) - self.max_train_length + 1)
                    x_q = x_q[:, window_offset : window_offset + self.max_train_length]
                    x_k = x_k[:, window_offset : window_offset + self.max_train_length]
                
                loss = self._process_one_batch_pretrain(x_q, x_k)

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                adjust_learning_rate(model_optim, self.args.learning_rate, iter_count, n_iters)
                
                itera_train_loss_cl.append(loss.cpu().item())
                
                if iter_count % 10 == 0:
                    print("\titers: {0} | loss: {1:.7f}".format(iter_count + 1, loss.cpu().item()))
                    speed = (time.time()-time_now)/iter_count
                    print('\tspeed: {:.4f}s/iter;'.format(speed))
                    time_now = time.time()

            if interrupted:
                break
                
        # best_model_path = path+'/'+'checkpoint.pth'
        # self.model.load_state_dict(torch.load(best_model_path))
        
        loss_record = {"itera_train_loss_cl": itera_train_loss_cl}

        # result save
        folder_path = self.args.des_path + '/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        torch.save(loss_record, folder_path + "pretraining_log.pt")

        return self.model

    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')
        
        self.model.eval()
        
        preds = []
        trues = []
        
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(test_loader):
            pred, true, _ = self._process_one_batch_task(
                test_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = self.args.des_path + '/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        np.save(folder_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        # np.save(folder_path+'pred.npy', preds)
        # np.save(folder_path+'true.npy', trues)

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')
        
        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path+'/'+'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()
        
        preds = []
        
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(pred_loader):
            pred, true, _ = self._process_one_batch_task(
                pred_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.detach().cpu().numpy())

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        
        # result save
        folder_path = self.args.des_path + '/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        np.save(folder_path+'real_prediction.npy', preds)
        
        return

    def _process_one_batch_task(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
        criterion =  self._select_criterion()
        
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)

        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        contrast_loss = 0.0

        # # decoder input
        # if self.args.padding==0:
        #     dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        # elif self.args.padding==1:
        #     dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        # dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).float().to(self.device)   # first label_len time stamps have actual values, pred_len time stamps have 0 values.
        # encoder - decoder
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if 'moco' in self.args.model or "cost" in self.args.model:
                    pred, _, _  = self.model(batch_x)
                else:
                    pred, _, _ = self.model(batch_x)
        else:
            if 'moco' in self.args.model or "cost" in self.args.model:
                    pred, _, _  = self.model(batch_x)

            else:
                pred, _, _ = self.model(batch_x)
                
        if self.args.inverse:
            pred = dataset_object.inverse_transform(pred)

        f_dim = -1 if self.args.features=='MS' else 0
        batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)

        loss_mse = criterion(pred, batch_y)

        return pred, batch_y, loss_mse

    def _process_one_batch_pretrain(self, x_q, x_k):
        
        x_q = x_q.float().to(self.device)
        x_k = x_k.float().to(self.device)

        contrast_loss = 0.0

        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if 'moco' in self.args.model or "cost" in self.args.model: # cost end2end or moco 
                    _, out1, contrast_loss  = self.model(x_q, x_k)
                else:
                    _, out1, out2 = self.model(x_q)
        else:
            if 'moco' in self.args.model or "cost" in self.args.model:
                    _, _, contrast_loss  = self.model(x_q, x_k)
            else:
                _, out1, out2 = self.model(x_q)
                

        # if self.args.loss_lambda > 0.0 and 'moco' not in self.args.model and "cost" not in self.args.model:
        #     contrast_loss = hierarchical_contrastive_loss(out1, out2, self.args.l2norm)

        return contrast_loss

    def _eval_with_pooling(self, x, mask=None, slicing=None, encoding_window=None):
        # print("x", x.shape)
        out = self.model.encoder_q(x.to(self.device), mask_flag=False)

        out = out[:, -1, :].cpu()
        out = rearrange(out.cpu(), 'b d -> b () d')
            
        return out


    def encode(self, data, mask=None, encoding_window=None, casual=False, sliding_length=None, sliding_padding=0, batch_size=None):
        ''' Compute representations using the model.
        
        Args:
            data (numpy.ndarray): This should have a shape of (n_instance, n_timestamps, n_features). All missing data should be set to NaN.
            mask (str): The mask used by encoder can be specified with this parameter. This can be set to 'binomial', 'continuous', 'all_true', 'all_false' or 'mask_last'.
            encoding_window (Union[str, int]): When this param is specified, the computed representation would the max pooling over this window. This can be set to 'full_series', 'multiscale' or an integer specifying the pooling kernel size.
            casual (bool): When this param is set to True, the future informations would not be encoded into representation of each timestamp.
            sliding_length (Union[int, NoneType]): The length of sliding window. When this param is specified, a sliding inference would be applied on the time series.
            sliding_padding (int): This param specifies the contextual data length used for inference every sliding windows.
            batch_size (Union[int, NoneType]): The batch size used for inference. If not specified, this would be the same batch size as training.
            
        Returns:
            repr: The representations for data.
        '''
        encoding_window = None
        slicing = None

        assert self.model is not None, 'please train or load a net first'
        
        assert data.ndim == 3
        if batch_size is None:
            batch_size = self.batch_size
        n_samples, ts_l, _ = data.shape

        org_training = self.model.training
        self.model.eval()
        
        dataset = TensorDataset(torch.from_numpy(data).to(torch.float))
        loader = DataLoader(dataset, batch_size=batch_size)
        
        with torch.no_grad():
            output = []
            for batch in loader:
                x = batch[0]
                if sliding_length is not None:
                    reprs = []
                    if n_samples < batch_size:
                        calc_buffer = []
                        calc_buffer_l = 0
                    for i in range(0, ts_l, sliding_length):
                        l = i - sliding_padding
                        r = i + sliding_length + (sliding_padding if not casual else 0)
                        x_sliding = torch_pad_nan(
                            x[:, max(l, 0) : min(r, ts_l)],
                            left=-l if l<0 else 0,
                            right=r-ts_l if r>ts_l else 0,
                            dim=1
                        )
                        if n_samples < batch_size:
                            if calc_buffer_l + n_samples > batch_size:
                                out = self._eval_with_pooling(
                                    torch.cat(calc_buffer, dim=0),
                                    mask,
                                    slicing=slicing,
                                    encoding_window=encoding_window
                                )
                                reprs += torch.split(out, n_samples)
                                calc_buffer = []
                                calc_buffer_l = 0
                            calc_buffer.append(x_sliding)
                            calc_buffer_l += n_samples
                        else:
                            out = self._eval_with_pooling(
                                x_sliding,
                                mask,
                                slicing=slicing,
                                encoding_window=encoding_window
                            )
                            reprs.append(out)

                    if n_samples < batch_size:
                        if calc_buffer_l > 0:
                            out = self._eval_with_pooling(
                                torch.cat(calc_buffer, dim=0),
                                mask,
                                slicing=slicing,
                                encoding_window=encoding_window
                            )
                            reprs += torch.split(out, n_samples)
                            calc_buffer = []
                            calc_buffer_l = 0
                    
                    out = torch.cat(reprs, dim=1)
                    if encoding_window == 'full_series':
                        out = F.max_pool1d(
                            out.transpose(1, 2).contiguous(),
                            kernel_size = out.size(1),
                        ).squeeze(1)
                else:
                    out = self._eval_with_pooling(x, mask, encoding_window=encoding_window)
                    if encoding_window == 'full_series':
                        out = out.squeeze(1)
                        
                output.append(out)
                
            output = torch.cat(output, dim=0)
            
        self.model.train(org_training)
        return output.numpy()
