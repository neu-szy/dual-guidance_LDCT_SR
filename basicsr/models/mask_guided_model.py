import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel


@MODEL_REGISTRY.register()
class Mask_Guided_Model(BaseModel):
    def __init__(self, opt):
        super(Mask_Guided_Model, self).__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])  # 在这里设置SR网络的参数
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g wi
            # th Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('hrLD_pixel_opt'):
            self.cri_pix_hrLD = build_loss(train_opt['hrLD_pixel_opt']).to(self.device)
        else:
            self.cri_pix_hrLD = None

        if train_opt.get('LRnd_pixel_opt'):
            self.cri_pix_LRnd = build_loss(train_opt['LRnd_pixel_opt']).to(self.device)
        else:
            self.cri_pix_LRnd = None

        if train_opt.get('hrnd_pixel_opt'):
            self.cri_pix_hrnd = build_loss(train_opt['hrnd_pixel_opt']).to(self.device)
        else:
            self.cri_pix_hrnd = None
        if train_opt.get('tv_opt'):
            self.cri_tv = build_loss(train_opt['tv_opt']).to(self.device)
        else:
            self.cri_tv = None
        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None
        if train_opt.get('gan_opt'):
            self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)
        else:
            self.cri_gan = None

        if self.cri_pix_hrLD is None and self.cri_pix_LRnd is None and self.cri_pix_hrnd is None and self.cri_tv is None and self.cri_perceptual is None and self.cri_gan is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'mask' in data:
            self.mask = data['mask'].to(self.device)
        else:
            self.mask = self.lq
        if 'avg_ct' in data:
            self.avg_ct = data['avg_ct'].to(self.device)
        else:
            self.avg_ct = self.lq
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        if 'gt_lr' in data:
            self.gt_lr = data['gt_lr'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output_hrLD, self.output_LRnd, self.output_hrnd = self.net_g(self.lq, self.avg_ct, self.mask)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix_hrLD:
            l_pix_hrLD = self.cri_pix_hrLD(self.output_hrLD, self.gt)
            l_total += l_pix_hrLD
            loss_dict['l_pix_hrLD'] = l_pix_hrLD
        if self.cri_pix_LRnd:
            l_pix_LRnd = self.cri_pix_hrLD(self.output_LRnd, self.gt_lr)
            l_total += l_pix_LRnd
            loss_dict['l_pix_LRnd'] = l_pix_LRnd
        if self.cri_pix_hrnd:
            l_pix_hrnd = self.cri_pix_hrnd(self.output_hrnd, self.gt)
            l_total += l_pix_hrnd
            loss_dict['l_pix_hrnd'] = l_pix_hrnd
        if self.cri_tv:
            l_tv = self.cri_tv(self.output_hrnd, self.gt)
            l_total += l_tv
            loss_dict['l_tv'] = l_tv
        if self.cri_gan:
            l_gan_pred = self.cri_gan(self.output_hrnd, False)
            l_gan_gt = self.cri_gan(self.gt, True)
            l_total += l_gan_pred
            l_total += l_gan_gt
            loss_dict['l_gan'] = l_gan_pred + l_gan_gt

        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output_hrnd, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        l_total.backward()

        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
        return l_total.item()

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                _, _, self.output_hrnd = self.net_g_ema(self.lq, self.avg_ct, self.mask)
        else:
            self.net_g.eval()
            with torch.no_grad():
                _, _, self.output_hrnd = self.net_g(self.lq, self.avg_ct, self.mask)
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics and not hasattr(self, 'metric_results'):  # only execute in the first run
            self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
        # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
        self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output_hrnd
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            # TB.add_scalar(metric, value, current_iter)
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output_hrnd.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)


@MODEL_REGISTRY.register()
class MaskGuidedLoopModel(BaseModel):
    def __init__(self, opt):
        super(MaskGuidedLoopModel, self).__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])  # 在这里设置SR网络的参数
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g wi
            # th Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('sr_pixel_opt'):
            self.cri_pix_sr = build_loss(train_opt['sr_pixel_opt']).to(self.device)
        else:
            self.cri_pix_sr = None

        if train_opt.get('dn_pixel_opt'):
            self.cri_pix_dn = build_loss(train_opt['dn_pixel_opt']).to(self.device)
        else:
            self.cri_pix_dn = None

        # if train_opt.get('hrnd_pixel_opt'):
        #     self.cri_pix_hrnd = build_loss(train_opt['hrnd_pixel_opt']).to(self.device)
        # else:
        #     self.cri_pix_hrnd = None
        if train_opt.get('tv_opt'):
            self.cri_tv = build_loss(train_opt['tv_opt']).to(self.device)
        else:
            self.cri_tv = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if self.cri_pix_sr is None and self.cri_pix_dn is None and self.cri_tv is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'mask' in data:
            self.mask = data['mask'].to(self.device)
        else:
            self.mask = self.lq
        # if 'avg_ct' in data:
        #     self.avg_ct = data['avg_ct'].to(self.device)
        # else:
        #     self.avg_ct = self.lq
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        if 'gt_lr' in data:
            self.gt_lr = data['gt_lr'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output_DNs, self.output_SRs = self.net_g(x=self.lq, mask=self.mask)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix_dn:
            l_pix_hrdn = 0
            for dn in self.output_DNs:
                l_pix_hrdn += self.cri_pix_dn(dn, self.gt_lr)
            l_total += l_pix_hrdn
            loss_dict['l_pix_dn'] = l_pix_hrdn
        if self.cri_pix_sr:
            l_pix_sr = 0
            for sr in self.output_SRs:
                l_pix_sr = self.cri_pix_sr(sr, self.gt)
            l_total += l_pix_sr
            loss_dict['l_pix_sr'] = l_pix_sr

        if self.cri_tv:
            l_tv = 0
            for sr in self.output_SRs:
                l_tv += self.cri_tv(sr, self.gt)
            l_total += l_tv
            loss_dict['l_tv'] = l_tv

        # perceptual loss
        if self.cri_perceptual:
            l_percep = 0
            l_style = 0
            for sr in self.output_SRs:
                l_percep_style = self.cri_perceptual(sr, self.gt)
                if l_percep is not None:
                    l_percep += l_percep_style[0]
                    l_total += l_percep
                    loss_dict['l_percep'] = l_percep
                if l_style is not None:
                    l_style += l_percep_style[1]
                    l_total += l_style
                    loss_dict['l_style'] = l_style

        l_total.backward()

        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
        return l_total.item()

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                _, self.output_SRs = self.net_g_ema(x=self.lq, mask=self.mask)
        else:
            self.net_g.eval()
            with torch.no_grad():
                _, self.output_SRs = self.net_g(x=self.lq, mask=self.mask)
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics and not hasattr(self, 'metric_results'):  # only execute in the first run
            self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
        # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
        self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.mask
            del self.output_SRs
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            # TB.add_scalar(metric, value, current_iter)
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output_SRs[-1].detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)