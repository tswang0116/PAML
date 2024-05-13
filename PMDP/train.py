import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.special import comb

from retrieval_model import Retrieval_Model
from model import ImageSubnet, TextSubnet, DiffusionModel, Discriminator, PretrainingModel, GANLoss
from utils import calc_hamming, return_results, CalcMap, CalcPSR, mkdir_p, image_normalization, image_restoration

import pdb

class PMDP(nn.Module):
    def __init__(self, args, Dcfg):
        super(PMDP, self).__init__()
        self.args = args
        self.Dcfg = Dcfg
        self._build_model(self.args, self.Dcfg)
        self._save_setting()
    
    def _build_model(self, args, Dcfg):
        self.retrieval_model = Retrieval_Model(self.args.retrieval_method, self.args.dataset, self.args.retrieval_bit, self.args.retrieval_models_path, self.args.dataset_path)
        self.retrieval_model.eval().cuda()
        self.image_subnet = ImageSubnet(self.args.subnet_bit).cuda()
        self.text_subnet = TextSubnet(self.Dcfg.tag_dim, self.args.subnet_bit).cuda()
        self.diffusion_model = DiffusionModel().cuda()
        self.discriminator = Discriminator().cuda()
        self.pretrain_model = PretrainingModel().cuda()
        self.criterionGAN = GANLoss().cuda()

    def _save_setting(self):
        output_dir = self.args.dataset + '_' + str(self.args.retrieval_bit) + '_' + self.args.retrieval_method  + '_' + str(self.args.subnet_bit)
        self.output = os.path.join(self.args.output_path, output_dir)
        self.model_dir = os.path.join(self.output, 'Model')
        self.image_dir = os.path.join(self.output, 'Image')
        mkdir_p(self.model_dir)
        mkdir_p(self.image_dir)

    def save_subnet(self):
        torch.save(self.image_subnet.state_dict(), os.path.join(self.model_dir, 'image_subnet.pth'))
        torch.save(self.text_subnet.state_dict(), os.path.join(self.model_dir, 'text_subnet.pth'))
    
    def save_diffusion_model(self):
        torch.save(self.diffusion_model.state_dict(), os.path.join(self.model_dir, 'diffusion_model.pth'))
        torch.save(self.discriminator.state_dict(), os.path.join(self.model_dir, 'discriminator.pth'))
    
    def load_subnet(self):
        self.image_subnet.load_state_dict(torch.load(os.path.join(self.model_dir, 'image_subnet.pth')))
        self.text_subnet.load_state_dict(torch.load(os.path.join(self.model_dir, 'text_subnet.pth')))
        self.image_subnet.eval().cuda()
        self.text_subnet.eval().cuda()

    def load_diffusion_model(self):
        self.diffusion_model.load_state_dict(torch.load(os.path.join(self.model_dir, 'diffusion_model.pth')))
        self.diffusion_model.eval().cuda()
    
    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
    
    def test_retrieval_model(self, Db_I, Db_T, Db_L, Te_I, Te_T, Te_L):
        print('test retrieval model...')
        IqB = self.retrieval_model.generate_image_hashcode(Te_I).cuda()
        TqB = self.retrieval_model.generate_text_hashcode(Te_T).cuda()
        IdB = self.retrieval_model.generate_image_hashcode(Db_I).cuda()
        TdB = self.retrieval_model.generate_text_hashcode(Db_T).cuda()
        I2T_map = CalcMap(IqB, TdB, Te_L, Db_L, self.args.map_k)
        T2I_map = CalcMap(TqB, IdB, Te_L, Db_L, self.args.map_k)
        I2I_map = CalcMap(IqB, IdB, Te_L, Db_L, self.args.map_k)
        T2T_map = CalcMap(TqB, TdB, Te_L, Db_L, self.args.map_k)
        print('I2T_map: {:.4f}'.format(I2T_map))
        print('T2I_map: {:.4f}'.format(T2I_map))
        print('I2I_map: {:.4f}'.format(I2I_map))
        print('T2T_map: {:.4f}'.format(T2T_map))
        T2I_psr_RO = CalcPSR(TqB, IqB, IdB, self.args.psr_k)
        I2I_psr_RO = CalcPSR(IqB, IqB, IdB, self.args.psr_k)
        I2T_psr_RO = CalcPSR(IqB, TqB, TdB, self.args.psr_k)
        T2T_psr_RO = CalcPSR(TqB, TqB, TdB, self.args.psr_k)
        print('T2I_psr_RO: {:.4f}'.format(T2I_psr_RO))
        print('I2I_psr_RO: {:.4f}'.format(I2I_psr_RO))
        print('I2T_psr_RO: {:.4f}'.format(I2T_psr_RO))
        print('T2T_psr_RO: {:.4f}'.format(T2T_psr_RO))
        print('test retrieval model done.')

    def retrieval_feedback_awareness(self, Tr_I, Tr_T, Tr_L):
        print('retrieval feedback awareness...')
        if ('I2T' in self.args.awareness) or ('I2I' in self.args.awareness):
            image_query_index = np.random.choice(range(Tr_I.size(0)), self.args.query_sample_number, replace = False)
        if ('T2I' in self.args.awareness) or ('T2T' in self.args.awareness):
            text_query_index = np.random.choice(range(Tr_T.size(0)), self.args.query_sample_number, replace = False)
        if 'I2T' in self.args.awareness:
            qIB = self.retrieval_model.generate_image_hashcode(Tr_I[image_query_index].type(torch.float).cuda()).cuda()
            dTB = self.retrieval_model.generate_text_hashcode(Tr_T).cuda()
            self.image_to_text_index = return_results(image_query_index, qIB, dTB, self.args.near_sample_number, self.args.rank_sample_number)
            self.image_to_text_index = self.image_to_text_index.long()
        if 'T2I' in self.args.awareness:
            qTB = self.retrieval_model.generate_text_hashcode(Tr_T[text_query_index].type(torch.float).cuda()).cuda()
            dIB = self.retrieval_model.generate_image_hashcode(Tr_I).cuda()
            self.text_to_image_index = return_results(text_query_index, qTB, dIB, self.args.near_sample_number, self.args.rank_sample_number)
            self.text_to_image_index = self.text_to_image_index.long()
        if 'I2I' in self.args.awareness:
            qIB = self.retrieval_model.generate_image_hashcode(Tr_I[image_query_index].type(torch.float).cuda()).cuda()
            dIB = self.retrieval_model.generate_image_hashcode(Tr_I).cuda()
            self.image_to_image_index = return_results(image_query_index, qIB, dIB, self.args.near_sample_number, self.args.rank_sample_number)
            self.image_to_image_index = self.image_to_image_index.long()
        if 'T2T' in self.args.awareness:
            qTB = self.retrieval_model.generate_text_hashcode(Tr_T[text_query_index].type(torch.float).cuda()).cuda()
            dTB = self.retrieval_model.generate_text_hashcode(Tr_T).cuda()
            self.text_to_text_index = return_results(text_query_index, qTB, dTB, self.args.near_sample_number, self.args.rank_sample_number)
            self.text_to_text_index = self.text_to_text_index.long()
        print('retrieval feedback awareness done.')

    def train_subnet(self, Tr_I, Tr_T, Tr_L):
        print('train subnet...')
        optimizer_text_subnet = torch.optim.Adam(self.text_subnet.parameters(), lr=self.args.subnet_text_learning_rate, betas=(0.5, 0.999))
        optimizer_image_subnet = torch.optim.Adam(filter(lambda p: p.requires_grad, self.image_subnet.parameters()), lr=self.args.subnet_image_learning_rate, betas=(0.5, 0.999))
        ranking_loss = torch.nn.MarginRankingLoss(margin=self.args.threshold_alpha)
        for epoch in range(self.args.subnet_epoch):
            index = np.random.permutation(self.args.query_sample_number)
            for i in range(self.args.query_sample_number // self.args.subnet_batch_size + 1):
                end_index = min((i+1)*self.args.subnet_batch_size, self.args.query_sample_number)
                num_index = end_index - i*self.args.subnet_batch_size
                ind = index[i*self.args.subnet_batch_size : end_index]
                optimizer_text_subnet.zero_grad()
                optimizer_image_subnet.zero_grad()
                loss_subnet = 0.
                for j in range(self.args.near_sample_number+1, self.args.near_sample_number+self.args.rank_sample_number):
                    for k in range(j+1, self.args.near_sample_number+self.args.rank_sample_number+1):
                        if 'I2T' in self.args.awareness:
                            anchor_IT = self.image_subnet(Tr_I[self.image_to_text_index[ind, 0]].type(torch.float).cuda())
                            rank1_IT = self.text_subnet(Tr_T[self.image_to_text_index[ind, j]].type(torch.float).cuda())
                            rank2_IT = self.text_subnet(Tr_T[self.image_to_text_index[ind, k]].type(torch.float).cuda())
                            ranking_target_IT = - (torch.ones(num_index) / (k-j)).type(torch.float).cuda() 
                            hamming_rank1_IT = calc_hamming(anchor_IT, rank1_IT) / self.args.subnet_bit
                            hamming_rank2_IT = calc_hamming(anchor_IT, rank2_IT) / self.args.subnet_bit
                            rank_loss_IT = ranking_loss(hamming_rank1_IT.cuda(), hamming_rank2_IT.cuda(), ranking_target_IT)
                            quant_loss_IT = (torch.sign(anchor_IT) - anchor_IT).pow(2).sum() / (num_index * self.args.subnet_bit)
                            balan_loss_IT = anchor_IT.sum(0).pow(2).sum() / (num_index * self.args.subnet_bit)
                            loss_subnet = loss_subnet + self.args.parameter_alpha * rank_loss_IT + self.args.parameter_beta * balan_loss_IT + self.args.parameter_gamma * quant_loss_IT
                        if 'T2I' in self.args.awareness:
                            anchor_TI = self.text_subnet(Tr_T[self.text_to_image_index[ind, 0]].type(torch.float).cuda())
                            rank1_TI = self.image_subnet(Tr_I[self.text_to_image_index[ind, j]].type(torch.float).cuda())
                            rank2_TI = self.image_subnet(Tr_I[self.text_to_image_index[ind, k]].type(torch.float).cuda())
                            ranking_target_TI = - (torch.ones(num_index) / (k-j)).type(torch.float).cuda()
                            hamming_rank1_TI = calc_hamming(anchor_TI, rank1_TI) / self.args.subnet_bit
                            hamming_rank2_TI = calc_hamming(anchor_TI, rank2_TI) / self.args.subnet_bit
                            rank_loss_TI = ranking_loss(hamming_rank1_TI.cuda(), hamming_rank2_TI.cuda(), ranking_target_TI)
                            balan_loss_TI = anchor_TI.sum(0).pow(2).sum() / (num_index * self.args.subnet_bit)
                            quant_loss_TI = (torch.sign(anchor_TI) - anchor_TI).pow(2).sum() / (num_index * self.args.subnet_bit)
                            loss_subnet = loss_subnet + self.args.parameter_alpha * rank_loss_TI + self.args.parameter_beta * balan_loss_TI + self.args.parameter_gamma * quant_loss_TI
                        if 'I2I' in self.args.awareness:
                            anchor_II = self.image_subnet(Tr_I[self.image_to_image_index[ind, 0]].type(torch.float).cuda())
                            rank1_II = self.image_subnet(Tr_I[self.image_to_image_index[ind, j]].type(torch.float).cuda())
                            rank2_II = self.image_subnet(Tr_I[self.image_to_image_index[ind, k]].type(torch.float).cuda())
                            ranking_target_II = - (torch.ones(num_index) / (k-j)).type(torch.float).cuda() 
                            hamming_rank1_II = calc_hamming(anchor_II, rank1_II) / self.args.subnet_bit
                            hamming_rank2_II = calc_hamming(anchor_II, rank2_II) / self.args.subnet_bit
                            rank_loss_II = ranking_loss(hamming_rank1_II.cuda(), hamming_rank2_II.cuda(), ranking_target_II)
                            quant_loss_II = (torch.sign(anchor_II) - anchor_II).pow(2).sum() / (num_index * self.args.subnet_bit)
                            balan_loss_II = anchor_II.sum(0).pow(2).sum() / (num_index * self.args.subnet_bit)
                            loss_subnet = loss_subnet + self.args.parameter_alpha * rank_loss_II + self.args.parameter_beta * balan_loss_II + self.args.parameter_gamma * quant_loss_II
                        if 'T2T' in self.args.awareness:
                            anchor_TT = self.text_subnet(Tr_T[self.text_to_text_index[ind, 0]].type(torch.float).cuda())
                            rank1_TT = self.text_subnet(Tr_T[self.text_to_text_index[ind, j]].type(torch.float).cuda())
                            rank2_TT = self.text_subnet(Tr_T[self.text_to_text_index[ind, k]].type(torch.float).cuda())
                            ranking_target_TT = - (torch.ones(num_index) / (k-j)).type(torch.float).cuda() 
                            hamming_rank1_TT = calc_hamming(anchor_TT, rank1_TT) / self.args.subnet_bit
                            hamming_rank2_TT = calc_hamming(anchor_TT, rank2_TT) / self.args.subnet_bit
                            rank_loss_TT = ranking_loss(hamming_rank1_TT.cuda(), hamming_rank2_TT.cuda(), ranking_target_TT)
                            quant_loss_TT = (torch.sign(anchor_TT) - anchor_TT).pow(2).sum() / (num_index * self.args.subnet_bit)
                            balan_loss_TT = anchor_TT.sum(0).pow(2).sum() / (num_index * self.args.subnet_bit)
                            loss_subnet = loss_subnet + self.args.parameter_alpha * rank_loss_TT + self.args.parameter_beta * balan_loss_TT + self.args.parameter_gamma * quant_loss_TT
                loss_subnet.backward()
                optimizer_text_subnet.step()
                optimizer_image_subnet.step()
            print('epoch:{:2d}  loss_subnet:{:.4f}'
                .format(epoch, loss_subnet))
        self.save_subnet()
        print('train subnet done.')
    
    def test_subnet(self, Db_I, Db_T, Db_L, Te_I, Te_T, Te_L):
        print('test subnet...')
        self.load_subnet()
        IqB = self.image_subnet.generate_hash_code(Te_I)
        TqB = self.text_subnet.generate_hash_code(Te_T)
        IdB = self.image_subnet.generate_hash_code(Db_I)
        TdB = self.text_subnet.generate_hash_code(Db_T)
        I2T_map = CalcMap(IqB, TdB, Te_L, Db_L, self.args.map_k)
        T2I_map = CalcMap(TqB, IdB, Te_L, Db_L, self.args.map_k)
        I2I_map = CalcMap(IqB, IdB, Te_L, Db_L, self.args.map_k)
        T2T_map = CalcMap(TqB, TdB, Te_L, Db_L, self.args.map_k)
        print('I2T_map: {:.4f}'.format(I2T_map))
        print('T2I_map: {:.4f}'.format(T2I_map))
        print('I2I_map: {:.4f}'.format(I2I_map))
        print('T2T_map: {:.4f}'.format(T2T_map))
        print('test subnet done.')

    def train_diffusion_model(self, Tr_I, Tr_T, Tr_L):
        print('train diffusion model...')
        self.load_subnet()
        optimizer_diffusion_model = torch.optim.Adam(self.diffusion_model.parameters(), lr=self.args.diffusion_learning_rate, betas=(0.5, 0.999))
        optimizer_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=self.args.diffusion_learning_rate, betas=(0.5, 0.999))
        mse_loss = torch.nn.MSELoss()
        ranking_loss_anti_neighbor = torch.nn.MarginRankingLoss(margin=self.args.threshold_beta)
        ranking_loss_generalization = torch.nn.MarginRankingLoss(margin=self.args.threshold_gamma)
        for epoch in range(self.args.diffusion_epoch):
            index = np.random.permutation(self.args.query_sample_number)
            for i in range(self.args.query_sample_number // self.args.diffusion_batch_size + 1):
                end_index = min((i+1)*self.args.diffusion_batch_size, self.args.query_sample_number)
                num_index = end_index - i*self.args.diffusion_batch_size
                ind = index[i*self.args.diffusion_batch_size : end_index]
                batch_original_image = image_normalization(Tr_I[self.image_to_text_index[ind, 0]].type(torch.float).cuda())
                batch_noise_image = 2 * torch.randn_like(batch_original_image).cuda() - 1
                batch_protected_image = self.diffusion_model(batch_original_image, batch_noise_image, 1)
                batch_protected_image_pixel = image_restoration(batch_protected_image)
                batch_noise_image_pixel = image_restoration(batch_noise_image)
                batch_protected_image_feature = self.image_subnet(batch_protected_image_pixel)
                batch_noise_image_feature = self.image_subnet(batch_noise_image_pixel)
                batch_protected_image_class = self.pretrain_model(batch_protected_image_pixel)
                # update discriminator
                if i % 3 ==0:
                    self.set_requires_grad(self.discriminator, True)
                    optimizer_discriminator.zero_grad()
                    batch_original_image_discriminator = self.discriminator(batch_original_image)
                    batch_protected_image_discriminator = self.discriminator(batch_protected_image.detach())
                    real_D_loss = self.criterionGAN(batch_original_image_discriminator, True)
                    adv_D_loss = self.criterionGAN(batch_protected_image_discriminator, False)
                    D_loss = (real_D_loss + adv_D_loss) / 2
                    D_loss.backward()
                    optimizer_discriminator.step()
                # update diffusion model
                self.set_requires_grad(self.discriminator, False)
                optimizer_diffusion_model.zero_grad()
                outlier_induction_loss = mse_loss(batch_protected_image_feature, batch_noise_image_feature)
                anti_neighbor_loss = .0
                for j in range(self.args.near_sample_number):
                    batch_text_code = self.text_subnet(Tr_T[self.image_to_text_index[ind, 1+j].long()].type(torch.float).cuda())
                    batch_image_code = self.image_subnet(Tr_I[self.image_to_image_index[ind, 1+j].long()].type(torch.float).cuda())
                    hamming_dist_IT = calc_hamming(batch_protected_image_feature, batch_text_code) / self.args.subnet_bit
                    hamming_dist_II = calc_hamming(batch_protected_image_feature, batch_image_code) / self.args.subnet_bit
                    anti_neighbor_loss = anti_neighbor_loss + ranking_loss_anti_neighbor(hamming_dist_IT.cuda(), torch.zeros(num_index).cuda(), torch.ones(num_index).cuda()) \
                        + ranking_loss_anti_neighbor(hamming_dist_II.cuda(), torch.zeros(num_index).cuda(), torch.ones(num_index).cuda())
                batch_protected_image_discriminator = self.discriminator(batch_protected_image)
                adversarial_loss = self.criterionGAN(batch_protected_image_discriminator, True)
                generalization_loss = ranking_loss_generalization(torch.zeros(num_index, 1000).cuda(), batch_protected_image_class.cuda(), torch.ones(num_index, 1000).cuda())
                reconstruction_loss = mse_loss(batch_protected_image, batch_original_image)
                diffusion_model_loss = self.args.parameter_lambda * outlier_induction_loss + self.args.parameter_epsilon * anti_neighbor_loss \
                    + self.args.parameter_mu * adversarial_loss + self.args.parameter_nu * generalization_loss + self.args.parameter_xi * reconstruction_loss 
                diffusion_model_loss.backward()
                optimizer_diffusion_model.step()
            print('epoch:{:2d}   D_loss:{:.4f}  G_loss:{:.4f}  OIL:{:.4f}  RL:{:.4f}  AL:{:.4f}  GL:{:.4f}  ANL:{:.4f}'
                .format(epoch, D_loss, diffusion_model_loss, outlier_induction_loss, reconstruction_loss, adversarial_loss, generalization_loss, anti_neighbor_loss))
        self.save_diffusion_model()
        print('train diffusion model done.')

    def test_diffusion_model(self, Db_I, Db_T, Db_L, Te_I, Te_T, Te_L):
        print('test diffusion model...')
        self.load_subnet()
        self.load_diffusion_model()
        IqB_SP = torch.zeros([self.Dcfg.query_size, self.args.subnet_bit]).cuda()
        IqB_RP = torch.zeros([self.Dcfg.query_size, self.args.retrieval_bit]).cuda()
        perceptibility = 0.0
        for i in range(self.Dcfg.query_size):
            original_image = image_normalization(Te_I[i].float().cuda())
            noise_image = 2 * torch.randn_like(original_image).cuda() - 1
            protected_image = self.diffusion_model(original_image.unsqueeze(0), noise_image.unsqueeze(0), 1)
            protected_image_pixel = image_restoration(protected_image)
            protected_subnet_image_code = self.image_subnet.generate_hash_code(protected_image_pixel)
            protected_retrieval_image_code = self.retrieval_model.generate_image_hashcode(protected_image_pixel)
            IqB_SP[i, :] = torch.sign(protected_subnet_image_code.cpu().data)
            IqB_RP[i, :] = torch.sign(protected_retrieval_image_code.cpu().data)
            perceptibility += F.mse_loss((original_image+1)/2, (protected_image[0]+1)/2).data
        IqB_SO = self.image_subnet.generate_hash_code(Te_I)
        TqB_SO = self.text_subnet.generate_hash_code(Te_T)
        IqB_RO = self.retrieval_model.generate_image_hashcode(Te_I).cuda()
        TqB_RO = self.retrieval_model.generate_text_hashcode(Te_T).cuda()
        IdB_SO = self.image_subnet.generate_hash_code(Db_I)
        TdB_SO = self.text_subnet.generate_hash_code(Db_T)
        IdB_RO = self.retrieval_model.generate_image_hashcode(Db_I).cuda()
        TdB_RO = self.retrieval_model.generate_text_hashcode(Db_T).cuda()
        I2I_map_SO = CalcMap(IqB_SO, IdB_SO, Te_L.cuda(), Db_L.cuda(), self.args.map_k)
        I2T_map_SO = CalcMap(IqB_SO, TdB_SO, Te_L.cuda(), Db_L.cuda(), self.args.map_k)
        I2I_map_SP = CalcMap(IqB_SP, IdB_SO, Te_L.cuda(), Db_L.cuda(), self.args.map_k)
        I2T_map_SP = CalcMap(IqB_SP, TdB_SO, Te_L.cuda(), Db_L.cuda(), self.args.map_k)
        I2I_map_RO = CalcMap(IqB_RO, IdB_RO, Te_L.cuda(), Db_L.cuda(), self.args.map_k)
        I2T_map_RO = CalcMap(IqB_RO, TdB_RO, Te_L.cuda(), Db_L.cuda(), self.args.map_k)
        I2I_map_RP = CalcMap(IqB_RP, IdB_RO, Te_L.cuda(), Db_L.cuda(), self.args.map_k)
        I2T_map_RP = CalcMap(IqB_RP, TdB_RO, Te_L.cuda(), Db_L.cuda(), self.args.map_k)
        I2I_psr_SO = CalcPSR(IqB_SO, IqB_SO, IdB_SO, self.args.psr_k)
        T2I_psr_SO = CalcPSR(TqB_SO, IqB_SO, IdB_SO, self.args.psr_k)
        I2I_psr_SP = CalcPSR(IqB_SO, IqB_SP, IdB_SO, self.args.psr_k)
        T2I_psr_SP = CalcPSR(TqB_SO, IqB_SP, IdB_SO, self.args.psr_k)
        I2I_psr_RO = CalcPSR(IqB_RO, IqB_RO, IdB_RO, self.args.psr_k)
        T2I_psr_RO = CalcPSR(TqB_RO, IqB_RO, IdB_RO, self.args.psr_k)
        I2I_psr_RP = CalcPSR(IqB_RO, IqB_RP, IdB_RO, self.args.psr_k)
        T2I_psr_RP = CalcPSR(TqB_RO, IqB_RP, IdB_RO, self.args.psr_k)
        print('perceptibility: {:.6f}'.format(torch.sqrt(perceptibility/self.Dcfg.query_size)))
        print('I2T_map_SO: {:.4f}'.format(I2T_map_SO))
        print('I2T_map_SP: {:.4f}'.format(I2T_map_SP))
        print('I2I_map_SO: {:.4f}'.format(I2I_map_SO))
        print('I2I_map_SP: {:.4f}'.format(I2I_map_SP))
        print('I2T_map_RO: {:.4f}'.format(I2T_map_RO))
        print('I2T_map_RP: {:.4f}'.format(I2T_map_RP))
        print('I2I_map_RO: {:.4f}'.format(I2I_map_RO))
        print('I2I_map_RP: {:.4f}'.format(I2I_map_RP))
        print('T2I_psr_SO: {:.4f}'.format(T2I_psr_SO))
        print('T2I_psr_SP: {:.4f}'.format(T2I_psr_SP))
        print('I2I_psr_SO: {:.4f}'.format(I2I_psr_SO))
        print('I2I_psr_SP: {:.4f}'.format(I2I_psr_SP))
        print('T2I_psr_RO: {:.4f}'.format(T2I_psr_RO))
        print('T2I_psr_RP: {:.4f}'.format(T2I_psr_RP))
        print('I2I_psr_RO: {:.4f}'.format(I2I_psr_RO))
        print('I2I_psr_RP: {:.4f}'.format(I2I_psr_RP))
        print("test diffusion model done.")

        