from datasetClasses import MusdbDataset, LibriSpeechDataset, AccompanimentVocalData, SpeechData, AccompanimentData
from TsaiMiniRocketDiscriminator import TsaiMiniRocketDiscriminator

import model.utils as model_utils
import utils
from model.waveunet import Waveunet

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

default_train_parameters = {
    # data loaders
    "num_workers":    1,

    # optimisation
    "lr_G":          1e-4,
    "lr_G2":         1e-4,
    "lr_D":          1e-4,
    "betas":         (0.9, 0.999),

    # loss weights
    "lambda_l1":     1,
    "lambda_cycle": .01,
    "lambda_identity": .01,


    # bookkeeping
    #"log_dir":       f"runs/cycleGAN_experiment_{now}",
    #"model_dir":     "models",
}


class VocalCycleGAN:

    def __init__(self, batch_size = 32, virtual_batch_size = 1, num_epochs = 10, smart_discriminator = False):
        cuda = torch.cuda.is_available()
        if cuda:
            self.device = torch.device('cuda')
            print("GPU:", torch.cuda.get_device_name(0))
        else:
            raise Exception("GPU not available. Please check your setup.")

        self.generator_vocal = None
        self.generator_speech = None
        self.discriminator_vocal = None
        self.discriminator_speech = None
        self.optimizer_DV = None
        self.optimizer_DS = None
        self.optimizer_GV = None
        self.optimizer_GS = None
        self.acc_voc_loader = None
        self.speech_loader = None
        self.acc_loader = None
        self.bce_loss = nn.BCELoss().to(self.device)
        self.l1_loss = nn.L1Loss().to(self.device)
        self.mse_loss = nn.MSELoss().to(self.device)
        self.lambda_l1 = default_train_parameters["lambda_l1"]
        self.lambda_cycle = default_train_parameters["lambda_cycle"]
        self.lambda_identity = default_train_parameters["lambda_identity"]
        self.virtual_batch_size = virtual_batch_size
        self.clip_length = 0
        self.input_size_generators = 0
        self.num_workers = default_train_parameters["num_workers"]
        self.smart_discriminator = smart_discriminator
        self.batch_size = batch_size
        self.num_epochs = num_epochs

	
    def fit(self, musdb_dataset, librispeech_dataset):

        musdb_length = musdb_dataset.mel_specs.shape[-1]
        librispeech_length = librispeech_dataset.mel_specs.shape[-1]

        if musdb_length == librispeech_length:
            self.clip_length = musdb_length
        else:
            raise ValueError("The lengths of the datasets do not match. Please check the dataset lengths.")

        model_config_gen_vocal= {
            "num_inputs": 256,  # Two spectrograms concatenated (2 * 128 mel bins)
            "num_outputs": 128,
            "num_channels": [512*2, 512*4, 512*8],
            "instruments": ["vocal"],
            "kernel_size": 3,
            "target_output_size": self.clip_length,
            "conv_type": "normal",
            "res": "fixed",
            "separate": False,
            "depth": 1,
            "strides": 2
        }

        model_config_gen_speech = {
            "num_inputs": 128,  # One spectrogram input
            "num_outputs": 128,
            "num_channels": [256*2, 256*4, 256*8],
            "instruments": ["speech"],
            "kernel_size": 3,
            "target_output_size": self.clip_length,
            "conv_type": "normal",
            "res": "fixed",
            "separate": False,
            "depth": 1,
            "strides": 2
        }

        # create models
        self.generator_vocal = Waveunet(**model_config_gen_vocal).to(self.device)
        self.generator_speech = Waveunet(**model_config_gen_speech).to(self.device)
        self.discriminator_vocal = TsaiMiniRocketDiscriminator().to(self.device)
        self.discriminator_speech = TsaiMiniRocketDiscriminator(freq_bins = 128,
                                                    hidden_dim = 512,
                                                    accompaniment = False).to(self.device)
        
        self.input_size_generators = self.generator_vocal.input_size
        
        # create dataloaders
        self.acc_voc_loader = DataLoader(
                AccompanimentVocalData(musdb_dataset, output_length = self.input_size_generators),
                batch_size = self.batch_size,
                shuffle = True,
                drop_last = True,
                num_workers = self.num_workers,
                pin_memory = True,
                persistent_workers = True
            )
        self.speech_loader = DataLoader(
                SpeechData(librispeech_dataset, output_length = self.input_size_generators),
                batch_size = self.batch_size,
                shuffle = True,
                drop_last = True,
                num_workers = self.num_workers,
                pin_memory = True,
                persistent_workers = True
            )
        self.acc_loader = DataLoader(
                AccompanimentData(musdb_dataset, output_length = self.input_size_generators),
                batch_size = self.batch_size,
                shuffle = True,
                drop_last = True,
                num_workers = self.num_workers,
                pin_memory = True,
                persistent_workers = True
        )

        speech_batch = next(iter(self.speech_loader))["no_pad"]

        # initialize rocket
        self.discriminator_vocal.fit_rocket(speech_batch)
        self.discriminator_speech.fit_rocket(speech_batch)

        # create optimizers
        self.optimizer_GV  = optim.Adam(self.generator_vocal.parameters(),  
                                        lr = default_train_parameters["lr_G"],  
                                        betas = default_train_parameters["betas"])
        self.optimizer_GS = optim.Adam(self.generator_speech.parameters(), 
                                       lr = default_train_parameters["lr_G2"], 
                                       betas = default_train_parameters["betas"])
        self.optimizer_DV  = optim.Adam(self.discriminator_vocal.parameters(), 
                                        lr = default_train_parameters["lr_D"], 
                                        betas = default_train_parameters["betas"])
        self.optimizer_DS  = optim.Adam(self.discriminator_speech.parameters(), 
                                        lr = default_train_parameters["lr_D"], 
                                        betas = default_train_parameters["betas"])


    def train_epoch_random_accomp(self):
        total_loss_DV = total_loss_DS = total_loss_GV = total_loss_GS = 0
        total_loss_adv_vocal = total_loss_adv_speech = total_loss_cycle_vocal = total_loss_cycle_speech = 0.0
        # Optionally record gradient norms per batch for diagnosing vanishing gradients.
        grad_norms_DV = []
        grad_norms_DS = []
        grad_norms_GV = []
        grad_norms_GS = []
        num_batches = 0


        alpha = 0.9      # Tuneable constant to gate the discriminator training
        running_loss_DV = running_loss_DS = 0.0
        dv_threshold = 0.6
        ds_threshold = 0.6

        self.optimizer_DV.zero_grad()
        self.optimizer_DS.zero_grad()
        self.optimizer_GV.zero_grad()
        self.optimizer_GS.zero_grad()

        num_DV_backwards = num_DS_backwards = 0

        loaders = tqdm(zip(self.acc_voc_loader, self.speech_loader, self.acc_loader), desc = "Training Batches")

        # ---- batch loop ----
        for (acc_voc, speech, accomp) in loaders:
            # Read in data
            x_acc = acc_voc["acc_pad"].float().to(self.device)       # [B,128,289]
            x_voc = acc_voc["voc_pad"].float().to(self.device)
            x_speech = speech["pad"].float().to(self.device)    # [B,128,289]
            x_accomp = accomp["pad"].float().to(self.device)
            x_in = torch.cat([x_speech, x_accomp], dim=1)     # [B,256,289]

            real_labels = torch.ones(self.batch_size, 1, device=self.device, requires_grad = False)
            fake_labels = torch.zeros(self.batch_size, 1, device=self.device, requires_grad = False)

            acc_np    = acc_voc["acc_no_pad"].float().to(self.device)
            voc_np    = acc_voc["voc_no_pad"].float().to(self.device)
            speech_np = speech["no_pad"].float().to(self.device)

            # Compute transformations with generators
            raw_fake_vocal = self.generator_vocal(x_in)["vocal"]
            fake_vocal = raw_fake_vocal.clone()
            fake_vocal_crop = torch.narrow(fake_vocal, 2, 0, self.clip_length).clone()

            raw_fake_speech = self.generator_speech(x_voc)["speech"]
            fake_speech = raw_fake_speech.clone()
            fake_speech_crop = torch.narrow(fake_speech, 2, 0, self.clip_length).clone()

            # Generate reconstructed speech/vocal
            fake_vocal_pad = self._pad_spectrogram(fake_vocal)  # you must define this
            raw_rec_speech = self.generator_speech(fake_vocal_pad)["speech"]
            rec_speech = raw_rec_speech.clone()
            rec_speech_crop = torch.narrow(rec_speech, 2, 0, self.clip_length).clone()

            fake_speech_pad = self._pad_spectrogram(fake_speech)  # you must define this
            fake_speech_with_acc = torch.cat([fake_speech_pad, x_acc], dim=1)
            raw_rec_vocal = self.generator_vocal(fake_speech_with_acc)["vocal"]
            rec_vocal = raw_rec_vocal.clone()
            rec_vocal_crop = torch.narrow(rec_vocal, 2, 0, self.clip_length).clone()

            # Identity generation
            identity_vocal = self.generator_vocal(torch.cat([x_voc, x_acc], dim=1))["vocal"]
            identity_vocal_crop = torch.narrow(identity_vocal, 2, 0, self.clip_length).clone()

            identity_speech = self.generator_speech(x_speech)["speech"]
            identity_speech_crop = torch.narrow(identity_speech, 2, 0, self.clip_length).clone()

            # Compute losses
            pred_real_vocal    = self.discriminator_vocal(voc_np, acc_np)
            pred_fake_vocal_D  = self.discriminator_vocal(fake_vocal_crop.detach(), acc_np)
            pred_real_speech   = self.discriminator_speech(speech_np)
            pred_fake_speech_D = self.discriminator_speech(fake_speech_crop.detach())

            loss_DV_fake = self.bce_loss(pred_fake_vocal_D, fake_labels)
            loss_DV_real = self.bce_loss(pred_real_vocal, real_labels)
            loss_DS_fake = self.bce_loss(pred_fake_speech_D, fake_labels)
            loss_DS_real = self.bce_loss(pred_real_speech, real_labels)

            # Minimizing adv losses is teaching the gens to trick the discs (labels are swapped)
            pred_fake_vocal  = self.discriminator_vocal(fake_vocal_crop, acc_np)
            pred_fake_speech = self.discriminator_speech(fake_speech_crop)
            loss_adv_vocal   = self.mse_loss(pred_fake_vocal, real_labels)
            loss_adv_speech  = self.mse_loss(pred_fake_speech, real_labels)

            loss_cycle_vocal  = self.l1_loss(rec_vocal_crop, voc_np)
            loss_cycle_speech = self.l1_loss(rec_speech_crop, speech_np)

            loss_identity_vocal  = self.l1_loss(identity_vocal_crop, voc_np)
            loss_identity_speech = self.l1_loss(identity_speech_crop, speech_np)

            # Combine losses into one loss per neural net
            loss_DV = 0.5 * (loss_DV_real + loss_DV_fake)
            loss_DS = 0.5 * (loss_DS_real + loss_DS_fake)
            loss_GV = self._convex_comb(loss_adv_vocal, loss_cycle_vocal, loss_identity_vocal)
            loss_GS = self._convex_comb(loss_adv_speech, loss_cycle_speech, loss_identity_speech)

            running_loss_DV = alpha * running_loss_DV + (1 - alpha) * loss_DV.item()
            running_loss_DS = alpha * running_loss_DS + (1 - alpha) * loss_DS.item()

            # Update generators
            ((loss_GV + loss_GS) / self.virtual_batch_size).backward()

            # Record gradients & take steps
            if (num_batches + 1) % self.virtual_batch_size == 0:
                # Record gradients
                grad_norm = 0.0
                count = 0
                for p in self.generator_vocal.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.norm().item()
                        count += 1
                if count > 0:
                    grad_norms_GV.append(grad_norm / count)

                grad_norm = 0.0
                count = 0
                for p in self.generator_speech.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.norm().item()
                        count += 1
                if count > 0:
                    grad_norms_GS.append(grad_norm / count)

                # Take steps
                self.optimizer_GV.step()
                self.optimizer_GS.step()
                self.optimizer_GV.zero_grad()
                self.optimizer_GS.zero_grad()

            # Update discriminators
            if running_loss_DV > dv_threshold or self.smart_discriminator:
                (loss_DV / self.virtual_batch_size).backward()
                num_DV_backwards += 1
                if (num_DV_backwards + 1) % self.virtual_batch_size == 0:
                    grad_norm = 0.0
                    count = 0
                    for p in self.discriminator_vocal.parameters():
                        if p.grad is not None:
                            grad_norm += p.grad.norm().item()
                            count += 1
                    if count > 0:
                        grad_norms_DV.append(grad_norm / count)

                    self.optimizer_DV.step()
                    self.optimizer_DV.zero_grad()
                    num_DV_backwards = 0

            if running_loss_DS > ds_threshold or self.smart_discriminator:
                (loss_DS / self.virtual_batch_size).backward()
                num_DS_backwards += 1
                if (num_DS_backwards+1) % self.virtual_batch_size == 0:
                    # record gradients
                    grad_norm = 0.0
                    count = 0
                    for p in self.discriminator_speech.parameters():
                        if p.grad is not None:
                            grad_norm += p.grad.norm().item()
                            count += 1
                    if count > 0:
                        grad_norms_DS.append(grad_norm / count)

                    self.optimizer_DS.step()
                    self.optimizer_DS.zero_grad()
                    num_DS_backwards = 0

            # Accumulate metrics
            total_loss_DV     += loss_DV.item()
            total_loss_DS     += loss_DS.item()
            total_loss_adv_vocal  += loss_adv_vocal.item()
            total_loss_adv_speech += loss_adv_speech.item()
            total_loss_cycle_vocal += loss_cycle_vocal.item()
            total_loss_cycle_speech += loss_cycle_speech.item()
            total_loss_GV      += loss_GV.item()
            total_loss_GS   += loss_GS.item()
            num_batches       += 1
        
        return {
            "loss_DV":      total_loss_DV / num_batches,
            "loss_DS":      total_loss_DS / num_batches,
            "loss_GV":      total_loss_GV / num_batches,
            "loss_GS":      total_loss_GS / num_batches,
            "loss_adv_vocal":  total_loss_adv_vocal / num_batches,
            "loss_adv_speech":  total_loss_adv_speech / num_batches,
            "loss_cycle_vocal":  total_loss_cycle_vocal / num_batches,
            "loss_cycle_speech":  total_loss_cycle_speech / num_batches,
            "avg_grad_norm_DV": sum(grad_norms_DV) / len(grad_norms_DV) if grad_norms_DV else 0.0,
            "avg_grad_norm_DS": sum(grad_norms_DS) / len(grad_norms_DS) if grad_norms_DS else 0.0,
            "avg_grad_norm_GV": sum(grad_norms_GV) / len(grad_norms_GV) if grad_norms_GV else 0.0,
            "avg_grad_norm_GS": sum(grad_norms_GS) / len(grad_norms_GS) if grad_norms_GS else 0.0,
            "num_DV_updates" : len(grad_norms_DV),
            "num_DS_updates" : len(grad_norms_DS)
        }

    def train(self):
        ############ MISSING SUMMARY WRITER CODE #################
        for epoch in range(self.num_epochs):
            print(f"\n=== Epoch {epoch+1}/{self.num_epochs} ===")
            epoch_metrics = self.train_epoch_random_accomp()

            print(f"Epoch {epoch+1} Metrics:")
            print(f"  Loss_DV:         {epoch_metrics['loss_DV']:.4f}")
            print(f"  Loss_DS:         {epoch_metrics['loss_DS']:.4f}")
            # print(f"  Loss_GV_total:   {epoch_metrics['loss_GV']:.4f}")
            # print(f"  Loss_GS_total:   {epoch_metrics['loss_GS']:.4f}")
            print(f"  Loss_adv_vocal:     {epoch_metrics['loss_adv_vocal']:.4f}")
            print(f"  Loss_adv_speech:     {epoch_metrics['loss_adv_speech']:.4f}")
            print(f"  Loss_Cycle Vocal:     {epoch_metrics['loss_cycle_vocal']:.4f}")
            print(f"  Loss_Cycle Speech:     {epoch_metrics['loss_cycle_speech']:.4f}")
            print(f"  Loss_Identity Vocal:     {epoch_metrics['loss_identity_vocal']:.4f}")
            print(f"  Loss_Identity Speech:     {epoch_metrics['loss_identity_speech']:.4f}")
            print(f"  Grad Norm DV:    {epoch_metrics['avg_grad_norm_DV']:.4f}")
            print(f"  Grad Norm DS:    {epoch_metrics['avg_grad_norm_DS']:.4f}")
            print(f"  Grad Norm GV:    {epoch_metrics['avg_grad_norm_GV']:.4f}")
            print(f"  Grad Norm GS:    {epoch_metrics['avg_grad_norm_GS']:.4f}")
            print(f"  num_DV_updates:    {epoch_metrics['num_DV_updates']:.4f}")
            print(f"  num_DS_updates:    {epoch_metrics['num_DS_updates']:.4f}")

        ################ NEED TO LOG METRICS TO TENSORBOARD #####################


    def _pad_spectrogram(self, batch):
        current_len = batch.size(-1)
        delta = self.input_size_generators - current_len

        if delta > 0:
            left_pad_len = (delta // 2) + (delta % 2)
            right_pad_len = delta // 2
            batch_pad = F.pad(batch, (left_pad_len, right_pad_len), "constant", -80)

        return batch_pad

    def _convex_comb(self, adv, cycle, identity):
        den = (1 + self.lambda_cycle + self.lambda_identity)
        return (adv + self.lambda_cycle * cycle + self.lambda_identity * identity) / den#from datasetClasses import MusdbDataset, LibriSpeechDataset, AccompanimentVocalData, SpeechData
