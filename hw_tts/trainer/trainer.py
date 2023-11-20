import random
from pathlib import Path
from random import shuffle

import PIL
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import ToTensor
from tqdm import tqdm

from hw_tts.base import BaseTrainer
from hw_tts.base.base_text_encoder import BaseTextEncoder
from hw_tts.logger.utils import plot_spectrogram_to_buf
from hw_tts.utils import inf_loop, MetricTracker
import hw_tts.waveglow as waveglow
import hw_tts.text as text_lib
import hw_tts.audio as audio_lib
from hw_tts.utils.util import get_WaveGlow
import time


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
            self,
            model,
            criterion,
            metrics,
            optimizer,
            config,
            device,
            dataloaders,
            lr_scheduler=None,
            len_epoch=None,
            skip_oom=True,
    ):
        super().__init__(model, criterion, metrics, optimizer, config, device, lr_scheduler)
        self.skip_oom = skip_oom
        self.config = config
        self.train_dataloader = dataloaders["train"]
        self.WaveGlow = get_WaveGlow()
        self.WaveGlow = self.WaveGlow.to(self.device)
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items() if k != "train"}
        self.log_step = 50

        self.train_metrics = MetricTracker(
            "loss", "grad norm", *[m.name for m in self.metrics if 'LM' not in m.name], writer=self.writer
        )
        self.evaluation_metrics = MetricTracker(
            "loss", *[m.name for m in self.metrics], writer=self.writer
        )

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the HPU
        """
        for tensor_for_gpu in ["text",
                               "mel_target",
                               "duration",
                               "pitch",
                               "energy",
                               "mel_pos",
                               "src_pos"]:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["grad_norm_clip"]
            )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)

        bar = tqdm(range(self.len_epoch), desc='train')
        i = 0
        for batch_idx_cut, batch_cut in enumerate(
                self.train_dataloader
        ):
            stop = False
            for batch_idx, batch in enumerate(batch_cut):
                bar.update(1)
                i += 1
                try:
                    batch = self.process_batch(
                        batch,
                        is_train=True,
                        metrics=self.train_metrics,
                    )
                except RuntimeError as e:
                    if "out of memory" in str(e) and self.skip_oom:
                        self.logger.warning("OOM on batch. Skipping batch.")
                        for p in self.model.parameters():
                            if p.grad is not None:
                                del p.grad  # free some memory
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
                self.train_metrics.update("grad norm", self.get_grad_norm())
                if (batch_idx + batch_idx_cut * self.config["data"]["train"]["batch_expand_size"]) % self.log_step == 0:
                    self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx + batch_idx_cut * self.config["data"]["train"]["batch_expand_size"])
                    self.logger.debug(
                        "Train Epoch: {} {} Loss: {:.6f}".format(
                            epoch, self._progress(batch_idx), batch["loss"].item()
                        )
                    )
                    self.writer.add_scalar(
                        "learning rate", self.lr_scheduler.get_last_lr()[0]
                    )
                    self._log_predictions(**batch, train=True, examples_to_log=1)
                    self._log_scalars(self.train_metrics)
                    # we don't want to reset train metrics at the start of every epoch
                    # because we are interested in recent train metrics
                    last_train_metrics = self.train_metrics.result()
                    self.train_metrics.reset()
                if i >= self.len_epoch:
                    stop = True
                    break
            if stop:
                break
        log = last_train_metrics

        # for part, dataloader in self.evaluation_dataloaders.items():
        #     val_log = self._evaluation_epoch(epoch, part, dataloader)
        #     log.update(**{f"{part}_{name}": value for name, value in val_log.items()})
        
        # self.lr_scheduler.step(log["val_loss"])

        return log

    def process_batch(self, batch, is_train: bool, metrics: MetricTracker):
        start = time.time()
        batch = self.move_batch_to_device(batch, self.device)
        # print("batch to device time", time.time() - start)
        if is_train:
            self.optimizer.zero_grad()
        start = time.time()
        outputs = self.model(**batch)
        # print("inference time", time.time() - start)
        batch.update(outputs)
        start = time.time()
        l1, l2, l3, l4 = self.criterion(**batch, train=is_train)
        # print("loss time", time.time() - start)
        batch["loss"] = l1 + l2 + l3 + l4
        if is_train:
            start = time.time()
            batch["loss"].backward()
            # print("backward time", time.time() - start)
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        metrics.update("loss", batch["loss"].item())
        for met in self.metrics:
            metrics.update(met.name, met(**batch), n=len(batch["text"]))
        return batch

    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.evaluation_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                    enumerate(dataloader),
                    desc=part,
                    total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch,
                    is_train=False,
                    metrics=self.evaluation_metrics,
                )
            self.writer.set_step(epoch * self.len_epoch, part)
            self._log_scalars(self.evaluation_metrics)
            self._log_predictions(**batch, train=False)

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins="auto")
        return self.evaluation_metrics.result()

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _log_predictions(
            self,
            mel_predicted,
            audio_name,
            train=False,
            examples_to_log=10,
            *args,
            **kwargs,
    ):
        # TODO: implement logging of beam search results
        if self.writer is None:
            return
        zipped = list(zip(mel_predicted, audio_name))
        shuffle(zipped)
        rows = {}
        i = 0
        for mel, name in zipped[:examples_to_log]:
            i += 1
            mel = mel.contiguous().transpose(-1, -2).unsqueeze(0)
            a = waveglow.inference.get_wav(
                mel, self.WaveGlow,
            )
            rows[name] = {
                "audio": self.writer.wandb.Audio(a, sample_rate=22050),
                "name": name
            }
        self.writer.add_table("predictions", pd.DataFrame.from_dict(rows, orient="index"))

    def _log_spectrogram(self, spectrogram_batch):
        spectrogram = random.choice(spectrogram_batch.cpu())
        image = PIL.Image.open(plot_spectrogram_to_buf(spectrogram))
        self.writer.add_image("spectrogram", ToTensor()(image))
    
    def _log_audio(self, audio_batch):
        # spectrogram = random.choice(spectrogram_batch.cpu())
        # image = PIL.Image.open(plot_spectrogram_to_buf(spectrogram))
        audio = random.choice(audio_batch.cpu())
        self.writer.add_audio("audio", audio, sample_rate=16000)

    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))
