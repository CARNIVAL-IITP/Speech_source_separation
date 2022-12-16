import json
import logging
from pathlib import Path
import torch
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import torch.nn.functional as F
from . import distrib
from .utils import bold, copy_state, pull_metric, serialize_model, swap_state, LogProgress
from .models.version_1 import center_trim,normalize_input,unnormalize_input
logger = logging.getLogger(__name__)

class Solver(object):
    def __init__(self, data, model,speaker_model, optimizer, args):
        self.tr_loader = data['tr_loader']
        self.cv_loader = data['cv_loader']
        self.model = model
        self.dmodel = distrib.wrap(model)
        if speaker_model is not None:
            self.speaker_model = speaker_model
            self.dspeaker_model = distrib.wrap(speaker_model)
        self.optimizer = optimizer
        if args.lr_sched == 'step':
            self.sched = StepLR(
                self.optimizer, step_size=args.step.step_size, gamma=args.step.gamma)
        elif args.lr_sched == 'plateau':
            self.sched = ReduceLROnPlateau(
                self.optimizer, factor=args.plateau.factor, patience=args.plateau.patience)
        else:
            self.sched = None

        # Training config
        self.device = args.device
        self.epochs = args.epochs
        self.max_norm = args.max_norm

        # Checkpoints
        self.continue_from = args.continue_from
        self.checkpoint = Path(
            args.checkpoint_file) if args.checkpoint else None
        if self.checkpoint:
            logger.debug("Checkpoint will be saved to %s",
                         self.checkpoint.resolve())
        self.history_file = args.history_file

        self.best_state = None
        self.restart = args.restart
        # keep track of losses
        self.history = []

        # Where to save samples
        self.samples_dir = args.samples_dir

        # logging
        self.num_prints = args.num_prints

        # for seperation tests
        self.args = args

    def _serialize(self, path):
        package = {}
        package['model'] = serialize_model(self.model)
        package['optimizer'] = self.optimizer.state_dict()
        package['history'] = self.history
        package['best_state'] = self.best_state
        package['args'] = self.args
        torch.save(package, path)

    def _reset(self):
        load_from = None
        # Reset
        if self.checkpoint and self.checkpoint.exists() and not self.restart:
            load_from = self.checkpoint
        elif self.continue_from:
            load_from = self.continue_from

        if load_from:
            logger.info(f'Loading checkpoint model: {load_from}')
            package = torch.load(load_from, 'cpu')
            if load_from == self.continue_from and self.args.continue_best:
                self.model.load_state_dict(package['best_state'])
            else:
                self.model.load_state_dict(package['model']['state'])

            if 'optimizer' in package and not self.args.continue_best:
                self.optimizer.load_state_dict(package['optimizer'])
            self.history = package['history']
            self.best_state = package['best_state']

    def train(self):
        if self.history:
            logger.info("Replaying metrics from previous run")
        for epoch, metrics in enumerate(self.history):
            info = " ".join(f"{k}={v:.5f}" for k, v in metrics.items())
            logger.info(f"Epoch {epoch}: {info}")

        for epoch in range(len(self.history),self.epochs):
            self.model.train()
            logger.info("Training...")
            start = time.time()
            train_loss = self._run_one_epoch(epoch)
            logger.info(bold(f'Train Summary | End of Epoch {epoch + 1} | '
                             f'Time {time.time() - start:.2f}s | Train Loss {train_loss:.5f}'))

            # Cross validation
            logger.info('-' * 70)
            logger.info('Cross validation...')
            self.model.eval()
            with torch.no_grad():
                valid_loss = self._run_one_epoch(epoch,cross_valid= True)
            logger.info(bold(f'Valid Summary | End of Epoch {epoch + 1} | '
                             f'Time {time.time() - start:.2f}s | Valid Loss {valid_loss:.5f}'))
            if self.sched:
                if self.args.lr_sched == 'plateau':
                    self.sched.step(valid_loss)
                else:
                    self.sched.step()
            best_loss = min(pull_metric(self.history, 'valid') + [valid_loss])
            metrics = {'train': train_loss,
                       'valid': valid_loss, 'best': best_loss}
            # Save the best model
            if valid_loss == best_loss or self.args.keep_last:
                logger.info(bold('New best valid loss %.4f'), valid_loss)
                self.best_state = copy_state(self.model.state_dict())
            self.history.append(metrics)
            info = " | ".join(
                f"{k.capitalize()} {v:.5f}" for k, v in metrics.items())
            logger.info('-' * 70)
            logger.info(bold(f"Overall Summary | Epoch {epoch + 1} | {info}"))

            if distrib.rank == 0:
                json.dump(self.history, open(self.history_file, "w"), indent=2)
                # Save model each epoch
                if self.checkpoint:
                    self._serialize(self.checkpoint)
                    logger.debug("Checkpoint saved to %s",
                                 self.checkpoint.resolve())

    def _run_one_epoch(self, epoch, cross_valid=False):
        total_loss = 0
        data_loader = self.tr_loader if not cross_valid else self.cv_loader

        # get a different order for distributed training, otherwise this will get ignored
        data_loader.epoch = epoch

        label = ["Train", "Valid"][cross_valid]
        name = label + f" | Epoch {epoch + 1}"
        logprog = LogProgress(logger, data_loader,
                              updates=self.num_prints, name=name)
        for i, data in enumerate(logprog):
            data,label_voice_signals,window_idx,_= [x.to(self.device) for x in data]

            data, means, stds = normalize_input(data)
            valid_length = self.model.valid_length(data.shape[-1])
            delta = valid_length - data.shape[-1]
            padded = F.pad(data, (delta // 2, delta - delta // 2))

            if self.args.model == "version_1":
                output_signal = self.model(padded, window_idx)
            elif self.args.model == "version_2":
                spk_embedding = self.speaker_model(ref_spk)
                output_signal = self.model(padded, window_idx,spk_embedding)
            else:
                assert 0
                
            output_signal = center_trim(output_signal, data)

            output_signal = unnormalize_input(output_signal, means, stds)
            output_voices = output_signal[:, 0]
            # only eval last layer
            if cross_valid:
                estimate_source = estimate_source[-1:]

            loss = 0
            cnt = len(estimate_source)
            # apply a loss function after each layer
            with torch.autograd.set_detect_anomaly(True):
                loss = self.model.loss(output_voices,label_voice_signals)
                if not cross_valid:
                    # optimize model in training mode
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   self.max_norm)
                    self.optimizer.step()

            total_loss += loss.item()
            logprog.update(loss=format(total_loss / (i + 1), ".5f"))

            # Just in case, clear some memory
            del loss, estimate_source
        return distrib.average([total_loss / (i + 1)], i + 1)[0]
