import logging
from .tensorboard import TensorboardWriter
from ..config import config


class StyleExtractorLogger:
    def __init__(self, epoch_now=0):
        self.epoch_now = epoch_now
        self.avg_step_loss = 0.0
        self.avg_epoch_loss = 0.0
        self.tensorboard = TensorboardWriter()

    def add_step_and_loss(self, loss, step):
        self.avg_step_loss += loss
        self.avg_epoch_loss += loss

        if step % config.tensorboard.loss_step == 0:
            self.avg_step_loss /= config.tensorboard.loss_step
            logging.info('epoch %d, %d step, loss = %.6f' % (self.epoch_now, step, self.avg_step_loss))

            tag = 'style_extractor/step_loss'
            x = step + self.steps_of_an_epoch * (self.epoch_now - 1)
            y = self.avg_step_loss
            self.tensorboard.add_scalar(tag=tag, x=x, y=y)

            self.avg_step_loss = 0.0

    def show_epoch_loss(self):
        self.avg_epoch_loss /= self.steps_of_an_epoch

        logging.info('Writing epoch loss...')

        tag = 'style_extractor/epoch_loss'
        x = self.epoch_now
        y = self.avg_epoch_loss
        self.tensorboard.add_scalar(tag=tag, x=x, y=y)

        self.avg_epoch_loss = 0.0

    def record_test_error(self, accuracy):
        tag = 'validate/accuracy'
        x = self.epoch_now
        y = accuracy
        self.tensorboard.add_scalar(tag=tag, x=x, y=y)

    @property
    def steps_of_an_epoch(self):
        return config.dataset.triplet_train_dataset_num // config.style_extractor.batch_size
