# -*- coding: utf-8 -*-
import glob
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime

import chainer
import logzero
from logzero import logger


class Resource(object):
    """
    Helper class for the experiment.
    """

    def __init__(self, args, train=True):
        self.args = args  # argparse object
        self.logger = logger
        self.start_time = datetime.today()
        self.config = None  # only used for the inference

        if train:  # for training
            self.output_dir = self._return_output_dir()
            self.create_output_dir()
            log_filename = 'train.log'
        else:  # for inference
            self.output_dir = args.out
            log_filename = 'inference.log'

        log_name = os.path.join(self.output_dir, log_filename)
        logzero.logfile(log_name)
        self.log_name = log_name
        self.logger.info('Log filename: [{}]'.format(log_name))

    def _return_output_dir(self):
        if self.args.dataset == "PE":
            dir_name = "{}/rep={}_decoder={}_elmo={}_actype={}_linktype={}_"\
                "epoch={}_optim={}_lr={}_batch={}_drop={}_droplstm={}_"\
                "lac={}_lshell={}_lacshell={}_ltype={}"\
                "/seed={}_date={}".format(
                    self.args.dataset,
                    self.args.reps_type,
                    self.args.decoder,
                    self.args.use_elmo,
                    self.args.ac_type_alpha,
                    self.args.link_type_alpha,
                    self.args.epoch,
                    self.args.optimizer,
                    self.args.lr,
                    self.args.batchsize,
                    self.args.dropout,
                    self.args.dropout_lstm,
                    self.args.lstm_ac,
                    self.args.lstm_shell,
                    self.args.lstm_ac_shell,
                    self.args.lstm_type,
                    self.args.seed,
                    self.args.dir_prefix)
        else:
            dir_name = "{}/rep={}_decoder={}_elmo={}_actype={}_linktype={}_"\
                "epoch={}_optim={}_lr={}_batch={}_drop={}_droplstm={}_"\
                "lac={}_lshell={}_lacshell={}_ltype={}"\
                "/seed={}_date={}/iter={}_fold={}".format(
                    self.args.dataset,
                    self.args.reps_type,
                    self.args.decoder,
                    self.args.use_elmo,
                    self.args.ac_type_alpha,
                    self.args.link_type_alpha,
                    self.args.epoch,
                    self.args.optimizer,
                    self.args.lr,
                    self.args.batchsize,
                    self.args.dropout,
                    self.args.dropout_lstm,
                    self.args.lstm_ac,
                    self.args.lstm_shell,
                    self.args.lstm_ac_shell,
                    self.args.lstm_type,
                    self.args.seed,
                    self.args.dir_prefix,
                    self.args.iteration,
                    self.args.fold)

        output_dir = os.path.abspath(os.path.join(self.args.out, dir_name))
        return output_dir

    def create_output_dir(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            self.logger.info(
                'Output Dir is created at [{}]'.format(self.output_dir))
        else:
            self.logger.info(
                'Output Dir [{}] alreaady exists'.format(self.output_dir))

    def dump_git_info(self):
        """
        returns git commit id, diffs from the latest commit
        """
        if os.system('git rev-parse 2> /dev/null > /dev/null') == 0:
            self.logger.info(
                'Git repository is found. Dumping logs & diffs...')
            git_log = '\n'.join(
                l for l in
                subprocess.check_output('git log --pretty=fuller | head -7', shell=True).decode('utf8').split('\n') if
                l)
            self.logger.info(git_log)

            git_diff = subprocess.check_output(
                'git diff', shell=True).decode('utf8')
            self.logger.info(git_diff)
        else:
            self.logger.warn('Git repository is not found. Continue...')

    def dump_command_info(self):
        """
        returns command line arguments / command path / name of the node
        """
        self.logger.info('Command name: {}'.format(' '.join(sys.argv)))
        self.logger.info('Command is executed at: [{}]'.format(os.getcwd()))
        self.logger.info(
            'Program is running at: [{}]'.format(os.uname().nodename))

    def dump_library_info(self):
        """
        returns chainer, cupy and cudnn version info
        """
        self.logger.info('Chainer Version: [{}]'.format(chainer.__version__))
        try:
            self.logger.info('CuPy Version: [{}]'.format(
                chainer.cuda.cupy.__version__))
        except AttributeError:
            self.logger.warn('CuPy was not found in your environment')

        if chainer.cuda.cudnn_enabled:
            self.logger.info('CuDNN is available')
        else:
            self.logger.warn('CuDNN is not available')

    def dump_python_info(self):
        """
        returns python version info
        """
        self.logger.info('Python Version: [{}]'.format(
            sys.version.replace('\n', '')))

    def save_config_file(self):
        """
        save argparse object into config.json
        config.json is used during the inference
        """
        with open(os.path.join(self.output_dir, 'config.json'), 'w') as fo:
            dumped_config = json.dumps(
                vars(self.args), sort_keys=True, indent=4)
            fo.write(dumped_config)
            self.logger.info('HyperParameters: {}'.format(dumped_config))

    def save_vocab_file(self):
        shutil.copy(self.args.vocab_path, self.output_dir)
        self.logger.info('Vocab file {} has been copied to {}'.format(
            self.args.vocab_path, self.output_dir))

    def dump_duration(self):
        end_time = datetime.today()
        self.logger.info('EXIT TIME: {}'.format(
            end_time.strftime('%Y%m%d - %H:%M:%S')))
        duration = end_time - self.start_time
        logger.info('Duration: {}'.format(str(duration)))
        logger.info('Remember: log is saved in {}'.format(self.output_dir))

    def load_config(self):
        """
        load config.json and recover hyperparameters that are used during the training
        """
        config_path = os.path.join(self.args.out, 'config.json')
        self.config = json.load(open(config_path, 'r'))
        self.logger.info('Loaded config from {}'.format(config_path))

    def get_vocab_path(self):
        query = os.path.join(self.args.out, '*.dict')
        return glob.glob(query)[0]

    def get_model_path(self):
        query = os.path.join(
            self.args.out, 'model_epoch_{}'.format(self.args.epoch))
        return glob.glob(query)[0]
