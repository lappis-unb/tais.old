from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import logging
import warnings
import os

from rasa_core import utils
from rasa_core.agent import Agent
from rasa_core.policies.fallback import FallbackPolicy
from bot.actions.fallback import CustomFallbackPolicy
from rasa_core.policies.keras_policy import KerasPolicy
from rasa_core.policies.memoization import MemoizationPolicy, AugmentedMemoizationPolicy
from rasa_core.featurizers import MaxHistoryTrackerFeaturizer, BinarySingleStateFeaturizer

logger = logging.getLogger(__name__)
EPOCHS = int(os.getenv('EPOCHS', 30))
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 30))
VALIDATION_SPLIT = int(os.getenv('VALIDATION_SPLIT', 30))
NLU_THRESHOLD = float(os.getenv('NLU_THRESHOLD', 0.6))
CORE_THRESHOLD = float(os.getenv('CORE_THRESHOLD', 0.6))
MAX_HISTORY = int(os.getenv('MAX_HISTORY', 2))
FALLBACK_ACTION_NAME = str(os.getenv('FALLBACK_ACTION_NAME', 'utter_default'))
AUGMENTATION = int(os.getenv('AUGMENTATION', 50))

def train_core(domain_file="bot/domain.yml",
                   model_path="bot/models/dialogue",
                   training_data_file="bot/data/stories"):
    
    MemoizationPolicy.USE_NLU_CONFIDENCE_AS_SCORE = True
    agent = Agent(domain_file,
                policies=[
                        KerasPolicy(
                            MaxHistoryTrackerFeaturizer(
                                BinarySingleStateFeaturizer(),
                                max_history=MAX_HISTORY)),
                        MemoizationPolicy(max_history=MAX_HISTORY),
                        CustomFallbackPolicy(
                            fallback_action_name=FALLBACK_ACTION_NAME,
                            nlu_threshold=NLU_THRESHOLD,
                            core_threshold=CORE_THRESHOLD)]
    )

    training_data = agent.load_data(training_data_file,augmentation_factor=AUGMENTATION)
    agent.train(
        training_data,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT
    )

    agent.persist(model_path)
    return agent

def train_nlu():
    from rasa_nlu.training_data import load_data
    from rasa_nlu import config
    from rasa_nlu.model import Trainer

    training_data = load_data('bot/data/intents')
    trainer = Trainer(config.load("bot/nlu_config.yml"))
    trainer.train(training_data)
    model_directory = trainer.persist('bot/models/nlu/', fixed_model_name="current")

    return model_directory


if __name__ == '__main__':
    utils.configure_colored_logging(loglevel="INFO")

    parser = argparse.ArgumentParser(description='starts the bot')

    parser.add_argument(
            'task',
            choices=["train-nlu", "train-core", "train"],
            help="what the bot should do - e.g. run or train?")
    task = parser.parse_args().task

    # decide what to do based on first parameter of the script
    if task == "train-nlu":
        train_nlu()
    elif task == "train-core":
        train_core()
    elif task == "train":
        train_nlu()
        train_core()
