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
from rasa_core.policies.memoization import MemoizationPolicy
from rasa_core.featurizers import MaxHistoryTrackerFeaturizer, BinarySingleStateFeaturizer

logger = logging.getLogger(__name__)
TRAINING_EPOCHS = int(os.getenv('TRAINING_EPOCHS', 150))

def train_core(domain_file="bot/domain.yml",
                   model_path="bot/models/dialogue",
                   training_data_file="bot/data/stories"):

    agent = Agent(domain_file,
                policies=[
                        KerasPolicy(
                            MaxHistoryTrackerFeaturizer(
                                BinarySingleStateFeaturizer(),
                                max_history=3)),
                        MemoizationPolicy(max_history=3),
                        CustomFallbackPolicy(
                            fallback_action_name="utter_default",
                            nlu_threshold=0.60,
                            core_threshold=0.60)]
    )

    training_data = agent.load_data(training_data_file)
    agent.train(
        training_data,
        epochs=TRAINING_EPOCHS,
        batch_size=10,
        validation_split=0.20
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
    model_directory = trainer.persist('bot/models/nlu/',
                                      fixed_model_name="current")

    return model_directory


if __name__ == '__main__':
    utils.configure_colored_logging(loglevel="INFO")

    parser = argparse.ArgumentParser(
            description='starts the bot')

    parser.add_argument(
            'task',
            choices=["train-nlu", "train-core", "run"],
            help="what the bot should do - e.g. run or train?")
    task = parser.parse_args().task

    # decide what to do based on first parameter of the script
    if task == "train-nlu":
        train_nlu()
    elif task == "train-core":
        train_core()
