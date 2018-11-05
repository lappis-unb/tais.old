from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import logging
import warnings
import os
import yaml


from rasa_core.utils import configure_colored_logging, read_yaml_file, AvailableEndpoints
from rasa_core.run import start_server, load_agent
from rasa_core.interpreter import NaturalLanguageInterpreter
from rasa_core.tracker_store import TrackerStore

from rasa_core import utils
from rasa_core.agent import Agent
from rasa_core.policies.keras_policy import KerasPolicy
from rasa_core.policies.memoization import MemoizationPolicy
from rasa_core.featurizers import MaxHistoryTrackerFeaturizer, BinarySingleStateFeaturizer, LabelTokenizerSingleStateFeaturizer, FullDialogueTrackerFeaturizer

from rasa_nlu.training_data import load_data
from rasa_nlu import config
from rasa_nlu.model import Trainer

from bot.actions.connector import RocketChatInput
from bot.actions.tracker_store import ElasticTrackerStore
from bot.actions.fallback import CustomFallbackPolicy

logger = logging.getLogger(__name__)
configure_colored_logging(loglevel='DEBUG')

#Env Vars
EPOCHS              = int(os.getenv('EPOCHS', 30))
BATCH_SIZE          = int(os.getenv('BATCH_SIZE', 10))
VALIDATION_SPLIT    = float(os.getenv('VALIDATION_SPLIT', 0.2))
NLU_THRESHOLD       = float(os.getenv('NLU_THRESHOLD', 0.6))
CORE_THRESHOLD      = float(os.getenv('CORE_THRESHOLD', 0.6))
MAX_HISTORY         = int(os.getenv('MAX_HISTORY', 2))
FB_ACTION_NAME = str(os.getenv('FALLBACK_ACTION_NAME', 'utter_default'))
AUGMENTATION        = int(os.getenv('AUGMENTATION', 50))

ROCKETCHAT_URL      = os.getenv('ROCKETCHAT_URL', 'http://localhost:3000')
ROCKETCHAT_USERNAME = os.getenv('ROCKETCHAT_USERNAME', 'admin')
ROCKETCHAT_PASSWORD = os.getenv('ROCKETCHAT_PASSWORD', 'admin')

# Elasticsearch
ES_HOST             = os.environ['ES_HOST'] if 'ES_HOST' in os.environ else 'http://elasticsearch:9200/'
ES_USER             = os.environ['ES_USER'] if 'ES_USER' in os.environ else ''
ES_PASS             = os.environ['ES_PASS'] if 'ES_PASS' in os.environ else ''
ES_INDEX            = os.environ['ES_INDEX'] if 'ES_INDEX' in os.environ else 'messages'


def train_core(domain_file="bot/domain.yml",
                model_path="bot/models/dialogue",
                training_data_file="bot/data/stories"):
    
    MemoizationPolicy.USE_NLU_CONFIDENCE_AS_SCORE = True
    # keras_1 = KerasPolicy(
    #             MaxHistoryTrackerFeaturizer(
    #                 BinarySingleStateFeaturizer(),
    #                 max_history=MAX_HISTORY
    #                 )
    #             )
    keras_2 = KerasPolicy(
                FullDialogueTrackerFeaturizer(
                    LabelTokenizerSingleStateFeaturizer()
                )
            )

    agent = Agent(domain_file,
                policies=[
                    keras_2,
                    MemoizationPolicy(max_history=MAX_HISTORY),
                    CustomFallbackPolicy(
                        fallback_action_name=FB_ACTION_NAME,
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
    
    training_data = load_data('bot/data/intents')
    trainer = Trainer(config.load("bot/nlu_config.yml"))
    trainer.train(training_data)
    model_directory = trainer.persist('bot/models/nlu/', fixed_model_name="current")

    return model_directory

def run_rocketchat(model_path="bot/models/dialogue", nlu_dir="bot/models/nlu/default/current", endpoints_file=None):

    input_channel = RocketChatInput(
        user=ROCKETCHAT_USERNAME,
        password=ROCKETCHAT_PASSWORD,
        server_url=ROCKETCHAT_URL
    )

    # _endpoints = AvailableEndpoints.read_endpoints(endpoints_file)
    _interpreter = NaturalLanguageInterpreter.create(nlu_dir)
    _tracker_store = ElasticTrackerStore(host=ES_HOST, user=ES_USER, password=ES_PASS, index=ES_INDEX)

    _agent = load_agent(model_path,
                        _interpreter, 
                        endpoints_file,
                        tracker_store=_tracker_store)

    http_server = start_server([input_channel], "", "", 5005, _agent)

    try:
        http_server.serve_forever()
    except Exception as exc:
        logger.exception(exc)

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
    elif task == "run_rocketchat":
        run_rocketchat()
    
