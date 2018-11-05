import logging
import os
import time
import datetime
import hashlib
import json
import certifi

from rasa_core.tracker_store import InMemoryTrackerStore
from rasa_core.events import ActionExecuted, BotUttered, UserUttered

from elasticsearch import Elasticsearch

logger = logging.getLogger(__name__)

ENABLE_ANALYTICS = os.getenv('ENABLE_ANALYTICS', 'False').lower() == 'true'
ENVIRONMENT_NAME = os.getenv('ENVIRONMENT_NAME', 'locahost')
BOT_VERSION = os.getenv('BOT_VERSION', 'notdefined')
HASH_GEN = hashlib.md5()

def gen_id(timestamp):
    HASH_GEN.update(str(timestamp).encode('utf-8'))
    _id = HASH_GEN.hexdigest()[10:]
    return _id

class ElasticTrackerStore(InMemoryTrackerStore):
    def __init__(self, host="http://elasticsearch:9200", user="", password="", index="messages", domain=None):
        # create ES connection
        es = Elasticsearch([host],http_auth=(user, password),timeout=60, max_retries=10, retry_on_timeout=True, ca_certs=certifi.where())

        es = Elasticsearch([os.getenv('ELASTICSEARCH_URL', 'elasticsearch:9200')])
        super(ElasticTrackerStore, self).__init__(domain)

    def save_user_message(self, tracker):
        if not tracker.latest_message.text:
            return

        timestamp = time.time()

        message = {
            'environment': ENVIRONMENT_NAME,
            'version': BOT_VERSION,

            'user_id': tracker.sender_id,
            'is_bot': False,
            'timestamp': timestamp,

            'text': tracker.latest_message.text,

            'entities': tracker.latest_message.entities,
            'intent_name': tracker.latest_message.intent['name'],
            'intent_confidence': tracker.latest_message.intent['confidence'],

            'utter_name': '',
            'is_fallback': False,
        }

        es.index(index='messages', doc_type='message',
                 id='{}_user_{}'.format(ENVIRONMENT_NAME, gen_id(timestamp)),
                 body=json.dumps(message))

    def save_bot_message(self, tracker):
        if not tracker.latest_message.text:
            return

        utters = []
        index = len(tracker.events) - 1
        while True:
            evt = tracker.events[index]
            if isinstance(evt, UserUttered):
                break
            elif isinstance(evt, BotUttered):
                while not isinstance(evt, ActionExecuted):
                    index -= 1
                    evt = tracker.events[index]
                utters.append(evt.action_name)
            index -= 1


        time_offset = 0
        for utter in utters[::-1]:
            time_offset += 100

            timestamp = (
                datetime.datetime.now() +
                datetime.timedelta(milliseconds=time_offset)
            ).timestamp()

            message = {
                'environment': ENVIRONMENT_NAME,
                'version': BOT_VERSION,
                'user_id': tracker.sender_id,

                'is_bot': True,

                'text': '',
                'timestamp': timestamp,

                'entities': [],
                'intent_name': '',
                'intent_confidence': '',

                'utter_name': utter,
                'is_fallback': utter == 'action_default_fallback',
            }

            es.index(index='messages', doc_type='message',
                     id='{}_bot_{}'.format(ENVIRONMENT_NAME, gen_id(timestamp)),
                     body=json.dumps(message))

    def save(self, tracker):
        if ENABLE_ANALYTICS:
            try:
                self.save_user_message(tracker)
                self.save_bot_message(tracker)
            except Exception as ex:
                logger.error('Could not track messages '
                             'for user {}'.format(tracker.sender_id))
                logger.error(str(ex))

        super(ElasticTrackerStore, self).save(tracker)
