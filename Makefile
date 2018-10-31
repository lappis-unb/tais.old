TEST_PATH=./

help:
	@echo "    install"
	@echo "        Installs the needed dependencies."
	@echo "    interactive"
	@echo "        Train the dialogue model in interactive mode, allowing you to correct it."
	@echo "    train-nlu"
	@echo "        Train the natural language understanding using Rasa NLU."
	@echo "    train-core"
	@echo "        Train a dialogue model using Rasa core."
	@echo "    cmdline"
	@echo "        Starts a commandline session and allows you to chat with the bot."
	@echo "    visualize"
	@echo "        Draws the dialogue training data as a graph."

install:
	pip install rasa_core
	pip install rasa_nlu[spacy]
	python -m spacy download en_core_web_md
	python -m spacy link --force en_core_web_md en

interactive:
	python -m rasa_core.train -d bot/domain.yml -s bot/data/stories -o bot/models/dialogue --epochs 30 --online --nlu bot/models/nlu/default/current/

train-nlu:
	python -m rasa_nlu.train -c bot/nlu_config.yml --fixed_model_name current --data bot/data/intents/ -o bot/models/dialogue
	
train-core:
	python -m rasa_core.train -s bot/data/stories -d bot/domain.yml -o bot/models/dialogue --epochs 30 --augmentation 50 --history 3 --nlu_threshold 0.6 --core_threshold 0.6 --fallback_action_name 'utter_default'

cmdline:
	python -m rasa_core.run -d bot/models/dialogue -u bot/models/nlu/default/current --debug 

visualize:
	python -m rasa_core.visualize -s bot/data/stories -d bot/domain.yml -o story_graph.png
