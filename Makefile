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
	@echo "    train"
	@echo "        Train NLU and CORE."
	@echo "    cmdline"
	@echo "        Starts a commandline session and allows you to chat with the bot."
	@echo "    visualize"
	@echo "        Draws the dialogue training data as a graph."

install:
	pip install -r requirements.txt
	
interactive:
	python -m rasa_core.train -d bot/domain.yml -s bot/data/stories -o bot/models/dialogue --epochs 30 --online --nlu bot/models/nlu/default/current/

train-nlu:
	python bot.py train-nlu
	
train-core:
	python -m rasa_core.train -d bot/domain.yml -s bot/data/stories -o bot/models/dialogue

train:
	python bot.py train

run-debug:
	python -m rasa_core.run -d bot/models/dialogue -u bot/models/nlu/default/current --debug 

run:
	python -m rasa_core.run -d bot/models/dialogue -u bot/models/nlu/default/current -vv

visualize:
	python -m rasa_core.visualize -s bot/data/stories -d bot/domain.yml -o story_graph.png
