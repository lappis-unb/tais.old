all:
	python -m rasa_core.train -d domain.yml -s data/stories -o models/dialogue --epochs 400
	python -m rasa_nlu.train -c nlu_config.yml --data data/intents -o models --fixed_model_name nlu --project tais --verbose
	python -m rasa_core.run --enable_api -d models/dialogue -u models/tais/nlu
run:
	python -m rasa_core.run --enable_api -d models/dialogue -u models/tais/nlu
