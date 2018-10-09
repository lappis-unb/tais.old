# Tais - Assistente Virtual da Cultura

## Ambiente RocketChat

```sh
sudo docker-compose up -d rocketchat
```

Entre no rocketchat com o login `admin` e senha `admin`. Execute os comandos
a seguir para configurar e rodar a Taís

```sh
python3 scripts/bot_config.py
sudo docker-compose up tais
```

Para que a assistente virtual inicie a conversa você deve criar um `trigger`.
Para isso, entre no rocketchat como `admin`, e vá no painel do Livechat na
seção de Triggers, clique em `New Trigger`. Preencha o Trigger da seguinte forma:

```yaml
Enabled: Yes
Name: Start Talk
Description: Start Talk
Condition: Visitor page URL
    Value: http://localhost:8000/
Action: Send Message
 Value: Impersonate next agent from queue
 Value: Oi eu sou a Taís, assistente virtual do minc, e estou aqui para te ajudar a esclarecer dúvidas sobre a Lei Rouanet, posso também solucionar problemas de proposta e projeto
```
## Testes

### Conversa no console

```sh
sudo docker build -t tais -f Dockerfile .
sudo docker run --rm --name tais -it -v $PWD/tais:/tais tais python train.py all 
```

### Teste de confiabilidade de frases

```sh
sudo docker run --rm --name tais -it -v $PWD/tais:/tais tais python confidence.py
```

```sh
python -m rasa_core.train -d domain.yml -s stories.md -o models/dialogue
```

### Treiando o bot

```sh
python -m rasa_core.run -d models/dialogue -u models/current/nlu
```

* Treinando `dialogue`:

Isso rá treinar o modelo de diálogo e armazenar o resultado em `models/dialogue`

```sh
python -m rasa_core.train -d domain.yml -s stories.md -o models/dialogue
```

* Treinar utilizando os exemplos do `nlu`:

```sh
python -m rasa_nlu.train -c nlu_config.yml --data nlu.md -o models --fixed_model_name nlu --project current --verbose
```

* Mandando `intents` diretamente na `domani`:

PAra testar o bot neste modo é necessário chamar as `intents` diretamente, como: `/greet` ou `goodbye`

```sh
python -m rasa_core.run -d models/dialogue
```

* Executando bot no console:

```sh
python -m rasa_core.run -d models/dialogue -u models/current/nlu
```

* Executando o bot com a `API` e arquivo de `log` habilitados:

```sh
python -m rasa_core.run --enable_api -d models/dialogue/ -u models/current/nlu/ -o out.log
```

### Testando bot com script

```sh
python scripts/api.py
```

## References
[Quickstart](https://rasa.com/docs/core/quickstart/)

## Training for production

```sh
python -m rasa_core.train -d bot/domain.yml -s bot/data/stories -o bot/models/dialogue --epochs 1200
```

```sh
python -m rasa_nlu.train -c bot/nlu_config.yml --data bot/data/intents -o bot/models --fixed_model_name nlu --project tais --verbose
```

```sh
python -m rasa_core.run -d bot/models/dialogue -u bot/models/tais/nlu
```
