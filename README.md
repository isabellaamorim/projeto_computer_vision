# projeto_computer_vision

### Resumo
Esse projeto tem o intuito de avaliar uma imagem estática de uma pessoa e fazer uma predição aproximada da temperatura do local que essa pessoa se encontra.

Para código base de identificação de roupas e pessoas, foi utilizado o repositório abaixo:

https://github.com/axinc-ai/ailia-models/tree/master/deep_fashion/clothing-detection

Essa temperatura esta em uma escala de 0 a 1, sendo 0 o máximo de frio e 1 o máximo de calor. Essa escala foi dividida entre Frio, Ameno e Quente.

Um site, utilizando Flask, foi desenvolvido para facilitar a visualização e a esperiÊncia do usuário com a aplicação.

### Instalando as dependências
Com o projeto baixado na sua máquina local, instale as seguintes dependências:

pip install watchdog

pip install Flask

pip install Werkzeug

### Rodando o projeto

No seu terminal, rode o comando python ./app.py e acesse a rota fornecida no prompt de comando no dashboart seu navegador padrão. (Preferência para o Google chrome)

Insira uma imagem no formato expecificado, clique em upload e, apos um tempo de processamento, veja a mágica acontecer.

