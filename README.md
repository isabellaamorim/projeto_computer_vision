# projeto_computer_vision

### Resumo
Esse projeto tem o intuito de avaliar uma imagem estática de uma pessoa e fazer uma predição aproximada do clima do local que essa pessoa se encontra com base nas roupas que a pessoa estiver usando.

Para o código base de identificação de roupas e pessoas, foi utilizado o repositório abaixo:

https://github.com/axinc-ai/ailia-models/tree/master/deep_fashion/clothing-detection

O clima está em uma escala de 0 a 1, sendo 0 o máximo de frio e 1 o máximo de calor. Essa escala foi dividida entre Frio, Ameno e Quente.

Um site, utilizando Flask, foi desenvolvido para facilitar a visualização e a experiência do usuário com a aplicação.

### Instalando as dependências
Com o projeto baixado na sua máquina local, no diretório abra o terminal e execute o seguinte comando para instalar as dependências:

pip install -r requirements.txt

### Rodando o projeto

No seu terminal, rode o comando python ./app.py e acesse a rota fornecida no prompt de comando no dashboart seu navegador padrão. (Preferência para o Google chrome)

Insira uma imagem no formato expecificado (na pasta 'imagens teste' há algumas imagens que podem ser utilizadas), clique em upload e, após um tempo de processamento, veja a mágica acontecer.

OBS: A primeira imagem demorará mais para ser analisada, pois o código precisará fazer o dowload dos pesos. 

