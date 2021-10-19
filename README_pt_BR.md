# Classificação de Fake News com base na reputação dos usuários, seus seguidores e seguidos

O projeto é uma implementação do método ICS, 
porém com adaptações para considerar a influência dos 
seguidores e seguidos de um usuário.

### Reprodução

Para uma demonstração da solução em funcionamento, 
é possível acessar uma cápsula de execução hospedada
pela CodeOcean neste [DOI](https://doi.org/10.24433/CO.7770742.v1).


### Reprodução Local

Caso queira executar localmente, siga as instruções:

Instalar versão do Python >= 3.9. 
Caso tenha instalada uma versão >= 3.6, deve funcionar
porém não foi testado.

Instalar as bibliotecas necessárias: scikit-learn, pandas e tqdm.
Caso utilize o gerenciador de pacotes pip, execute os seguintes 
comandos:

```
pip install scikit-learn
pip install pandas
pip install tqdm
```

Todos os arquivos CSV empregados no estudo estão já disponíveis nesta
pasta sob o diretório `/datasets`.

Para execução, basta chamar a função main no arquivo `main.py`.
Pela linha de comando:

[Windows]
```
python main.py
```

[Linux]
```
python3 main.py
```

