## Instruções de Uso

### Pré-requisitos

- Compilador C++ compatível com C++17.
- `make` instalado no sistema.

### Comandos do Makefile

O `Makefile` inclui diversos comandos para facilitar a manipulação do projeto. Abaixo estão os comandos disponíveis e suas descrições:

- **help**: Exibe esta ajuda.
- **build**: Compila o projeto e gera o executável no diretório `./build`.
- **run**: Executa o programa compilado.
- **clear**: Limpa todos os arquivos gerados durante a compilação.

### Exemplos de Uso

1. **Compilar o Projeto**

   Para compilar o projeto, use o comando:
   ```sh
   make build
    ```
2. **Executar o Programa**
    
    Após compilar o projeto, você pode executá-lo com o comando:
   ```sh
   make run
    ```

3. **Limpar Arquivos Compilados**

    Para limpar todos os arquivos gerados durante a compilação, utilize:
    ```sh
    make clean
    ```

4. **Exibir Ajuda**

    Para exibir a ajuda com a descrição de todos os comandos disponíveis no Makefile, use:
    ```sh
    make help
    ```