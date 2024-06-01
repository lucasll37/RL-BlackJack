# [CT-213] Laboratório 3: Implementação de MergeSort com Threads

## Descrição do Projeto

Este projeto foi desenvolvido como parte do curso de Engenharia de Computação, especificamente para a disciplina de Sistemas Operacionais. O objetivo principal é implementar um algoritmo de ordenação Merge Sort utilizando threads para paralelizar o processo.

## Estrutura do Projeto

O projeto está organizado da seguinte forma:

```
├── .vscode/
├── build/
├── docs/
├── examples/
├── src/
│ ├── main.cpp
│ ├── mergesort.cpp
│ └── mergesort.h
├── Makefile
└── README.md
```

- **.vscode/**: Configurações específicas do Visual Studio Code.
- **build/**: Diretório onde o executável será gerado após a compilação.
- **docs/**: Roteiro proposto da atividade.
- **examples/**: Exemplos de uso fornecidos na aula.
- **src/**: Contém os arquivos de código-fonte (`main.cpp`, `mergesort.cpp`, `mergesort.h`).
- **Makefile**: Arquivo para automatizar a compilação, execução e limpeza do projeto.

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