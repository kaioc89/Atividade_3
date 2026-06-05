# Banco de Dados - Atividade 2

Este diretorio reune os artefatos de banco da atividade, alinhados ao fluxo oficial do repositorio.

O banco local usado pela infra padrao e o PostgreSQL do `docker compose`, com configuracao lida de `.env` na raiz do projeto. O nome padrao do banco e `app_dev`.

## Estrutura

- `database/ddl_banco/ddl_atividade_2.sql`: DDL base da atividade.
- `database/oab_bench/`: arquivos locais do dataset `OAB_Bench`.
- `database/respostas_alunos/`: CSVs com respostas da Atividade 1.
- `database/dumps/`: dumps gerados a partir do banco local.
- `database/scripts_etl/`: scripts de carga de datasets e respostas.

## Preparacao da base

Antes de gerar qualquer dump, use a infraestrutura oficial da raiz:

```bash
make db-up
make db-migrate-or-create
make db-restore-validate
```

Isso garante que o PostgreSQL local esteja de pe, com o backup principal restaurado e validado.

Quando o banco ativo ja foi restaurado manualmente ou quando a branch introduz evolucoes de schema, use:

```bash
make db-ensure-schema
```

Esse fluxo atualiza apenas estrutura e seeds determinísticos de AV3 no `DATABASE_URL` ativo. Ele nao executa restore, nao limpa schemas e nao sobrescreve `backup_atividade_2.sql`.

## Faixas canonicas de perguntas carregadas

A base canonica do projeto nao carrega o dataset inteiro. Ela carrega apenas as faixas efetivamente cobertas pelos alunos:

- `OAB_Bench`: abertas `71..140`
- `OAB_Exames`: multipla escolha `739..1476`

Com isso, os dumps oficiais ja nascem sem perguntas fora do escopo respondido pela turma.

## Cinco comandos que os desenvolvedores vao usar

### 1. Gerar apenas o dump do DDL

```bash
make db-dump-structure
```

Gera:

```text
database/dumps/dump_estrutura_vazia.sql
```

### 2. Gerar o dump das perguntas

```bash
make db-dump-questions
```

Gera:

```text
database/dumps/dump_perguntas.sql
```

### 3. Gerar o dump das respostas

```bash
make db-dump-responses
```

Gera:

```text
database/dumps/dump_respostas.sql
```

### 4. Promover a base atual para o backup canônico da raiz

```bash
make db-backup-promote
```

Atualiza:

```text
backup_atividade_2.sql
```

O arquivo timestampado em `outputs/backup/atividade_2_YYYYmmdd_HHMMSS.sql` sempre e gerado por `make db-backup` e preserva o historico local. `backup_atividade_2.sql` e o artefato canônico protegido: ele so e sobrescrito por promocao explicita e somente depois de validar que `public.respostas_atividade_1`, `public.avaliacoes_juiz`, `av3.rag_chunks`, `av3.rag_embeddings` e `av3.retrieval_runs` ativos possuem contagem maior que zero. A base fixa de reset usada por `make db-migrate-or-create` fica em `backup_atividade_2_reset.sql`.

### 5. Gerar tudo de uma vez

```bash
make db-dump-all
```

Esse comando gera:

- `database/dumps/dump_estrutura_vazia.sql`
- `database/dumps/dump_perguntas.sql`
- `database/dumps/dump_respostas.sql`
- `outputs/backup/atividade_2_YYYYmmdd_HHMMSS.sql`

## Restore esperado

Os dumps fracionados devem ser usados nesta ordem:

1. `dump_estrutura_vazia.sql`
2. `dump_perguntas.sql`
3. `dump_respostas.sql`

Os dumps de perguntas e respostas sao `data-only`, entao assumem que a estrutura ja existe no banco de destino.

## Fluxo para incorporar respostas de um novo aluno

Exemplo: o aluno `x` gerou seus arquivos e colocou no diretorio:

```text
database/respostas_alunos/respostas_objetivas_aluno_x.csv
database/respostas_alunos/respostas_discursivas_aluno_x.csv
```

### Passo a passo

1. Suba o banco e restaure a base atual do projeto:

```bash
make db-up
make db-migrate-or-create
make db-restore-validate
```

2. Confirme que os arquivos do novo aluno estao em `database/respostas_alunos/`.

O importador descobre automaticamente todos os arquivos com estes padroes:

- `respostas_objetivas_*.csv`
- `respostas_discursivas_*.csv`

3. Reimporte as respostas para todos os modelos presentes nesses CSVs:

```bash
.venv/bin/python database/scripts_etl/importar_respostas_atividade_1.py --replace
```

Use `--replace` para reconstruir as respostas dos modelos encontrados nos CSVs e evitar duplicidade.

4. Gere um novo dump de respostas:

```bash
make db-dump-responses
```

5. Promova a base atual para o backup canônico da raiz:

```bash
make db-backup-promote
```

6. Se voce tambem quiser atualizar todos os artefatos fracionados:

```bash
make db-dump-all
```

## Observacoes importantes sobre o importador

- Para `OAB_Exames`, o importador aceita `id_pergunta` numerico.
- Para `OAB_Bench`, o importador aceita:
  - `id_pergunta` real do banco;
  - `question_id` textual salvo em `metadados`;
  - numero sequencial da questao dentro do dataset.
- `texto_resposta` vazio e preservado como string vazia quando o modelo nao respondeu nada.
- A carga canonica de perguntas ja considera apenas as faixas efetivamente respondidas pelos alunos.

## Scripts ETL

Os scripts em `database/scripts_etl/` usam a convencao atual do projeto:

- banco padrao `app_dev`;
- arquivos lidos dentro de `database/oab_bench/` e `database/respostas_alunos/`.

### OAB_Exames

```bash
HF_HOME=.hf_cache .venv/bin/python database/scripts_etl/import_oab_exames.py --start-id 739 --end-id 1476 --truncate
```

### OAB_Bench

```bash
.venv/bin/python database/scripts_etl/import_oab_bench.py --start-seq 71 --end-seq 140 --replace
```

### Respostas da Atividade 1

```bash
.venv/bin/python database/scripts_etl/importar_respostas_atividade_1.py --replace
```
