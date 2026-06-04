-- ============================================================
-- Atividade 2 - PostgreSQL schema
-- Domain: juridical
-- Objective: store datasets, AV1 answers, prompt versions,
-- and LLM-as-a-Judge evaluations with traceability.
-- ============================================================

-- =========================
-- 1. Models
-- =========================
CREATE TABLE modelos (
    id_modelo SERIAL PRIMARY KEY,
    nome_modelo VARCHAR(100) NOT NULL,
    versao VARCHAR(50),
    parametro_precisao VARCHAR(20),
    tipo_modelo VARCHAR(20) NOT NULL
        CHECK (tipo_modelo IN ('candidato', 'juiz', 'ambos'))
);

-- =========================
-- 2. Datasets
-- =========================
CREATE TABLE datasets (
    id_dataset SERIAL PRIMARY KEY,
    nome_dataset VARCHAR(100) NOT NULL,
    dominio VARCHAR(50) NOT NULL
);

-- =========================
-- 3. Questions
-- =========================
CREATE TABLE perguntas (
    id_pergunta SERIAL PRIMARY KEY,

    id_dataset INTEGER NOT NULL
        REFERENCES datasets(id_dataset),

    enunciado TEXT NOT NULL,
    resposta_ouro TEXT NOT NULL,
    metadados JSONB
);

-- =========================
-- 4. AV1 answers
-- =========================
CREATE TABLE respostas_atividade_1 (
    id_resposta SERIAL PRIMARY KEY,

    id_pergunta INTEGER NOT NULL
        REFERENCES perguntas(id_pergunta),

    id_modelo INTEGER NOT NULL
        REFERENCES modelos(id_modelo),

    texto_resposta TEXT NOT NULL,

    tempo_inferencia_ms FLOAT
        CHECK (tempo_inferencia_ms >= 0),

    data_geracao TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =========================
-- 5. Judge prompt versions
-- =========================
CREATE TABLE prompt_juizes (
    id_prompt_juiz SERIAL PRIMARY KEY,

    id_dataset INTEGER NOT NULL
        REFERENCES datasets(id_dataset),

    versao INTEGER NOT NULL,
    ds_prompt TEXT NOT NULL,
    ds_persona TEXT NOT NULL,
    ds_contexto TEXT NOT NULL,
    ds_rubrica TEXT NOT NULL,
    ds_saida TEXT NOT NULL,

    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(120) NOT NULL DEFAULT 'system',
    ativo BOOLEAN NOT NULL DEFAULT FALSE,

    UNIQUE (id_dataset, versao)
);

-- =========================
-- 6. Judge evaluations
-- =========================
CREATE TABLE avaliacoes_juiz (
    id_avaliacao SERIAL PRIMARY KEY,

    id_resposta_ativa1 INTEGER NOT NULL
        REFERENCES respostas_atividade_1(id_resposta),

    id_modelo_juiz INTEGER NOT NULL
        REFERENCES modelos(id_modelo),

    id_prompt_juiz INTEGER NOT NULL
        REFERENCES prompt_juizes(id_prompt_juiz),

    nota_atribuida INTEGER NOT NULL
        CHECK (nota_atribuida BETWEEN 1 AND 5),

    chain_of_thought TEXT NOT NULL,
    papel_juiz VARCHAR(20),
    rodada_julgamento VARCHAR(30),
    motivo_acionamento TEXT,
    status_avaliacao VARCHAR(20) DEFAULT 'success',

    data_avaliacao TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =========================
-- 7. Auxiliary judge evaluation details
-- =========================
CREATE TABLE avaliacao_juiz_detalhes (
    id_detalhe SERIAL PRIMARY KEY,

    id_avaliacao INTEGER NOT NULL UNIQUE
        REFERENCES avaliacoes_juiz(id_avaliacao)
        ON DELETE CASCADE,

    legal_accuracy TEXT,
    hallucination_risk TEXT,
    rubric_alignment TEXT,
    requires_human_review BOOLEAN,
    criteria JSONB NOT NULL DEFAULT '{}'::jsonb,
    raw_output_jsonb JSONB,
    source_log_path TEXT,
    run_id TEXT,

    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- =========================
-- 8. Human meta-evaluations
-- =========================
CREATE TABLE meta_avaliacoes (
    id_meta_avaliacao SERIAL PRIMARY KEY,

    id_avaliacao INTEGER NOT NULL
        REFERENCES avaliacoes_juiz(id_avaliacao)
        ON DELETE CASCADE,

    nm_avaliador VARCHAR(120) NOT NULL,

    vl_nota INTEGER NOT NULL
        CHECK (vl_nota BETWEEN 1 AND 5),

    ds_justificativa TEXT NOT NULL,

    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- =========================
-- 9. AV3 RAG curation imports
-- =========================
CREATE SCHEMA IF NOT EXISTS av3;
CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public;

CREATE TABLE av3.curadoria_import_runs (
    id_import_run SERIAL PRIMARY KEY,
    dataset_code VARCHAR(10) NOT NULL,
    dataset_name VARCHAR(100) NOT NULL,
    filename TEXT NOT NULL,
    payload_hash CHAR(64) NOT NULL,
    imported_by VARCHAR(120) NOT NULL,
    imported_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    item_count INTEGER NOT NULL
        CHECK (item_count >= 0),
    article_count INTEGER NOT NULL
        CHECK (article_count >= 0),
    ativo BOOLEAN NOT NULL DEFAULT FALSE,
    UNIQUE (dataset_code, payload_hash)
);

CREATE TABLE av3.curadoria_import_items_raw (
    id_raw_item SERIAL PRIMARY KEY,
    id_import_run INTEGER NOT NULL
        REFERENCES av3.curadoria_import_runs(id_import_run)
        ON DELETE CASCADE,
    dataset_code VARCHAR(10) NOT NULL,
    question_external_id TEXT NOT NULL,
    question_sequence INTEGER NOT NULL,
    id_pergunta INTEGER
        REFERENCES perguntas(id_pergunta),
    payload_hash CHAR(64) NOT NULL,
    payload_jsonb JSONB NOT NULL
);

CREATE TABLE av3.curadoria_questoes (
    id_curadoria SERIAL PRIMARY KEY,
    id_import_run INTEGER NOT NULL
        REFERENCES av3.curadoria_import_runs(id_import_run)
        ON DELETE CASCADE,
    dataset_code VARCHAR(10) NOT NULL,
    id_pergunta INTEGER NOT NULL
        REFERENCES perguntas(id_pergunta),
    question_external_id TEXT NOT NULL,
    question_sequence INTEGER NOT NULL,
    tipo_questao TEXT NOT NULL,
    prompt_system TEXT,
    questao TEXT NOT NULL,
    gabarito_jsonb JSONB NOT NULL,
    perguntas_jsonb JSONB,
    alternativas_jsonb JSONB,
    pontuacao_total NUMERIC(10,2),
    dificuldade_nivel TEXT,
    dificuldade_escala INTEGER,
    dificuldade_criterios_jsonb JSONB NOT NULL DEFAULT '[]'::jsonb,
    disciplina TEXT,
    assunto TEXT,
    tema TEXT,
    norma TEXT,
    lei TEXT,
    url TEXT,
    urn TEXT,
    curador TEXT,
    dt_classificacao TIMESTAMP,
    metadados_jsonb JSONB NOT NULL DEFAULT '{}'::jsonb,
    raw_payload_jsonb JSONB NOT NULL DEFAULT '{}'::jsonb,
    payload_hash CHAR(64) NOT NULL
);

CREATE TABLE av3.curadoria_artigos (
    id_curadoria_artigo SERIAL PRIMARY KEY,
    id_curadoria INTEGER NOT NULL
        REFERENCES av3.curadoria_questoes(id_curadoria)
        ON DELETE CASCADE,
    ordem INTEGER NOT NULL
        CHECK (ordem >= 1),
    artigo TEXT NOT NULL,
    topico TEXT,
    relevancia TEXT,
    tipo TEXT
);

-- =========================
-- 10. AV3 vector-ready RAG base
-- =========================
CREATE TABLE av3.embedding_model_configs (
    id_embedding_config SERIAL PRIMARY KEY,
    dataset_code VARCHAR(10) NOT NULL UNIQUE,
    dataset_name VARCHAR(80) NOT NULL,
    provider VARCHAR(60) NOT NULL,
    model_name VARCHAR(160) NOT NULL,
    dimensions INTEGER
        CHECK (dimensions IS NULL OR dimensions >= 1),
    api_base_url TEXT,
    notes TEXT,
    updated_by VARCHAR(120) NOT NULL DEFAULT 'system',
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE av3.rag_documents (
    id_document SERIAL PRIMARY KEY,
    id_import_run INTEGER NOT NULL
        REFERENCES av3.curadoria_import_runs(id_import_run)
        ON DELETE CASCADE,
    dataset_code VARCHAR(10) NOT NULL,
    dataset_name VARCHAR(100) NOT NULL,
    document_key CHAR(64) NOT NULL,
    source_name TEXT NOT NULL,
    source_type VARCHAR(40) NOT NULL,
    source_url TEXT,
    title TEXT NOT NULL,
    lei TEXT,
    norma TEXT,
    urn TEXT,
    temporal_reason TEXT,
    inclusion_criteria TEXT,
    metadata_jsonb JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (id_import_run, document_key)
);

CREATE TABLE av3.rag_chunks (
    id_chunk SERIAL PRIMARY KEY,
    id_document INTEGER NOT NULL
        REFERENCES av3.rag_documents(id_document)
        ON DELETE CASCADE,
    id_curadoria INTEGER
        REFERENCES av3.curadoria_questoes(id_curadoria)
        ON DELETE SET NULL,
    id_curadoria_artigo INTEGER
        REFERENCES av3.curadoria_artigos(id_curadoria_artigo)
        ON DELETE SET NULL,
    id_pergunta INTEGER
        REFERENCES perguntas(id_pergunta)
        ON DELETE SET NULL,
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    token_count INTEGER NOT NULL DEFAULT 0,
    chunking_strategy VARCHAR(60) NOT NULL,
    source_kind VARCHAR(40) NOT NULL,
    artigo TEXT,
    topico TEXT,
    relevancia VARCHAR(40),
    tipo VARCHAR(40),
    tema TEXT,
    assunto TEXT,
    metadata_jsonb JSONB NOT NULL DEFAULT '{}'::jsonb,
    content_hash CHAR(64) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (id_document, chunk_index)
);

CREATE TABLE av3.rag_embeddings (
    id_embedding SERIAL PRIMARY KEY,
    id_chunk INTEGER NOT NULL
        REFERENCES av3.rag_chunks(id_chunk)
        ON DELETE CASCADE,
    embedding_model VARCHAR(120) NOT NULL,
    embedding_dimensions INTEGER,
    embedding_vector vector,
    metadata_jsonb JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (id_chunk, embedding_model)
);

CREATE TABLE av3.retrieval_runs (
    id_retrieval_run SERIAL PRIMARY KEY,
    id_import_run INTEGER NOT NULL
        REFERENCES av3.curadoria_import_runs(id_import_run)
        ON DELETE CASCADE,
    dataset_code VARCHAR(10) NOT NULL,
    name VARCHAR(160) NOT NULL,
    retrieval_strategy VARCHAR(60) NOT NULL,
    embedding_model VARCHAR(120),
    top_k INTEGER NOT NULL
        CHECK (top_k >= 1),
    vector_enabled BOOLEAN NOT NULL DEFAULT TRUE,
    lexical_enabled BOOLEAN NOT NULL DEFAULT FALSE,
    rerank_enabled BOOLEAN NOT NULL DEFAULT FALSE,
    ativo BOOLEAN NOT NULL DEFAULT FALSE,
    metadata_jsonb JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- =========================
-- 11. AV3 candidate RAG runs
-- =========================
CREATE TABLE av3.prompt_candidatos (
    id_prompt_candidato SERIAL PRIMARY KEY,
    dataset_code VARCHAR(10) NOT NULL,
    versao INTEGER NOT NULL,
    ds_persona TEXT NOT NULL,
    ds_contexto TEXT NOT NULL,
    ds_instrucao_rag TEXT NOT NULL,
    ds_saida TEXT NOT NULL,
    ativo BOOLEAN NOT NULL DEFAULT FALSE,
    created_by VARCHAR(120) NOT NULL DEFAULT 'system',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (dataset_code, versao)
);

CREATE TABLE av3.candidate_runs (
    id_candidate_run SERIAL PRIMARY KEY,
    dataset_code VARCHAR(10) NOT NULL,
    id_retrieval_run INTEGER NOT NULL
        REFERENCES av3.retrieval_runs(id_retrieval_run),
    id_prompt_candidato INTEGER NOT NULL
        REFERENCES av3.prompt_candidatos(id_prompt_candidato),
    model_name VARCHAR(160) NOT NULL,
    provider VARCHAR(80) NOT NULL,
    temperature NUMERIC(5,3),
    max_tokens INTEGER,
    top_p NUMERIC(5,3),
    batch_size INTEGER NOT NULL
        CHECK (batch_size >= 1),
    run_status VARCHAR(30) NOT NULL DEFAULT 'created'
        CHECK (run_status IN ('created', 'running', 'completed', 'failed', 'cancelled')),
    started_at TIMESTAMP,
    finished_at TIMESTAMP,
    created_by VARCHAR(120) NOT NULL DEFAULT 'system',
    metadata_jsonb JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE av3.candidate_answers (
    id_candidate_answer SERIAL PRIMARY KEY,
    id_candidate_run INTEGER NOT NULL
        REFERENCES av3.candidate_runs(id_candidate_run)
        ON DELETE CASCADE,
    id_pergunta INTEGER NOT NULL
        REFERENCES perguntas(id_pergunta),
    model_name VARCHAR(160) NOT NULL,
    answer_text TEXT,
    final_choice VARCHAR(10),
    rendered_prompt TEXT NOT NULL,
    status VARCHAR(30) NOT NULL DEFAULT 'created'
        CHECK (status IN ('created', 'running', 'success', 'failed', 'skipped')),
    error_message TEXT,
    latency_ms INTEGER
        CHECK (latency_ms IS NULL OR latency_ms >= 0),
    raw_response_jsonb JSONB,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (id_candidate_run, id_pergunta)
);

CREATE TABLE av3.candidate_answer_context_chunks (
    id_answer_context_chunk SERIAL PRIMARY KEY,
    id_candidate_answer INTEGER NOT NULL
        REFERENCES av3.candidate_answers(id_candidate_answer)
        ON DELETE CASCADE,
    id_chunk INTEGER NOT NULL
        REFERENCES av3.rag_chunks(id_chunk),
    rank INTEGER NOT NULL
        CHECK (rank >= 1),
    similarity_score NUMERIC(10,6),
    chunk_text_snapshot TEXT NOT NULL,
    source_url TEXT,
    metadata_jsonb JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (id_candidate_answer, rank),
    UNIQUE (id_candidate_answer, id_chunk)
);

-- =========================
-- Indexes
-- =========================

-- Questions by dataset.
CREATE INDEX idx_perguntas_dataset
ON perguntas(id_dataset);

-- Answers by question.
CREATE INDEX idx_respostas_pergunta
ON respostas_atividade_1(id_pergunta);

-- Answers by candidate model.
CREATE INDEX idx_respostas_modelo
ON respostas_atividade_1(id_modelo);

-- Active prompt version per dataset.
CREATE UNIQUE INDEX idx_prompt_juizes_active_per_dataset
ON prompt_juizes(id_dataset)
WHERE ativo;

-- Evaluations by answer.
CREATE INDEX idx_avaliacoes_resposta
ON avaliacoes_juiz(id_resposta_ativa1);

-- Evaluations by judge model.
CREATE INDEX idx_avaliacoes_juiz
ON avaliacoes_juiz(id_modelo_juiz);

-- Meta-evaluations by evaluated row.
CREATE INDEX idx_meta_avaliacoes_avaliacao
ON meta_avaliacoes(id_avaliacao);

-- Active RAG curation import per dataset.
CREATE UNIQUE INDEX idx_curadoria_import_runs_active_dataset
ON av3.curadoria_import_runs(dataset_code)
WHERE ativo;

-- Raw imported items by run.
CREATE INDEX idx_curadoria_import_items_run
ON av3.curadoria_import_items_raw(id_import_run);

-- Normalized curated questions by dataset and run.
CREATE INDEX idx_curadoria_questoes_dataset
ON av3.curadoria_questoes(dataset_code, id_import_run, question_sequence);

-- Curated articles by curation entry.
CREATE INDEX idx_curadoria_artigos_curadoria
ON av3.curadoria_artigos(id_curadoria, ordem);

-- Active retrieval run per dataset.
CREATE UNIQUE INDEX idx_retrieval_runs_active_dataset
ON av3.retrieval_runs(dataset_code)
WHERE ativo;

-- Materialized RAG documents by import run and dataset.
CREATE INDEX idx_rag_documents_import_dataset
ON av3.rag_documents(id_import_run, dataset_code);

-- Materialized chunks by document and order.
CREATE INDEX idx_rag_chunks_document
ON av3.rag_chunks(id_document, chunk_index);

-- Materialized chunks by related question.
CREATE INDEX idx_rag_chunks_question
ON av3.rag_chunks(id_pergunta, source_kind);

-- Embeddings lookup by chunk and model.
CREATE INDEX idx_rag_embeddings_chunk_model
ON av3.rag_embeddings(id_chunk, embedding_model);

-- Embedding model configuration history by dataset.
CREATE INDEX idx_embedding_model_configs_dataset
ON av3.embedding_model_configs(dataset_code, updated_at DESC);

-- Retrieval runs by source import and dataset.
CREATE INDEX idx_retrieval_runs_import_dataset
ON av3.retrieval_runs(id_import_run, dataset_code, created_at DESC);

-- Active candidate prompt version per dataset.
CREATE UNIQUE INDEX idx_prompt_candidatos_active_dataset
ON av3.prompt_candidatos(dataset_code)
WHERE ativo;

-- Candidate runs by dataset and recency.
CREATE INDEX idx_candidate_runs_dataset_created
ON av3.candidate_runs(dataset_code, created_at DESC);

-- Candidate answers by run and status.
CREATE INDEX idx_candidate_answers_run_status
ON av3.candidate_answers(id_candidate_run, status);

-- Context chunk snapshots by answer and rank.
CREATE INDEX idx_candidate_answer_context_chunks_answer_rank
ON av3.candidate_answer_context_chunks(id_candidate_answer, rank);
