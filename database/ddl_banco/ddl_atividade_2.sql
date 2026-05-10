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
