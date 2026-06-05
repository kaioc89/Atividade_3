SELECT 'public.respostas_atividade_1' AS table_name, COUNT(*)::bigint AS row_count
FROM public.respostas_atividade_1
UNION ALL
SELECT 'public.avaliacoes_juiz' AS table_name, COUNT(*)::bigint AS row_count
FROM public.avaliacoes_juiz
UNION ALL
SELECT 'av3.rag_chunks' AS table_name, COUNT(*)::bigint AS row_count
FROM av3.rag_chunks
UNION ALL
SELECT 'av3.rag_embeddings' AS table_name, COUNT(*)::bigint AS row_count
FROM av3.rag_embeddings
UNION ALL
SELECT 'av3.retrieval_runs.ativo=true' AS table_name, COUNT(*)::bigint AS row_count
FROM av3.retrieval_runs
WHERE ativo = TRUE;
