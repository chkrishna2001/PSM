copy (
  with
  q as (
    select 'commonsense' as subset, * from read_parquet('nano-psm/data-pipeline/data/raw/realtalk-mteb/commonsense-queries/test-00000-of-00001.parquet')
    union all select 'multi_hop' as subset, * from read_parquet('nano-psm/data-pipeline/data/raw/realtalk-mteb/multi_hop-queries/test-00000-of-00001.parquet')
    union all select 'temporal_reasoning' as subset, * from read_parquet('nano-psm/data-pipeline/data/raw/realtalk-mteb/temporal_reasoning-queries/test-00000-of-00001.parquet')
  ),
  c as (
    select 'commonsense' as subset, * from read_parquet('nano-psm/data-pipeline/data/raw/realtalk-mteb/commonsense-corpus/test-00000-of-00001.parquet')
    union all select 'multi_hop' as subset, * from read_parquet('nano-psm/data-pipeline/data/raw/realtalk-mteb/multi_hop-corpus/test-00000-of-00001.parquet')
    union all select 'temporal_reasoning' as subset, * from read_parquet('nano-psm/data-pipeline/data/raw/realtalk-mteb/temporal_reasoning-corpus/test-00000-of-00001.parquet')
  ),
  qr as (
    select 'commonsense' as subset, "query-id" as query_id, "corpus-id" as corpus_id, score from read_parquet('nano-psm/data-pipeline/data/raw/realtalk-mteb/commonsense-qrels/test-00000-of-00001.parquet')
    union all select 'multi_hop' as subset, "query-id" as query_id, "corpus-id" as corpus_id, score from read_parquet('nano-psm/data-pipeline/data/raw/realtalk-mteb/multi_hop-qrels/test-00000-of-00001.parquet')
    union all select 'temporal_reasoning' as subset, "query-id" as query_id, "corpus-id" as corpus_id, score from read_parquet('nano-psm/data-pipeline/data/raw/realtalk-mteb/temporal_reasoning-qrels/test-00000-of-00001.parquet')
  )
  select
    q.subset,
    q.id as query_id,
    q.text as query,
    c.id as positive_id,
    c.text as positive,
    null::varchar as negative_id,
    null::varchar as negative
  from q
  join qr on q.subset = qr.subset and q.id = qr.query_id
  join c on c.subset = qr.subset and c.id = qr.corpus_id
  limit 1200
) to 'nano-psm/data-pipeline/data/raw/realtalk-mteb/realtalk-training.jsonl' (format json);
