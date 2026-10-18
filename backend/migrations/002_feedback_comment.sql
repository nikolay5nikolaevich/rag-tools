-- Отзыв с текстом (T-22, CONTRACTS.md §5). Поле nullable и добавляется отдельной
-- миграцией: verdict остаётся в `feedback`, а комментарий необязателен и пишется
-- той же записью журнала, по её id из meta.query_id.

ALTER TABLE query_log ADD COLUMN IF NOT EXISTS feedback_comment text;
