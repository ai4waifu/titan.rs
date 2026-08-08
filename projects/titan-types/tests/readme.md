# Grammar Reference

```bash
wee test
```

## CRUD

### Create

```vql
table_1.insert {
    col_a = 0
    col_b = CURRENT_TIMESTAMP
}
```

```sql
INSERT INTO table_1 (col_a, col_b)
VALUES (0, CURRENT_TIMESTAMP);
```

### Read

### List

### Update

```vql
table_1.update(id) {
    col_a = 0
    col_b = CURRENT_TIMESTAMP
}
```

```sql
UPDATE table_1
SET col_a = 0,
    col_b = CURRENT_TIMESTAMP
WHERE id = :id;
```

```vql
table_1.update(id) {
    if a > 0 && b < 0 {
        money = 0
        update_time = CURRENT_TIMESTAMP
    }
    else {
        money = -1
        update_time = CURRENT_TIMESTAMP
    }
}
```

```sql
UPDATE table_1
SET money       = CASE
                      WHEN a > 0 AND b < 0 THEN 0
                      ELSE -1
    END,
    update_time = CURRENT_TIMESTAMP
WHERE id = :id;
```

### Delete

```sql
DELETE
FROM table_1
WHERE id = :id;
```