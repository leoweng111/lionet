# lionet is a Quantitative **futures factor-generating and backtesting framework**

## Structure

### 1. daily

Files for scheduled futures update jobs.

### 2. data

API for querying and updating futures data.

### 3. error

Error classes.

### 4. factors

Methods for factor processing, backtesting, and factor generation.

### 5. models

Machine learning models for signal generation.

### 6. mongo

MongoDB configuration and operation API.

### 7. stats

Statistic methods.

### 8. test

Basic tests.

### 9. notebook

Guidance notebooks.

### 10. quick checks

Run the typed GP smoke test with derived leaf features:

```bash
python -u test/typed_gp_smoke.py
```
