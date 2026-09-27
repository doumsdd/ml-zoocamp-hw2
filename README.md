# ML Zoomcamp 2026 - Homework 2: Regression

This repository contains the solution for Week 2 of the [Machine Learning Zoomcamp 2026](https://github.com/DataTalksClub/machine-learning-zoomcamp).

## Task

Build a linear regression model to predict car fuel efficiency (`fuel_efficiency_mpg`) using the 2026 Car Fuel Efficiency dataset.

## Files

- `hw2_solution.py` - Python script with the complete solution (Q1–Q6)
- `car_fuel_efficiency_2026.csv` - Dataset used for the homework

## Questions

1. Identify the column with missing values
2. Compute the median of `horsepower`
3. Compare imputation strategies (fill with 0 vs mean)
4. Find the best regularization parameter `r`
5. Evaluate model stability across different random seeds
6. Final test RMSE with combined train+val set

## Answers

| Question | Answer |
|----------|--------|
| Q1 | `horsepower` |
| Q2 | `254` |
| Q3 | With mean |
| Q4 | `0` |
| Q5 | `0.029` |
| Q6 | `2.236` |

## How to run

```bash
pip install pandas numpy
python hw2_solution.py
```
