# Day 3: Testing Market Efficiency with Linear Regression

##  Hypothesis
Can past stock returns predict future returns? According to the Efficient Markets Hypothesis, they shouldn't.

## Experiment Design
- **Data:** Apple stock (AAPL), 2018-2023
- **Features:** Previous 5 days of returns
- **Target:** Tomorrow's return
- **Model:** Linear Regression with Gradient Descent (my implementation)
- **Split:** 80% train (2018-2022), 20% test (2023)

##  Results
Train/Test Split
------------------------------
Training period: 2018-01-10 to 2022-10-17
Testing period: 2022-10-18 to 2023-12-28
Training samples: 1,201
Testing samples: 301

 MODEL PERFORMANCE
==============================
R² Score: -0.000009
Mean Squared Error:0.000246
Noise threshold (2/√n): 0.115278
Is R² significant? False 
Conclusion: R² is not statistically significant



### Visual Evidence
![Market Efficiency Results](image.png)
*Note: The scatter plot shows random noise, no discernible pattern*

###  Interpretation
My linear regression model achieved an R² score of -0.000009, which is effectively zero. This means:
- The model explains **0%** of the variance in stock returns
- Past 5 days of returns have **no predictive power** for tomorrow's return
- The negative R² (slightly below zero) indicates the model performs worse than simply predicting the average return

###  Statistical Significance
With a noise threshold of 0.115, my R² of -0.000009 is **not statistically significant**. Any apparent pattern is indistinguishable from random noise.

###  Key Insight for Quant Finance
This experiment confirms a fundamental truth: **simple linear models on raw returns cannot predict stock prices**. Markets appear efficient at this basic level. Real quant strategies succeed by:
1. Using more sophisticated features (order book data, alternative data)
2. Capturing non-linear relationships
3. Focusing on higher frequencies (microseconds, not days)
4. Managing risk rather than just predicting direction

###  What I Learned Today
- ✅ Returns are mostly random at daily frequency
- ✅ Linear regression alone won't make you a quant
- ✅ Statistical significance matters more than raw numbers
- ✅ Negative R² is possible (model worse than guessing mean)

### 🔗 Connect to Tomorrow
If simple models fail, what's next?
- Classification models: predict "up" vs "down" instead of exact returns
- Feature engineering: create technical indicators (RSI, MACD)
- Alternative data: volume, volatility, market sentiment

###  Raw Output
<img width="866" height="391" alt="output1" src="https://github.com/user-attachments/assets/a4d6c273-fa48-4268-8c03-c714f21b7a23" />
<img width="1385" height="989" alt="output" src="https://github.com/user-attachments/assets/5e08347e-8209-4e7d-9c5a-eb70d6607b95" />


