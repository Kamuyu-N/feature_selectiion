# Machine Learning Trading Bot V1

### Feature creation:
- All Technical indicators present in the talib library are used to create the various features and have various lengths per indicator i.e( SMA 200, SMA 50,  SMA 21)
- Creation of Heiken Ashi candles used to smooth out price movement and illustrate trend strength and direction

### Feature Selection
-

### Model Optimization 

    
A machine learning-based trading bot that predicts market movements using XGBoost, Random Forest, and Gradient Boosting. The model is optimized to maximize recall for two specific classes related to trading signals.

## Features  
- Multi-class classification for trading decisions  
- Feature selection using RFECV and correlation filtering  
- Hyperparameter tuning with Bayesian Optimization and Optuna  
- Optimized recall for key trading classes  
- Stacking ensemble approach for improved predictions  

## Installation  

### Prerequisites  
Make sure you have Python 3.8+ installed along with the following dependencies:  
- `scikit-learn`  
- `xgboost`  
- `lightgbm`  
- `pandas`  
- `numpy`  
- `optuna`  

### Setup  
Clone the repository and install dependencies:  

```bash
git clone https://github.com/yourusername/trading-bot.git
cd trading-bot
pip install -r requirements.txt
