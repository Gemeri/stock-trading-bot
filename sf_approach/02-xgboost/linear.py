import numpy as np
import math
from sklearn.linear_model import LinearRegression

def stock_price_direction(time_indices:list[int], stock_price_predictions:list[float]) -> float:
    
    # Example stock price predictions
    y = np.array(stock_price_predictions)
    x = np.array(time_indices).reshape(-1, 1)  # Time indices

    # Fit linear regression
    model = LinearRegression()
    model.fit(x, y)

    slope = model.coef_[0]

    # Compute angle in degrees
    angle_rad = math.atan(slope)
    angle_deg = math.degrees(angle_rad)

    return angle_deg
