import numpy as np

r = 4

circleWidth = (2 * r) + 1
[X, Y] = np.meshgrid(np.arange(1, (circleWidth + 1)), np.arange(1, (circleWidth + 1)))
distFromCenter = np.sqrt(np.power(X - (r + 1), 2) + np.power(Y - (r + 1), 2))
onPixels = np.abs(distFromCenter - r) < 0.5
print(onPixels)