# %% Importing Data
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# %% Confusion Matrix
actual = np.array(['dog', 'dog', 'dog', 'not dog', 'dog', 'not dog', 'dog', 'dog', 'not dog', 'not dog'])
predict = np.array(['dog', 'not dog', 'dog', 'not dog', 'dog', 'dog', 'dog', 'dog', 'not dog', 'not dog'])

#%% Plotting Confusion Matrix
w = confusion_matrix(actual, predict)
labels = ['dog', 'not dog']
sns.heatmap(w,
            annot=True,
            fmt='g',
            cmap='Oranges',
            xticklabels=labels,
            yticklabels=labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
# %%
