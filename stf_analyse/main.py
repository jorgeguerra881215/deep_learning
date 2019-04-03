import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings('ignore')




if __name__ == '__main__':
    data = pd.read_csv('data/data_all_result.txt', sep = ' ')
    data = data[['State', 'Label']]
    data = data[data['State'].str.len() > 3]
    data = data.replace(['Normal','Botnet'], [0,1])

    # How much attack are
    f, ax = plt.subplots(1, 2, figsize=(18, 8))
    data['Label'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[0], shadow=True)
    ax[0].set_title('Attack')
    ax[0].set_ylabel('')
    sns.countplot('Label', data=data, ax=ax[1])
    ax[1].set_title('Attack')
    plt.show()

    #print('DONE')