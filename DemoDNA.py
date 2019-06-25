from NFS import *
from numpy import genfromtxt
from sklearn.svm import LinearSVC
from sklearn import metrics
from pandas import read_csv
from sklearn.model_selection import train_test_split

#%% Test on UCI gene expression cancer RNA-Seq Data Set 
print("Importing RNA-Seq data...")
X_data = genfromtxt('./data/data.csv', delimiter=',',skip_header=1)
X_data=X_data[:,1:]
X_labels=read_csv('./data/data.csv',nrows=1)
X_labels=X_labels.columns[1:]
labels=read_csv('./data/labels.csv',header=0)
y_data=labels['Class']=='BRCA' # Check for BRCA labels
y_data=1.0*y_data.to_numpy()

# Split training / test set
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.33, random_state=42)

# Test Naive Feature Selection, followed by l2 SVM
k=100 # Target number of features
nfs_res=nfs(X_train,y_train,k)
clfsv = LinearSVC(random_state=0, tol=1e-5)
clfsv.fit(X_train[:,nfs_res['idx']], y_train)
y_pred_NFS = clfsv.predict(X_test[:,nfs_res['idx']])
score_nfs = metrics.accuracy_score(y_test==1, y_pred_NFS)
print("NFS accuracy:\t%0.3f" % score_nfs)

print('Positive genes:')
print([X_labels[nfs_res['idx'][i]] for i in range(100) if clfsv.coef_[0][i]>=0])


#%% Plot sparsity / likelihood tradeoff
import matplotlib.pyplot as plt
kmax=5000
xvals=range(5,kmax,100)
resv=np.zeros(len(xvals))
for i in range(0,len(xvals)):
    k=xvals[i]
    nfs_res=nfs(X_train,y_train,k)
    resv[i]=nfs_res['objv']

plt.style.use('seaborn-white')
plt.plot(xvals, resv,'-b',linewidth=4)
plt.xlabel('Sparsity k',fontsize=14)
plt.ylabel('Likelihood',fontsize=14)
plt.savefig('tradeoff.pdf')
plt.show()


#%% Plot curve
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode
import plotly.io as pio

trace = go.Scatter(
    x=list(xvals),
    y=resv,
    line = dict(
        color = ('rgb(22, 96, 167)'),
        width = 4)
)
data = [trace]
layout = go.Layout(
    xaxis=go.layout.XAxis(
        title=go.layout.xaxis.Title(
            text='<b>Sparsity k',
            font=dict(
                family='Courier New, monospace',
                size=18,
                color='#5f5f5f',
            )
        )
    ),
    yaxis=go.layout.YAxis(
        title=go.layout.yaxis.Title(
            text='<b>Likelihood<br><br>',
            font=dict(
                family='Courier New, monospace',
                size=18,
                color='#5f5f5f'
            )
        )
    )
)
fig = go.Figure(data=data, layout=layout)
pio.write_image(fig, 'tradeoff.pdf')
#py.iplot(data, layout=layout, filename='basic-line')
#py.plot(data, filename = 'basic-line', auto_open=True)

#%%
