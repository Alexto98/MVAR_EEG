# MVAR_EEG
### Connectivity Measures of EEG signals using a Multivariate Autoregressive Model (MVAR)

The goal of this project is to understand how it is possible to quantify couplings between EEG time-series.
The quantification can be performed in both time and frequency domain.
The easier approach in time domain is to evaluate the correlation between time-series. Doing that, you are 
understanding if two time-series are connected; and so, if the brain regions that produce those time-series are 
working together. This is called functional connectivity. However, one important information missed is the 
causality of this connection, so which time series is influencing the other. If you understand that, you can better 
understand the flow of information between different brain regions. This is called effective connectivity.

In time domain, a study of effective connectivity of time-series is based on an important class of linear time-independent discrete models, called multivariate autoregressive models (MVAR). Those models are a generalization of the simpler autoregressive model (AR).

In this project, I will implement from scratch a MVAR model, fitting it on a 9 channels acquisition of 5 
seconds with sampling frequency of 128 Hz. The model will try to predict the value of all channels at a time 
point N, based on a linear combination of the p previous samples of the that channel and all the others. 
Changing the value of p will influence the prediction of the model. As a result, I will select the value of p (that 
is called model order) that leads to the best prediction by applying two different techniques of model selection: 
one based on a two-fold cross validation, and one based on statistical criteria like AIC and BIC.
