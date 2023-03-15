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

The aim of this work is to perform effective connectivity on the EEG signal, therefore try to
understand which signals are piloting the others and so which are the flows of information. As a result, I need some metrics that are able to evaluate causality of the different connections.
Moreover, I would like to perform the analysis in the frequency domain, since usually you can retrieve useful information in this domain, for example regarding brain rhythms. Such metrics are based on the Fourier transform of the coefficients of a MVAR model, that is a model able to give you causality information in the time domain (for example the Granger causality).
That’s why I fitted a MVAR model to the data in the previous stage.


More details in the pdf files named "Assignment_1_Alessio_Tonello.pdf" and "Assignment_2_Alessio_Tonello.pdf".

MATLAB code in "Assignment_2_Alessio_Tonello.m".
