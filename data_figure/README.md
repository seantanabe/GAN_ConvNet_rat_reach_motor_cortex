reach_neural_activity.mat

y_pred_allall: neural activity predicted by ConvNet model. 
y_true_allall: neural activity observed from rats. 
Each have 55 cells corresponding to rats and training days identified by rr_day (1st column: rat id, 2nd column: trainig day).
Each rat/day cell has number of cells corresponding to the neurons. 
And each neuron cell has 21x10.
1st dimension (21) corresponding to time -400ms~600ms with 20ms increments (indentified by t_ms).
2nd dimension corresponds to the 10 cross-validation test neural activity. 


fig_2C_dt.mat

Distance data for figure 2C. 
Row cells: early, mid, late days of training. 
Column: observed, shuffled. 

