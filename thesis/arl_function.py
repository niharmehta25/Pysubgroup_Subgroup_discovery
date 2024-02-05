import numpy as np

def calculate_arl_score( y, y_prob,b):
      
        sorted_indices = np.argsort(y)
        sorted_true = y[sorted_indices]
        sorted_ranking = y_prob[sorted_indices]
    
        true_indices = np.where(sorted_true == 1)[0]
        false_indices = np.where(sorted_true == 0)[0]
    
        false_rankings = sorted_ranking[false_indices]
        true_rankings = sorted_ranking[true_indices]
    
     
        false_rankings_broadcasted = np.expand_dims(false_rankings, axis=1)
        true_rankings_broadcasted = np.expand_dims(true_rankings, axis=0)
    
        higher_rankings = false_rankings_broadcasted > true_rankings_broadcasted
        equal_rankings = false_rankings_broadcasted == true_rankings_broadcasted
    
        PENNi_sum = np.sum(higher_rankings, axis=1) + 0.5 * np.sum(equal_rankings, axis=1)
    
        
        sorted_true_true_indices = sorted_true[true_indices][:, np.newaxis]
    
       
        numerator_sum = np.sum(sorted_true_true_indices * PENNi_sum)
        denominator_sum = np.sum(y)

    
        if denominator_sum == 0 or np.isnan(denominator_sum):
            arl_score = np.nan
        else:
            arl_score = (numerator_sum / denominator_sum) *b
        return arl_score


