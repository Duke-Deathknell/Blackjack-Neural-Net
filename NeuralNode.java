package nnblackjack;

import java.util.Random;

/**Michael Alsbergas, 5104112
 * Blackjack Neural Network, Cosc 4P67
 */
public class NeuralNode {
    
    double[] input;
    double[] weight;
    double[] momentum;
    double output;
    double error;
    int[] prev;
    
    double biasWeight;
    double biasMoment;
    double biasPrev = 1;
    
    //Initialize a node
    public NeuralNode (int nInput, double moment, int seed){
        Random gen = new Random(2016 + seed*7);
        input = new double[nInput];
        weight = new double[nInput];
        momentum = new double[nInput];
        prev = new int[nInput];
        
        for (int i = 0; i < nInput; i++){
            weight[i] = gen.nextDouble() - 0.5;
            momentum[i] = moment;
            prev[i] = 1;
        }
    }
    
    //Take input and produce output for the node
    public void passInput (double[] pass){
        double calc = biasWeight;
        for (int i = 0; i < pass.length ; i++){
            input[i] = pass[i];
        }
         
        for (int i = 0; i < input.length; i++){
            calc = calc + input[i]*weight[i];
        }
        
        output = 1.0 / (1.0 + Math.exp(calc * -1.0));
    }
    
    //Update the weight of the node
    public void update(double learn){
        double delta; 
        
        delta = output * (1 - output) * learn * error; //output = f(x)
                
        for (int i =0; i < input.length; i++){
            weight[i] = weight[i] + delta * input[i] * momentum[i];
            
            if (error * prev[i] > 0){
                momentum[i] = momentum[i] * 2;
            }
            else {
                momentum[i] = momentum[i] / 2;
                prev[i] = prev[i] * -1;
            }
        }
        
        biasWeight = biasWeight + delta * biasMoment;
            
        if (delta * biasPrev > 0){
            biasMoment = biasMoment *2;
        }
        else {
           biasMoment = biasMoment /2;
           biasPrev = biasPrev * -1;
       }
    }
}
