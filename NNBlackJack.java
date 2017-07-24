package nnblackjack;

/**Michael Alsbergas, 5104112
 * Blackjack Neural Network, Cosc 4P67
 */

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.Random;

public class NNBlackJack {
    
    Random gen; 
    NeuralNode[][] net;
    double learn = 0.1;
    double errorSum;
    
    int num = 3; // number of hidden nodes   

    public NNBlackJack(){
        int dealer1, dealer2, player, soft, dSoft;
        int wins, draws, z = 0;
        boolean hit;
        PrintWriter report1, report2, report3, report4, report5;
        
        //Init the network
        net = new NeuralNode[2][];
        net[0] = new NeuralNode[num]; //num hidden nodes
        net[1] = new NeuralNode[1]; //1 output node
        
        for (int i =0; i < num; i++){
            net[0][i] = new NeuralNode(3, 0.1, i); //3 inputs, 0.1 intital momentum 
        }
        net[1][0] = new NeuralNode(num, 0.1, 20);
        
        //Create report file
        try {
                report1 = new PrintWriter("0Wins.txt");
                report2 = new PrintWriter("0Draws.txt");
                report3 = new PrintWriter("0Choices.txt");
                report4 = new PrintWriter("0%Wins.txt");
                report5 = new PrintWriter("0%Draws.txt");
            } catch (FileNotFoundException ex) {
                System.out.println("ERROR!!");
                report1 = null;
                report2 = null;
                report3 = null;
                report4 = null;
                report5 = null;
            }
        
        //Start
        for (int x = 0; x < 500; x++){
            wins = 0;
            draws = 0;
            errorSum = 0;
            for (int y = 0; y < 500; y++){
                hit = true;
                soft = 0;
                dSoft = 0;
                gen = new Random(y); 
                
                int temp = drawCard();
                player = temp; 
                if (temp == 1){ // 1 is Ace
                    soft = 1;
                    player = player + 10;
                }
                
                temp = drawCard();
                dealer1 = temp; 
                if (temp == 1){ // 1 is Ace
                    dSoft = 1;
                    dealer1 = dealer1 + 10;
                }
                
                temp = drawCard(); 
                player = player + temp; 
                if (temp == 1){ // 1 is Ace
                    if (soft != 1){ player = player + 10; } //if previous card isn't Ace
                    soft = 1;
                }
                
                dealer2 = drawCard(); //Dealer's face down card
                
                //Hit or Stay? 
                while (hit){
                    //pass forward
                    hit = decide(dealer1, player, soft);
                    
                    if (hit){
                        report3.println("0");
                        
                        temp = drawCard(); 
                        player = player + temp; 
                        if (player > 21 && soft == 1){ //Ace = 1 instead of 11
                            player = player - 10;
                            soft = 0;
                        }
                        else if (player > 21) { //Bust
                            hit = false;
                            player = 0; 
                        }
                        else if (temp == 1 && player + 10 <= 21){ //Got an Ace
                            player = player + 10; 
                            soft = 1;
                        }
                    }
                    else { report3.println("1"); }
                } //Player turn ends
                
                //Dealer Turn 
                dealer1 = dealer1 + dealer2; 
                
                if (dealer2 == 1 && dSoft == 0){
                    dealer1 = dealer1 + 10;
                    dSoft = 1;
                }
                
                while (dealer1 < 17){
                    dealer2 = drawCard();
                    dealer1 = dealer1 + dealer2;
                    
                    if (dealer1 > 21 && dSoft == 1){ //Ace = 1 instead of 11
                        dealer1 = dealer1 - 10;
                        dSoft = 0;
                    }
                    else if (dealer1 > 21) { //Bust
                        dealer1 = 0; 
                        break;
                    }
                    else if (dealer2 == 1 && dealer1 + 10 <= 21){ //Got an Ace
                        dealer1 = dealer1 + 10; 
                        dSoft = 1;
                    }
                }//End of Dealer turn
                
                //Winner
                if (player > dealer1){ wins = wins + 1; }
                else if (player == dealer1){ draws = draws + 1; }
                
                //Train?
                if (y >= (1*x) && y < (1*x +10)){
                    if (player > dealer1){
                        //train on win!
                        if (net[1][0].output < 0.5){
                            errorSum = errorSum + (0 - net[1][0].output);
                        }
                        else {
                            errorSum = errorSum + (1 - net[1][0].output);
                        }
                    }
                    else{
                        //train on loss
                        if (net[1][0].output < 0.5){
                            errorSum = errorSum + (1 - net[1][0].output);
                        }
                        else {
                            errorSum = errorSum + (0 - net[1][0].output);
                        }
                    }
                }
                train();
            }
            //Print Reports
            System.out.println(wins + "  X  " + draws);            
            report1.println(wins);
            report2.println(draws);       
            double temp = wins / 5.0;
            report4.println(temp);
            temp = draws / 5.0;
            report5.println(temp);
            
            learn = learn * (1-learn);
            if (x == 499 && z < 50) { x = 0;  z++; } //Repeat 50 times
        }
        report1.close();
        report2.close();
        report3.close();
        report4.close();
        report5.close();
    }
    
    //Forward pass of inputs through network
    private boolean decide(int d1, int val, int soft){
        double[] input;
        double[] pass;
        input = new double [3];
        pass = new double [num];
        
        input[0] = d1;
        input[1] = val;
        input[2] = soft;
        
        for (int i =0; i < num; i++){
            net[0][i].passInput(input);
            pass[i] = net[0][i].output;
        }
        net[1][0].passInput(pass);
        
        //0 for hit, 1 for stay
        if (net[1][0].output < 0.5){  return true; }
        else {  return false; }
                
    }
    
    //Train network based on sum of errors
    private void train(){
        net[1][0].error = errorSum;
        
        for (int i = 0; i < num; i++){
            net[0][i].error = net[1][0].error * net[1][0].weight[i];
            net[0][i].update(learn);
        }
        net[1][0].update(learn);
        
    }
    
    //Draw a card from the deck, return its value
    private int drawCard (){
        int val; 
        
        val = gen.nextInt(13) + 1; //1 to 13
        
        if (val > 10) { val = 10; } //face card
        
        return val;
    }
    
    public static void main(String[] args) {
        NNBlackJack b = new NNBlackJack(); // TODO code application logic here
    }
}
