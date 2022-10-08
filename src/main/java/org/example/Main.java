package org.example;

import java.io.File;
import java.io.IOException;

public class Main {
    public static void main(String[] args) throws IOException {
        // apply the XOR training set
        double[][] binaryInputLayerValue = new double[][] {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        double[][] binaryExpectedOutput = new double[][] {{0},{1},{1},{0}};

        double[][] bipolarInputLayerValue = new double[][] {{-1, -1}, {-1, 1}, {1, -1}, {1, 1}};
        double[][] bipolarExpectedOutput = new double[][] {{-1}, {1}, {1}, {-1}};

        // initialize totalEpoch, times, avgEpoch
        int totalEpoch = 0;
        int times = 100;
        int avgEpoch = 0;

        // create a binary NeuralNet
        //NeuralNet binary = new NeuralNet(2, 4, 1,0.2, 0.9, -0.5, 0.5,0, 1, binaryInputLayerValue, binaryExpectedOutput);

        // create a bipolar NeuralNet
        NeuralNet bipolar = new NeuralNet(2, 4, 1,0.2, 0.9, -0.5, 0.5,-1, 1, bipolarInputLayerValue, bipolarExpectedOutput);
        for(int i = 0; i < times; i++) {
            //totalEpoch += binary.trainTest();
            totalEpoch += bipolar.trainTest();
        }
        avgEpoch = totalEpoch/times;
        System.out.println("\nOn average " + avgEpoch + " epochs is taken to reach a total error of less than 0.05");


        // save and load the file
        File newFile = new File("weights.txt");
//        binary.save(newFile);
//        binary.load("weights.txt");

        bipolar.save(newFile);
        bipolar.load("weights.txt");

        System.out.println("Program successfully finished.");

    }
}