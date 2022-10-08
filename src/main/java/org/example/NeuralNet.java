package org.example;

import java.io.*;

public class NeuralNet implements NeuralNetInterface {
    final double acceptError = 0.05;

    private int argNumInputs; // The number of inputs in your input vector
    private int argNumHidden; // The number of hidden neurons in your hidden layer. Only a single hidden layer is supported
    private int argNumOutput; // The number of output neuron in your output layer.
    private double argLearningRate; // The learning rate coefficient
    private double argMomentumTerm; // The momentum coefficient
    private double argA; // Integer lower bound of sigmoid used by the output neuron only.
    private double argB; // Integer upper bound of sigmoid used by the output neuron only.
    private boolean isBipolar; // indicate whether the NeuralNet is binary or bipolar

    private double[][] weightsFromInToHi; // input-to-hidden weight
    private double[][] weightsFromHiToOut; // hidden-to-output weight

    private double[][] inputLayerValue; // input layer matrix
    private double[] hiddenLayerValue; // hidden layer matrix
    private double[] outputLayerValue; // output layer matrix
    private double[][] expectedOutput; // expected output matrix

    private double[][] weightChgInToHi; // input-to-hidden weight change
    private double[][] weightChgHiToOut; // hidden-to-output weight change

    private double weightsRangeLower; // equal to -0.5
    private double weightsRangeUpper; // equal to +0.5

    private double[] errorSigHidden; // hidden error signals
    private double[] errorSigOutput; // output error signals



    public NeuralNet (int argNumInputs, int argNumHidden, int argNumOutput, double argLearningRate, double argMomentumTerm, double weightsRangeLower, double weightsRangeUpper, double argA, double argB, double[][] inputLayerValue, double[][] expectedOutput) {
        this.argNumInputs = argNumInputs;
        this.argNumHidden = argNumHidden;
        this.argNumOutput = argNumOutput;
        this.argLearningRate = argLearningRate;
        this.argMomentumTerm = argMomentumTerm;
        this.argA = argA;
        this.argB = argB;
        this.weightsRangeLower = weightsRangeLower;
        this.weightsRangeUpper = weightsRangeUpper;
        this.inputLayerValue = inputLayerValue;
        this.expectedOutput = expectedOutput;

        if(argA == -1 && argB == 1) {
            isBipolar = true;
        }

        if (argA == 0 && argB == 1){
            isBipolar = false;
        }

        weightsFromInToHi = new double[argNumInputs+1][argNumHidden];
        weightsFromHiToOut = new double[argNumHidden+1][argNumOutput];

        hiddenLayerValue = new double[argNumHidden+1];
        outputLayerValue = new double[argNumOutput];

        addBiasToVectors(inputLayerValue);

        weightChgInToHi = new double[argNumInputs+1][argNumHidden];
        weightChgHiToOut = new double[argNumHidden+1][argNumOutput];

        errorSigHidden = new double[argNumHidden];
        errorSigOutput = new double[argNumOutput];
    }

    /**
     * add bias to the input XOR training set matrix
     * @param inputVectors The input XOR training set matrix
     */
    public void addBiasToVectors(double[][] inputVectors){
        int rowNumber = inputVectors.length;
        int columnNumber = inputVectors[0].length;
        inputLayerValue = new double[rowNumber][columnNumber+1];
        for (int i = 0; i < rowNumber; i++){
            for (int j = 0; j < columnNumber; j++){
                inputLayerValue[i][j] = inputVectors[i][j];
            }
            inputLayerValue[i][columnNumber] = bias;
        }
    }

    /**
     * Return a bipolar sigmoid of the input X
     * @param x The input
     * @return f(x) = 2 / (1+e(-x)) - 1
     */
    @Override
    public double sigmoid(double x) {
        return 2.0 / (1.0 + Math.exp(-x)) - 1.0;
    }

    /**
     * This method implements a general sigmoid with asymptotes bounded by (a,b)
     * @param x The input
     * @return f(x) = b_minus_a / (1 + e(-x)) - minus_a
     */
    @Override
    public double customSigmoid(double x) {
        return (argB - argA) / (1.0 + Math.exp(-x)) - (-argA);
    }

    /**
     * Initialize the weights to random values.
     * For say 2 inputs, the input vector is [0] & [1]. We add [2] for the bias.
     * Like wise for hidden units. For say 2 hidden units which are stored in an array.
     * [0] & [1] are the hidden & [2] the bias.
     * We also initialise the last weight change arrays. This is to implement the alpha term.
     */
    @Override
    public void initializeWeights() {
        for(int i = 0; i < argNumInputs+1; i++) {
            for(int j = 0; j < argNumHidden; j++) {
                // initialize input-to-hidden weights to random values in the range -0.5 to +0.5
                weightsFromInToHi[i][j] = (Math.random() * (weightsRangeUpper - weightsRangeLower)) + weightsRangeLower;
            }
        }

        for(int i = 0; i < argNumHidden+1; i++) {
            for(int j = 0; j < argNumOutput; j++) {
                // initialize hidden-to-output weights to random values in the range -0.5 to +0.5
                weightsFromHiToOut[i][j] = (Math.random() * (weightsRangeUpper - weightsRangeLower)) + weightsRangeLower;
            }
        }
    }

    /**
     * @param rowNum The row number of the matrix
     * @param colNum The column number of the matrix
     * @param weights The weight matrix
     * @param updateValue The update value for weight matrix
     * @param weightsChange The weight changed matrix
     * @param updateChangeValue The update value for weight changed matrix
     */
    public void zeroWeightsHelper(int rowNum, int colNum, double[][] weights, double updateValue, double[][] weightsChange, double updateChangeValue) {
        for(int i = 0; i < rowNum; i++) {
            for(int j = 0; j < colNum; j++) {
                // set layer weights to 0
                weights[i][j] = updateValue;
                // set layer weight changes to 0
                weightsChange[i][j] = updateChangeValue;
            }
        }
    }

    /**
     * initialize weight matrix and weight changed matrix to 0
     */
    @Override
    public void zeroWeights() {
        zeroWeightsHelper(argNumInputs+1, argNumHidden, weightsFromInToHi, 0.0, weightChgInToHi, 0.0);
        zeroWeightsHelper(argNumHidden+1, argNumOutput, weightsFromHiToOut, 0.0, weightChgHiToOut, 0.0);
    }

    public void forwardPropagation(double[] inputLayer) {
        for(int i = 0; i < argNumHidden; i++) {
            hiddenLayerValue[i] = 0.0;
            for(int j = 0; j < argNumInputs+1; j++) {
                hiddenLayerValue[i] += inputLayer[j] * weightsFromInToHi[j][i];
            }
            hiddenLayerValue[i] = customSigmoid(hiddenLayerValue[i]);
        }

        hiddenLayerValue[argNumHidden] = bias;

        for(int i = 0; i < argNumOutput; i++) {
            outputLayerValue[i] = 0.0;
            for(int j = 0; j < argNumHidden+1; j++) {
                outputLayerValue[i] += hiddenLayerValue[j] * weightsFromHiToOut[j][i];
            }
            outputLayerValue[i] = customSigmoid(outputLayerValue[i]);
        }
    }

    public double sigmoidDerivative(double value) {
        if(!isBipolar) {
            // if Binary: y_j * (1 - y_j)
            return value * (1 - value);
        } else {
            // if Bipolar: 1/2 * (1 - y_j^2)
            return 0.5 * (1 - Math.pow(value, 2));
        }
    }

    public void backPropagation(double[] inputExpected, double[] inputLayer) {
        // initial all hidden error Signals δ to 0
        for(int i = 0; i < argNumHidden; i++) {
            errorSigHidden[i] = 0.0;
        }

        // initial all output error Signals δ to 0
        for(int i = 0; i < argNumOutput; i++) {
            errorSigOutput[i] = 0.0;
        }

        // back propagation from output to hidden
        for(int i = 0; i < argNumOutput; i++) {
            // if Binary: δ_j = y_j * (1 - y_j) * (C_j - y_j)
            // if Bipolar: δ_j = 1/2 * (1 - y_j^2) * (C_j - y_j)
            errorSigOutput[i] = (inputExpected[i] - outputLayerValue[i]) * sigmoidDerivative(outputLayerValue[i]);

            for(int j = 0; j < argNumHidden+1; j++) {
                // delta_w_ji = argMomentumTerm * delta_w_ji + argLearningRate * δ_j * x_i
                weightChgHiToOut[j][i] = argMomentumTerm * weightChgHiToOut[j][i] + argLearningRate * errorSigOutput[i] * hiddenLayerValue[j];
                // w_ji = w_ji + delta_w_ji
                weightsFromHiToOut[j][i] += weightChgHiToOut[j][i];
            }
        }

        // back propagation from hidden to input
        for(int i = 0; i < argNumHidden; i++) {
            for(int j = 0; j < argNumOutput; j++) {
                // if Binary: δ_j = y_j * (1 - y_j) * sum_h(δ_h * w_hj)
                // if Bipolar: δ_j = 1/2 * (1 - y_j^2) * sum_h(δ_h * w_hj)
                errorSigHidden[i] += errorSigOutput[j] * weightsFromHiToOut[i][j];
            }
            errorSigHidden[i] *= sigmoidDerivative(hiddenLayerValue[i]);

            for(int j = 0; j < argNumInputs + 1; j++) {
                // delta_w_ji = argMomentumTerm * delta_w_ji + argLearningRate * δ_j * x_i
                weightChgInToHi[j][i] = argMomentumTerm * weightChgInToHi[j][i] + argLearningRate * errorSigHidden[i] * inputLayer[j];
                // w_ji = w_ji + delta_w_ji
                weightsFromInToHi[j][i] += weightChgInToHi[j][i];
            }
        }
    }

    /**
     * @return The epoch number which just makes the total error less than 0.05
     */
    public int trainTest() {
        int epoch = 0;
        double totalError = 0.0;
        zeroWeights();
        initializeWeights();

        do {
            totalError = 0.0;
            for(int inputIndex = 0; inputIndex < inputLayerValue.length; inputIndex++) {
                totalError += train(inputLayerValue[inputIndex], expectedOutput[inputIndex][0]);
                backPropagation(expectedOutput[inputIndex], inputLayerValue[inputIndex]);
            }
            totalError /= 2;
            epoch++;
            System.out.println("epoch: " + epoch + " total error: " + totalError);
        } while(totalError > acceptError);
        return epoch;
    }

    /**
     * @param X The input vector. An array of doubles.
     * @return The value returned by th LUT or NN for this input vector
     */
    @Override
    public double outputFor(double[] X) {
        forwardPropagation(X);
        return outputLayerValue[0];
    }

    /**
     * This method will tell the NN or the LUT the output
     * value that should be mapped to the given input vector. I.e.
     * the desired correct output value for an input.
     * @param X The input vector
     * @param argValue The new value to learn
     * @return The error in the output for that input vector
     */
    @Override
    public double train(double[] X, double argValue) {
        return Math.pow((outputFor(X) - argValue), 2);
    }

    /**
     * A method to help save the weight matrices to the file.
     * @param rowNum The matrix's row number.
     * @param colNum The matrix's column number.
     * @param weightsMatrix The weight matrices which need to be saved.
     * @param fileWriter The writing tool.
     */
    public void saveHelper(int rowNum, int colNum, double[][] weightsMatrix, FileWriter fileWriter) throws IOException {
        String weight = "";
        for(int i = 0; i < rowNum; i++) {
            for(int j = 0; j < colNum; j++) {
                weight = weightsMatrix[i][j] + " ";
                fileWriter.write(weight);
            }
            fileWriter.write("\n");
        }
    }

    /**
     * A method to write either a LUT or weights of an neural net to a file.
     * @param argFile of type File.
     */
    @Override
    public void save(File argFile) {
        String path = argFile.getAbsolutePath();
        String wtFromInToHi = "The weights between input layer and hidden layer.\n";
        String wtFromHiToOut = "The weights between hidden layer and output layer.\n";

        try{
            FileWriter fileWriter = new FileWriter(path);
            fileWriter.write(wtFromInToHi);
            fileWriter.write(Integer.toString(argNumInputs+1) + " " + Integer.toString(argNumHidden) + "\n");
            saveHelper(argNumInputs+1, argNumHidden, weightsFromInToHi, fileWriter);

            fileWriter.write(wtFromHiToOut);
            fileWriter.write(Integer.toString(argNumHidden+1) + " " + Integer.toString(argNumOutput) + "\n");
            saveHelper(argNumHidden+1, argNumOutput, weightsFromHiToOut, fileWriter);
            fileWriter.close();
        } catch(IOException e) {
            System.out.println("IO Exception: " + e);
        }
    }

    /**
     * A method to help load the weight matrices from the file to the variable.
     * @param rowNum The matrix's row number.
     * @param colNum The matrix's column number.
     * @param weightsMatrix The weight matrices which need to be saved.
     * @param bufferedReader The data from the buffer
     */
    public void loadHelper(int rowNum, int colNum, double[][] weightsMatrix, BufferedReader bufferedReader) throws IOException {
        String line = "";
        for(int i = 0; i < rowNum; i++) {
            line = bufferedReader.readLine();
            String[] splitLine = line.split(" ");
            for(int j = 0; j < colNum; j++) {
                weightsMatrix[i][j] = Double.parseDouble(splitLine[j]);
            }
        }
    }

    /**
     * A method to help handle the error message.
     * @param correctRowNum The correct row number of the matrix.
     * @param correctColNum The correct column number of the matrix.
     * @param bufferedReader Reading data from the file.
     */
    public void loadErrorMsg(int correctRowNum, int correctColNum, BufferedReader bufferedReader) throws IOException {
        String line = bufferedReader.readLine();
        String[] splitLine = line.split("\\s+");
        int loadRowNum = Integer.parseInt(splitLine[0]);
        int loadColNum = Integer.parseInt(splitLine[1]);
        if(loadRowNum != correctRowNum || loadColNum  != correctColNum) {
            System.err.println("The neural net whose structure does not match the data in the file.");
            System.exit(1);
        }
    }

    /**
     * Loads the LUT or neural net weights from file. The load must of course
     * have knowledge of how the data was written out by the save method.
     * You should raise an error in the case that an attempt is being
     * made to load data into an LUT or neural net whose structure does not match
     * the data in the file. (e.g. wrong number of hidden neurons).
     * @throws IOException
     */
    @Override
    public void load(String argFileName) throws IOException {
        BufferedReader bufferedReader = new BufferedReader(new FileReader(argFileName));

        String line = bufferedReader.readLine();
        System.out.println("Reading: " + line);
        loadErrorMsg(argNumInputs+1, argNumHidden, bufferedReader);
        loadHelper(argNumInputs+1, argNumHidden, weightsFromInToHi, bufferedReader);

        line = bufferedReader.readLine();
        System.out.println("Reading: " + line);
        loadErrorMsg(argNumHidden+1, argNumOutput, bufferedReader);
        loadHelper(argNumHidden+1, argNumOutput, weightsFromHiToOut, bufferedReader);
        bufferedReader.close();

        System.out.println();
        for(int i = 0; i < argNumInputs+1; i++) {
            String newLine = "";
            for(int j = 0; j < argNumHidden; j++) {
                newLine = newLine.concat(weightsFromInToHi[i][j]+" ");
            }
            System.out.println(newLine);
        }

        System.out.println();
        for(int i = 0; i < argNumHidden+1; i++) {
            String newLine = "";
            for(int j = 0; j < argNumOutput; j++) {
                newLine = newLine.concat(weightsFromHiToOut[i][j] + " ");
            }
            System.out.println(newLine);
        }
        System.out.println();
    }
}
