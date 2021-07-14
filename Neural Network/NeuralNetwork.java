import java.awt.image.AreaAveragingScaleFilter;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;

public class NeuralNetwork {
    Matrix weightInputHidden, weightHiddenOutput;
    private final double learningRate = 0.1;
    private final int iterations = 500;
    ArrayList<Double> errorPerIteration;
    double min, max;
    NeuralNetwork(int inputNodes, int hiddenNodes, int outputNodes){
        // + 1 for the added bias in input
        weightInputHidden = new Matrix(hiddenNodes, inputNodes + 1);
        errorPerIteration = new ArrayList<>();
        weightInputHidden.Random();      // initialize weight with random values

        weightHiddenOutput = new Matrix(outputNodes, hiddenNodes);
        weightHiddenOutput.Random();
        /*
        weightHiddenOutput.data[0][0] = 1;
        weightHiddenOutput.data[0][1] = 0.8;

        weightInputHidden.data[0][0] = 0.3;
        weightInputHidden.data[0][1] = -0.9;
        weightInputHidden.data[0][2] = 1;
        weightInputHidden.data[1][0] = -1.2;
        weightInputHidden.data[1][1] = 1;
        weightInputHidden.data[1][2] = 1;

         */
    }

    // this will do forward pass and BACK PROPAGATION algorithms on each training example
    void fit(double [] X, double [] Y) throws WrongDimensionException {

        // forward pass
        Matrix input = new Matrix(X);          // takes a row vector and converts it to a col vector
        Matrix hiddenLayer = Matrix.multiply(weightInputHidden, input);
        hiddenLayer.sigmoid();
        Matrix outputLayer = Matrix.multiply(weightHiddenOutput,hiddenLayer);
        outputLayer.sigmoid();


        Matrix target = new Matrix(Y);


        // back propagation
        Matrix error = Matrix.subtract(outputLayer, target);
        Matrix MSE = new Matrix(error);
        MSE.square();
        double sum = 0;
        for(int i=0;i<MSE.row;++i){
            sum+= MSE.data[i][0];
        }
        errorPerIteration.add(sum/2);
        Matrix gradient = outputLayer.sigmoidDerivative();   // derivative out activated outputLayer
        gradient.multiply(error);                            // element wise multiplication of outputLayer error
        Matrix gradient2 = new Matrix(gradient);
        gradient2.multiply(learningRate);                     // scalar multiplication

        // transpose of the layer before output(hidden) multiplied by our gradient
        Matrix deltaHiddenOutput =  Matrix.multiply(gradient2, Matrix.transpose(hiddenLayer));


        //weightHiddenOutput.subtract(deltaHiddenOutput);  // update weights connecting hidden layer with output layer

        // output error multiplied by transpose of weight that connects output layer with hidden layer
        // sum 1 to k ( delta output subscript k * weight output subscript kj)
        //error.shape();
        //weightHiddenOutput.shape();
        Matrix hidden_errors = Matrix.multiply(Matrix.transpose(weightHiddenOutput), gradient);

        // derivative of activated hidden layer
        Matrix hiddenLayerGradient = hiddenLayer.sigmoidDerivative();
        // element wise multiplication
        hiddenLayerGradient.multiply(hidden_errors);
        // scalar multiplication

        hiddenLayerGradient.multiply(learningRate);

        Matrix deltaInputHidden = Matrix.multiply(hiddenLayerGradient, Matrix.transpose(input));


        weightInputHidden.subtract(deltaInputHidden);
        weightHiddenOutput.subtract(deltaHiddenOutput);

        //weightInputHidden.print();
        //weightHiddenOutput.print();
    }

    public void predict(double [] X) throws WrongDimensionException {
       // weightInputHidden.print();
        //weightHiddenOutput.print();
        Matrix input = new Matrix(X);
        Matrix hidden = Matrix.multiply(weightInputHidden, input);
        hidden.sigmoid();

        Matrix output = Matrix.multiply(weightHiddenOutput,hidden);
        output.sigmoid();
        // inverse of min-max normalization
        output.multiply((this.max - this.min));
        output.add(this.min);
        output.print();
        // return output;
    }

    void runNN(double [][] input, double [][] output) throws WrongDimensionException, IOException {
        Matrix X = new Matrix(input);
        Matrix Y = new Matrix(output);
        X.gaussianNormalization();
        X.addBiasColumn();
        Y.MinMaxNormalization();

        this.max = Y.max;
        this.min = Y.min;

        for(int i=0; i < this.iterations; ++i){
            errorPerIteration.clear();
            for(int index = 0; index < X.row -3 ; ++index){
                fit(X.data[index], Y.data[index]);
            }
            double sum = 0;
            for(int j=0;j<errorPerIteration.size();++j) sum+= errorPerIteration.get(j);
            //System.out.println(sum/(X.row - 3));
        }

        double error = 0;
        for(int j=0;j<errorPerIteration.size();++j) error+=errorPerIteration.get(j);
        System.out.println("Error after " + iterations + " iterations is: " + error/(X.row-3));

        BufferedWriter writer = new BufferedWriter(new FileWriter("weights.txt", true));
        writer.write("The weights connecting output to hidden layer:\n");
        for(int i=0;i<weightHiddenOutput.row;++i){
            for(int j=0;j<weightHiddenOutput.col;++j){
                writer.write(String.valueOf(weightHiddenOutput.data[i][j]) + " ");
            }
            writer.write("\n");
        }
        writer.write("\n");
        writer.write("The weights connecting hidden layer to input layer:\n");
        for(int i=0;i<weightInputHidden.row;++i){
            for(int j=0;j<weightInputHidden.col;++j){
                writer.write(String.valueOf(weightInputHidden.data[i][j]) + " ");
            }
            writer.write("\n");
        }
        writer.write("\n");
        writer.close();
    }
}
