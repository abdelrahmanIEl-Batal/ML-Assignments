import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

public class Main {


    public static void main(String[] args) throws WrongDimensionException, IOException {


        File file = new File("train.txt");
        BufferedReader br = new BufferedReader(new FileReader(file));
        String st;
        st = br.readLine();
        String [] parameters = st.split(" ");
        int m = Integer.parseInt(parameters[0]);
        int l = Integer.parseInt(parameters[1]);
        int n = Integer.parseInt(parameters[2]);
        st = br.readLine();
        int trainingExamples = Integer.parseInt(st);

        // m is number of features
        double [][] X = new double[trainingExamples][m];
        double [][] Y = new double[trainingExamples][n];
        int x = 0, y;
        while ((st = br.readLine()) != null) {
            parameters = st.split(" ");
            ArrayList<String> nums = new ArrayList<>();
            for(String s : parameters) if(!(s.equals(""))) nums.add(s);
            y = 0;
            for(int i=0; i < nums.size() - n; ++i) X[x][y++] = Double.parseDouble(nums.get(i));
            y = 0;
            for(int i=nums.size() - n; i < nums.size(); ++i) Y[x][y++] = Double.parseDouble(nums.get(i));
            x++;
        }

        //System.out.println(m + " " + l + " " + n);

        NeuralNetwork neuralNetwork = new NeuralNetwork(m,l,n);
        neuralNetwork.runNN(X, Y);
        //neuralNetwork.predict(new double[]{1, 331.0  ,  0.0   ,  0.0 ,  192.0  ,   0.0 ,  1025.0  , 821.0  , 7 }); // real output is 17.44
        //neuralNetwork.predict(new double[]{1, 296.0  ,  0.0  ,   0.0  , 192.0 ,    0.0  , 1085.0  , 765.0  ,    7 }); // 14.2
        neuralNetwork.predict(new double[]{1, 469.0 , 117.2  ,   0.0  , 137.8  ,  32.2  ,  852.1 ,  840.5  ,   28});  // 66.9
    }
}
