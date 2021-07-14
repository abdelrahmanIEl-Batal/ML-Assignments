import java.util.ArrayList;
import java.util.Arrays;

public class Matrix {
    double [][] data;
    int row,col;
    double max, min;   // will be used to denormalize

    Matrix(double [][] arr){
        this.row = arr.length;
        this.col = arr[0].length;
        data = new double[row][col];
        for(int i=0;i<row;++i){
            for(int j = 0; j< col; ++j) this.data[i][j] = arr[i][j];
        }
    }

    Matrix(Matrix a){
        this.row = a.row;
        this.col = a.col;
        data = new double[row][col];
        for(int i=0;i<this.row;++i){
            for(int j=0;j<this.col;++j) this.data[i][j] = a.data[i][j];
        }
    }
    Matrix(double [] a){    // takes 1d array and return a 2d matrix n row and 1 column
        this.row = a.length;
        this.col = 1;
        this.data = new double[row][col];
        for(int i=0;i<row;++i) this.data[i][0] = a[i];
    }

    Matrix(int r, int c){
        this.row = r;
        this.col = c;
        data = new double[row][col];
        for(int i=0;i<this.row;++i)
            for(int j=0; j<this.col;++j) this.data[i][j] = 0;
    }

    public void Random(){
        for(int i=0;i<row;++i)
            for(int j=0;j<col;++j) this.data[i][j] = Math.random();
    }

    public void add(double value){
        for(int i=0;i<this.row;++i){
            for(int j=0;j<this.col;++j) this.data[i][j] += value;
        }
    }

    public void addBiasColumn(){
        Matrix temp = new Matrix(this.row,this.col);
        for(int i=0;i<this.row;++i){
            for(int j=0; j <this.col; ++j) temp.data[i][j] = this.data[i][j];
        }
        this.col = this.col + 1;
        this.data = new double[row][col];
        for(int i=0;i<this.row;++i) this.data[i][0] = 1;
        for(int i=0;i<this.row;++i){
            for(int j = 1; j < this.col; ++j) this.data[i][j] = temp.data[i][j-1];
        }
    }

    public void gaussianNormalization(){
        for(int i=0;i<this.col;++i){
            double [] temp = new double[this.row];

            for(int j=0;j<this.row;++j){
                temp[j] = this.data[j][i];
            }
            double mean = calculateMean(temp);
            double std = calculateSD(temp, mean);
            for(int j=0;j<this.row;++j){
                this.data[j][i] = (this.data[j][i] - mean) / std;
            }
        }
    }

    public void MinMaxNormalization(){
        for(int i=0;i<this.col;++i){
            double minimum = Double.MAX_VALUE;
            double mx = Double.MIN_VALUE;
            for(int j=0;j<this.row;++j){
                minimum = Math.min(minimum, this.data[j][i]);
                mx = Math.max(mx, this.data[j][i]);
            }
            this.max = mx;
            this.min = minimum;
            for(int j=0;j<this.row;++j){
                this.data[j][i] = (this.data[j][i] - min) / (this.max - this.min);
            }
        }
    }

    private double calculateMean(double [] a){
        double sum = 0.0;
        for(double num : a) sum+= num;
        return sum/(a.length*1.0);
    }

    private double calculateSD(double [] a, double mean){
        double res = 0.0;
        for (double v : a) res += Math.pow(mean - v, 2);
        return Math.sqrt(res/a.length);
    }

    public static Matrix add(Matrix a, Matrix b) throws WrongDimensionException{
        if(a.col!=b.col || a.row!=b.row){
            throw new WrongDimensionException("Dimensions does not match for adding\nDimensions of both matrices must be the same\n");
        }
        Matrix result = new Matrix(a.row, a.col);
        for(int i=0;i<a.row;++i){
            for(int j=0;j<a.col;++j){
                result.data[i][j] = a.data[i][j] + b.data[i][j];
            }
        }
        return result;
    }

    public void add(Matrix a) throws WrongDimensionException{
        if(a.col!=this.col || a.row!=this.row){
            throw new WrongDimensionException("Dimensions does not match for adding\nDimensions of both matrices must be the same\n");
        }
        for(int i=0;i<this.row; ++i){
            for(int j=0;j<this.col;++j) this.data[i][j]+= a.data[i][j];
        }
    }
    public static Matrix subtract(Matrix a, Matrix b) throws WrongDimensionException{
        if(a.col!=b.col || a.row!=b.row){
            throw new WrongDimensionException("Dimensions does not match for subtracting\nDimensions of both matrices must be the same\n");
        }
        Matrix result = new Matrix(a.row, a.col);
        for(int i=0;i<a.row;++i){
            for(int j=0;j<a.col;++j){
                result.data[i][j] = a.data[i][j] - b.data[i][j];
            }
        }
        return result;
    }

    public void subtract(Matrix a){
        for(int i=0;i<this.row;++i){
            for(int j=0;j<this.col;++j) this.data[i][j] -= a.data[i][j];
        }
    }
    public static Matrix multiply(Matrix a, Matrix b) throws WrongDimensionException{
        if(a.col!=b.row){
            throw new WrongDimensionException("Dimensions does not match for multiplying\nDimensions of 1st Matrix col and 2nd Matrix row must match" +
                    "\nFound " + a.col + " and " + b.row +"\n");
        }
        Matrix result = new Matrix(a.row, b.col);
        for(int i=0;i<a.row; ++i){
            for(int j=0;j<b.col;++j){
                for(int k = 0; k < a.col; ++k) result.data[i][j]+= a.data[i][k] * b.data[k][j];
            }
        }
        return result;
    }

    public void multiply(Matrix a) throws WrongDimensionException{
        if(a.col!=this.col || a.row!=this.row){
            throw new WrongDimensionException("Dimensions must be the same\n");
        }
        for(int i=0;i<this.row;++i){
            for(int j=0;j<this.col;++j) this.data[i][j] = this.data[i][j] * a.data[i][j];
        }
    }

    public void multiply(double val){
        for(int i=0;i<this.row; ++i){
            for(int j=0;j<this.col;++j){
                this.data[i][j] *= val;
            }
        }
    }

    public static Matrix transpose(Matrix a){
        Matrix res = new Matrix(a.col, a.row);
        for(int i=0;i<a.row;++i){
            for(int j=0;j<a.col;++j){
                res.data[j][i] = a.data[i][j];
            }
        }
        return res;
    }

    public void sigmoid(){
        for(int i=0;i<this.row;++i)
            for(int j=0;j<this.col;++j) this.data[i][j] = 1/(1+Math.exp(-1 * this.data[i][j]));
    }

    public Matrix sigmoidDerivative(){
        Matrix result = new Matrix(this.row, this.col);
        for(int i=0;i<this.row; ++i){
            for(int j=0; j <this.col; ++j){
                result.data[i][j] = this.data[i][j] * (1 - this.data[i][j]);
            }
        }
        return result;
    }

    public void print(){
        System.out.println("--------------------");
        for(int i=0;i<this.row;++i){
            for(int j=0;j<this.col;++j) System.out.print(this.data[i][j] + " ");
            System.out.println();
        }
    }

    public void square(){
        for(int i=0;i<this.row;++i){
            for(int j=0;j<this.col;++j) this.data[i][j] = Math.pow(this.data[i][j], 2);
        }
    }

    public void shape(){
        System.out.println("(" + row +", "+col+ ")");
    }
}
