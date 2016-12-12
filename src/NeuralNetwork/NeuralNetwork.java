package NeuralNetwork;

import pacman.game.util.IO;

import java.math.BigDecimal;
import java.math.RoundingMode;

/**
 * Created by ramonserranolopez on 24/11/16.
 */
public class NeuralNetwork {

    private static String fileName = "dataset.txt";
    private static float[][] dataset;

    private int _inputNeurons;
    private Neuron[] _hiddenLayer;
    private Neuron _outputLayer;

    private float[][] _weightIncrementsIH;
    private float[] _weightIncrementsHO;

    private float _learningRate = 0.2f;

    private int _epochs;

    public NeuralNetwork(int inputNeurons, int hiddenNeurons, int outputNeurons) {

        _inputNeurons = inputNeurons;
        _weightIncrementsIH = new float[hiddenNeurons][_inputNeurons + 1];
        _weightIncrementsHO = new float[hiddenNeurons + 1];
        _epochs = 0;

        _hiddenLayer = new Neuron[hiddenNeurons];

        for(int i = 0; i < hiddenNeurons; i++) {
            _hiddenLayer[i] = new Neuron(_inputNeurons, LayerType.HIDDEN);
        }

        _outputLayer = new Neuron(hiddenNeurons, LayerType.OUTPUT);

        if(dataset == null) {
            dataset = LoadPacmanData();
        }
    }

    public void Train() {
        BackPropagation(dataset, _learningRate);
    }

    public void BackPropagation(float[][] dataset, float learningRate) {
        //TODO: Falta inicializarlo aqui para poder ir porbando diferentes learning rates hasta encontrar el optimo
        //HACK: lo hemos puesto en el constructor solo se va a hacer una vez por tanto no podemos probar diferentes learning rates
        // Inicializar wih y who a valores aleatorios pequeños

        float error = Float.MAX_VALUE;
        float lastError = Float.MAX_VALUE;
        float errorCuadraticoMedio = 0;

        // Repetir mientras no se cumpla condición de parada
        do {
            lastError = error;
            error = 0;

            //Inicializar Δwih y Δwho a 0
            _weightIncrementsIH = new float[_hiddenLayer.length][_hiddenLayer[0].getWeights().length];
            _weightIncrementsHO = new float[_outputLayer.getWeights().length];

            for(float[] tuple : dataset) {

                //TODO: cambiar para que coja las entradas de forma dimanica y la salida
                //HACK: aqui seteo a pelo
                float[] inputs = {tuple[0], tuple[1]};
                float expetedOutput = tuple[2];

                //calcular la salida de la red neuronal (feedfordward)
                float output = forwardPropagation(inputs);

                //TODO: Usar el error cuadrático medio
                /*if(!isCorrectlyClassified(output, expetedOutput, 2)) {
                    errors++;
                }*/

                //calcular los errores de la capa de salida
                _outputLayer.calculateError(expetedOutput);
                error += Math.pow((expetedOutput - output),2);


                //calcular Δwho para los pesos de todas conexiones entre
                //la capa oculta y la capa de salida (who)
                for (int i = 0; i < _hiddenLayer.length; i++) {
                    float outputError = _outputLayer.getError();
                    float hiddenNeuronExit = _hiddenLayer[i].getExit();
                    _weightIncrementsHO[i] += _learningRate * outputError * hiddenNeuronExit;
                }

                _weightIncrementsHO[_hiddenLayer.length] += _learningRate * _outputLayer.getError() * Neuron.getBias();


                //calcular los errores de la capa oculta
                float hiddenErrorSumatory = 0;

                for (int i = 0; i < _outputLayer.getWeights().length; i++) {
                    float outputError = _outputLayer.getError();
                    float weightHO = _outputLayer.getWeights()[i];
                    hiddenErrorSumatory += outputError * weightHO;
                }

                for(Neuron neuron : _hiddenLayer) {
                    neuron.calculateError(hiddenErrorSumatory);
                }

                //calcular Δwih para los pesos de todas conexiones entre
                //la capa de entrada y la capa oculta(wih).
                for (int i = 0; i < _hiddenLayer.length; i++) {
                    for(int j = 0; j < inputs.length; j++) {
                        _weightIncrementsIH[i][j] += _learningRate * _hiddenLayer[i].getError() * inputs[j];
                    }
                    _weightIncrementsIH[i][inputs.length] += _learningRate * _hiddenLayer[i].getError() * Neuron.getBias();
                }

                System.out.println();
            }

            //actualizar los pesos de la red who y wih
            UpdateWeights();
            _epochs++;

            errorCuadraticoMedio = (float) (error/dataset.length);
            System.out.print("\r" + "Epoch amount: " + _epochs + " - ECM: " + errorCuadraticoMedio /* + " - Last error " + lastError*/);
        } while (errorCuadraticoMedio < lastError);
        System.out.println("END TRAINING");
    }

    public boolean isClassified(float errors, float validPercentage) {
        float errorAmountNormalized = (float) errors / dataset.length;
        float percentage = errorAmountNormalized * 100f;


        return (percentage <= validPercentage) ? false : true;
    }

    public float forwardPropagation (float [] _inputLayer) {

        float[] hiddenLayerExits = new float[_hiddenLayer.length];

        for(int i = 0; i < _hiddenLayer.length; i++) {
            hiddenLayerExits[i] = _hiddenLayer[i].feedForward(_inputLayer);
        }

        return _outputLayer.feedForward(hiddenLayerExits);
    }

    public void UpdateWeights() {
        // Hacemos update del peso del bias de cada neurona de la capa oculta y de la neurona de salida
        for (int i = 0; i < _outputLayer.getWeights().length; i++) {
            if(_weightIncrementsHO[i] != 0){
                float weight = _outputLayer.getWeights()[i];
                float weightIncrement = _weightIncrementsHO[i];
                float value = weight + weightIncrement;
                _outputLayer.setWeight(i,value);
            }
        }

        for (int i = 0; i < _hiddenLayer.length; i++) {
            for(int j = 0; j < _hiddenLayer[i].getWeights().length; j++) {
                if(_weightIncrementsIH[i][j] != 0) {
                    float weight = _hiddenLayer[i].getWeights()[j];
                    float weightIncrement = _weightIncrementsIH[i][j];
                    float value = weight + weightIncrement;
                    _hiddenLayer[i].setWeight(j,value);
                }
            }
        }
    }

    public boolean isCorrectlyClassified(float output, float expetedOutput, int n) {
        return round(output, n) == round(expetedOutput, n);
        //return output == expetedOutput;
    }

    public float[][] LoadPacmanData () {
        String data = IO.loadFile(fileName);
        String[] dataLines = data.split("\n");

        String[] lineSplited = dataLines[0].split(",");
        float[][] dataset = new float[dataLines.length][lineSplited.length];

        for(int i = 0; i < dataLines.length; i++) {

            lineSplited = dataLines[i].split(",");

            for (int j = 0; j < lineSplited.length; j++) {
                dataset[i][j] = Float.parseFloat((lineSplited[j]));
            }
        }

        return dataset;
    }

    public static double round(double value, int places) {
        if (places < 0) throw new IllegalArgumentException();

        BigDecimal bd = new BigDecimal(value);
        bd = bd.setScale(places, RoundingMode.HALF_UP);
        return bd.doubleValue();
    }


    public static void main (String [ ] args) {
        int input = 2;
        int hide = 2 * input + 1;
        int output = 1;

        NeuralNetwork neuralNetwork = new NeuralNetwork(input, hide, output);
        neuralNetwork.Train();

    }
}
