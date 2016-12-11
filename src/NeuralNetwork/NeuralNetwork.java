package NeuralNetwork;

import pacman.game.util.IO;

import java.math.BigDecimal;
import java.math.RoundingMode;

/**
 * Created by ramonserranolopez on 24/11/16.
 */
public class NeuralNetwork {

    private static String fileName = "logicOR.txt";
    private static float[][] dataset;

    private Neuron[] _hiddenLayer;
    private Neuron _outputLayer;

    private float _weightIncrementsIH;
    private float _weightIncrementsHO;

    private float _learningRate = 0.01f;

    private int _epochs;

    public NeuralNetwork(int inputNeurons, int hiddenNeurons, int outputNeurons) {

        _weightIncrementsIH = 0f;
        _weightIncrementsHO = 0f;
        _epochs = 0;

        _hiddenLayer = new Neuron[hiddenNeurons];

        for(int i = 0; i < hiddenNeurons; i++) {
            _hiddenLayer[i] = new Neuron(inputNeurons, LayerType.HIDDEN);
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

        int errors = 0;

        // Repetir mientras no se cumpla condición de parada
        do {
            errors = 0;
            for(float[] tuple : dataset) {

                //TODO: cambiar para que coja las entradas de forma dimanica y la salida
                //HACK: aqui seteo a pelo
                float[] inputs = {tuple[0], tuple[1]};
                float expetedOutput = tuple[2];

                //Repetir hasta que no se cumpla la condicion de parada!
                //Inicializar Δwih y Δwho a 0
                _weightIncrementsIH = 0f;
                _weightIncrementsHO = 0f;

                //calcular la salida de la red neuronal (feedfordward)
                float output = forwardPropagation(inputs);

                if(!isCorrectlyClassified(output, expetedOutput, 2)) {
                    errors++;

                    //calcular los errores de la capa de salida
                    _outputLayer.calculateError(expetedOutput);



                    //calcular Δwho para los pesos de todas conexiones entre
                    //la capa oculta y la capa de salida (who)
                    for (int i = 0; i < _hiddenLayer.length; i++) {
                        float outputError = _outputLayer.getError();
                        float hiddenNeuronExit = _hiddenLayer[i].getExit();
                        _weightIncrementsHO += _learningRate * outputError * hiddenNeuronExit;
                    }

                    //calcular los errores de la capa oculta
                    float hiddenErrorSumatory = 0;

                    for (int i = 0; i < _hiddenLayer.length; i++) {
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
                            _weightIncrementsIH += _learningRate * _hiddenLayer[i].getError() * inputs[j];
                        }
                    }
                }
            }

            //actualizar los pesos de la red who y wih
            UpdateWeights(2);
            _epochs++;
        } while (isClassified(errors, 30));
        System.out.println("END TRAINING");
    }

    public boolean isClassified(int errors, float validPercentage) {
        float errorAmountNormalized = (float) errors / dataset.length;
        float percentage = errorAmountNormalized * 100f;

        System.out.print("\r" + "Epoch amount: " + _epochs + " - Percentage Error: " + percentage + "%");
        return (percentage <= validPercentage) ? false : true;
    }

    public float forwardPropagation (float [] _inputLayer) {

        float[] hiddenLayerExits = new float[_hiddenLayer.length];

        for(int i = 0; i < _hiddenLayer.length; i++) {
            hiddenLayerExits[i] = _hiddenLayer[i].feedForward(_inputLayer);
        }

        return _outputLayer.feedForward(hiddenLayerExits);
    }

    public void UpdateWeights(int inputLength) {
        // Hacemos update del peso del bias de cada neurona de la capa oculta y de la neurona de salida
        for (int i = 0; i < _outputLayer.getWeights().length; i++) {
            float value = _outputLayer.getWeights()[i] + _weightIncrementsHO;
            _outputLayer.setWeight(i,value);
        }

        for (int i = 0; i < _hiddenLayer.length; i++) {
            for(int j = 0; j < _hiddenLayer[i].getWeights().length; j++) {
                float value = _hiddenLayer[i].getWeights()[j] + _weightIncrementsIH;
                _hiddenLayer[i].setWeight(j,value);
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
