package NeuralNetwork;

import pacman.game.util.IO;

import javax.swing.*;
import java.math.BigDecimal;
import java.math.RoundingMode;

/**
 * Created by ramonserranolopez on 24/11/16.
 */
public class NeuralNetwork {

    private static String fileName = "dataset.txt";
    private static float[][] dataset;

    private Neuron[] _hiddenLayer;
    private Neuron _outputLayer;

    private float weightIncrementsIH;
    private float weightIncrementsHO;

    private float _learningRate = 0.01f;

    public NeuralNetwork(int inputNeurons, int hiddenNeurons, int outputNeurons) {

        weightIncrementsIH = 0f;
        weightIncrementsHO = 0f;

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

        boolean classifedCorrectly = true;

        do {
            for(float[] tuple : dataset) {

                //TODO: cambiar para que coja las entradas de forma dimanica y la salida
                //HACK: aqui seteo a pelo
                float[] inputs = {tuple[0], tuple[1]};
                float expetedOutput = tuple[2];

                //calcular la salida de la red neuronal (feedfordward)
                float output = forwardPropagation(inputs);

                if(!isCorrectlyClassified(output, expetedOutput, 1)) {
                    classifedCorrectly = false;

                    //calcular los errores de la capa de salida
                    _outputLayer.calculateError(expetedOutput);

                    float hiddenErrorSumatory = 0;

                    //calcular Δwho para los pesos de todas conexiones entre
                    //la capa oculta y la capa de salida (who)
                    for (int i = 0; i < _hiddenLayer.length; i++) {
                        weightIncrementsHO += _learningRate * _outputLayer.getError() * _hiddenLayer[i].getExit();
                        hiddenErrorSumatory += _outputLayer.getError() * _outputLayer.getWeights()[i];
                    }

                    //calcular los errores de la capa oculta
                    for(Neuron neuron : _hiddenLayer) {
                        neuron.calculateError(hiddenErrorSumatory);
                    }

                    //calcular Δwih para los pesos de todas conexiones entre
                    //la capa de entrada y la capa oculta(wih).
                    for (int i = 0; i < _hiddenLayer.length; i++) {
                        for(int j = 0; j < inputs.length; j++) {
                            weightIncrementsIH += _learningRate * _hiddenLayer[i].getError() * inputs[j];
                        }
                    }
                }
            }
            //TODO: Hacerlo dinamico
            //actualizar los pesos de la red who y wih
            UpdateWeights(2);
        } while (!classifedCorrectly);
        System.out.println("END TRAINING");
    }

    public float forwardPropagation (float [] _inputLayer) {

        float[] hiddenLayerExits = new float[_hiddenLayer.length];

        for(int i = 0; i < _hiddenLayer.length; i++) {
            hiddenLayerExits[i] = _hiddenLayer[i].feedForward(_inputLayer);
        }

        return _outputLayer.feedForward(hiddenLayerExits);
    }

    public void UpdateWeights(int inputLength) {
        for (int i = 0; i < _hiddenLayer.length; i++) {
            float value = _outputLayer.getWeights()[i] + weightIncrementsHO;
            _outputLayer.setWeight(i,value);
        }

        for (int i = 0; i < _hiddenLayer.length; i++) {
            for(int j = 0; j < inputLength; j++) {
                float value = _hiddenLayer[i].getWeights()[j] + weightIncrementsIH;
                _hiddenLayer[i].setWeight(j,value);
            }
        }
    }

    public boolean isCorrectlyClassified(float output, float expetedOutput, int n) {
        return round(output, n) == round(expetedOutput, n);
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
