package NeuralNetwork;

import pacman.game.util.IO;

/**
 * Created by ramonserranolopez on 24/11/16.
 */
public class NeuralNetwork {

    private static String fileName = "dataset.txt";
    private static float[][] dataset;

    private Neuron[] _hiddenLayer;
    private Neuron[] _outputLayer;

    public NeuralNetwork(int inputNeurons, int hiddenNeurons, int outputNeurons) {

        //_inputLayer = new float[inputNeurons];
        _hiddenLayer = new Neuron[hiddenNeurons];
        _outputLayer = new Neuron[outputNeurons];

        /*for(Neuron neuron : _hiddenLayer) {
            neuron = new Neuron(inputNeurons);
        }*/
        for(int i = 0; i < hiddenNeurons; i++) {
            _hiddenLayer[i] = new Neuron(inputNeurons);
        }
        for(Neuron neuron : _outputLayer) {
            neuron = new Neuron(hiddenNeurons);
        }

        dataset = LoadPacmanData();
    }

    public void feedForward (float [] _inputLayer) {

        float[] exitsHide = new float[_hiddenLayer.length];

        for(int i = 0; i < _hiddenLayer.length; i++) {
            exitsHide[i] = _hiddenLayer[i].feedForward(_inputLayer);
        }

        for(Neuron neuron : _outputLayer) {
            neuron.feedForward((exitsHide));
        }
    }

    public static void main (String [ ] args) {
        int input = 2;
        int hide = 2 * input + 1;
        int output = 1;
        NeuralNetwork neuralNetwork = new NeuralNetwork(input, hide, output);
        /*for(float[] tuple : dataset) {
            neuralNetwork.feedForward(tuple);
        }*/

        for (int i = 0; i < dataset.length; i++) {
            float[] inputs = {dataset[i][0], dataset[i][1]};
            neuralNetwork.feedForward(inputs);
        }
        System.out.printf("End Forward");
    }

    public float[][] LoadPacmanData () {
        String data = IO.loadFile(fileName);
        String[] dataLine = data.split("\n");

        float[][] dataset = new float[dataLine.length][3];

        for(int i = 0; i < dataLine.length; i++) {

            String[] lineSplited = new String[3];
            lineSplited = dataLine[i].split(",");

            for (int j = 0; j < 3; j++) {
                dataset[i][j] = Float.parseFloat((lineSplited[j]));
            }
        }

        return dataset;
    }
}
