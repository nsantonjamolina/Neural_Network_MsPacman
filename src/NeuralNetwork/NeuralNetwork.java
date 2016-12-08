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

        _hiddenLayer = new Neuron[hiddenNeurons];
        _outputLayer = new Neuron[outputNeurons];

        /*for(Neuron neuron : _hiddenLayer) {
            neuron = new Neuron(inputNeurons);
        }*/
        for(int i = 0; i < hiddenNeurons; i++) {
            _hiddenLayer[i] = new Neuron(inputNeurons);
        }
        for(int i = 0; i < outputNeurons; i++) {
            _outputLayer[i] = new Neuron(hiddenNeurons);
        }
        if(dataset == null) {
            dataset = LoadPacmanData();
        }
    }
    

    public void forwardPropagation (float [] _inputLayer) {

        float[] hiddenLayerExits = new float[_hiddenLayer.length];

        for(int i = 0; i < _hiddenLayer.length; i++) {
            hiddenLayerExits[i] = _hiddenLayer[i].feedForward(_inputLayer);
        }

        float[] outputLayerExits = new float[_outputLayer.length];

        for(int i = 0; i < _outputLayer.length; i++) {
            outputLayerExits[i] = _outputLayer[i].feedForward(hiddenLayerExits);
        }

        System.out.printf("End forward propagation");
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

    public void Backpropagation() {

    }

    public void UpdateWeights() {

    }

    public void Training() {
        /*forwardPropagation();
        Backpropagation();
        UpdateWeights();*/
    }


    public static void main (String [ ] args) {
        int input = 2;
        int hide = 2 * input + 1;
        int output = 1;
        NeuralNetwork neuralNetwork = new NeuralNetwork(input, hide, output);

        for (int i = 0; i < dataset.length; i++) {
            float[] inputs = {dataset[i][0], dataset[i][1]};
            neuralNetwork.forwardPropagation(inputs);
        }
        System.out.printf("End Forward");
    }
}
