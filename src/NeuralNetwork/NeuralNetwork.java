package NeuralNetwork;

/**
 * Created by ramonserranolopez on 24/11/16.
 */
public class NeuralNetwork {

    private float[] _inputLayer;
    private Neuron[] hiddenLayer;
    private Neuron[] _outputLayer;

    public NeuralNetwork(int inputNeurons, int hiddenNeurons, int outputNeurons) {

        _inputLayer = new float[inputNeurons];
        hiddenLayer = new Neuron[hiddenNeurons];
        _outputLayer = new Neuron[outputNeurons];

        for(Neuron neuron : hiddenLayer) {
            neuron = new Neuron(inputNeurons);
        }
        for(Neuron neuron : _outputLayer) {
            neuron = new Neuron(hiddenNeurons);
        }
    }

    public void feedForward () {

        float[] exitsHide = new float[hiddenLayer.length];

        for(int i = 0; i < hiddenLayer.length; i++) {
            exitsHide[i] = hiddenLayer[i].feedForward(_inputLayer);
        }

        for(Neuron neuron : _outputLayer) {
            neuron.feedForward((exitsHide));
        }
    }

    public static void main (String [ ] args) {
        int input = 3;
        int hide = 2 * input + 1;
        int output = 1;
        NeuralNetwork neuralNetwork = new NeuralNetwork(input, hide, output);
    }
}
