package NeuralNetwork;

/**
 * Created by ramonserranolopez on 24/11/16.
 */
public class NeuralNetwork {

    private float[] _inputLayer;
    private Neuron[] _hideLayer;
    private Neuron[] _outputLayer;

    public NeuralNetwork(int neuronsInput, int neuronsHide, int neuronsOutput) {

        _inputLayer = new float[neuronsInput];
        _hideLayer = new Neuron[neuronsHide];
        _outputLayer = new Neuron[neuronsOutput];

        for(Neuron neuron : _hideLayer) {
            neuron = new Neuron(neuronsInput);
        }
        for(Neuron neuron : _outputLayer) {
            neuron = new Neuron(neuronsHide);
        }
    }

    public void feedForward () {

        float[] exitsHide = new float[_hideLayer.length];

        for(int i = 0; i < _hideLayer.length; i++) {
            exitsHide[i] = _hideLayer[i].feedForward(_inputLayer);
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
