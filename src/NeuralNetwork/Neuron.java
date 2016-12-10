package NeuralNetwork;

import java.util.Random;

/**
 * Created by ramonserranolopez on 24/11/16.
 */

enum LayerType {
    HIDDEN,
    OUTPUT
}
public class Neuron {

    //El ultimo peso es el del bias
    private float[] _weights;
    private float _exit;
    private float _error;
    private LayerType _currentLayer;

    private static int bias = 1;

    public Neuron(int inputs, LayerType type) {

        _weights = new float[inputs + 1];
        _exit = 0f;
        _error = 0f;
        _currentLayer = type;

        for(int i = 0; i < _weights.length; i++) {
            float randValue = randomFloat(-1,1);
            _weights[i] = randValue;
        }
    }

    public float[] multiplication(float[] _inputs) {

        float[] products = new float[_inputs.length + 1];

        for(int i = 0; i < _inputs.length; i++) {
            float productValue = _inputs[i] * _weights[i];
            products[i]  = productValue;
        }

        products[_inputs.length] = bias * _weights[_inputs.length];

        return products;
    }

    public float sumatory(float[] inputs) {

        float sumatory = 0;
        float[] products = multiplication(inputs);

        for(float product : products) {
            sumatory += product;
        }

        return sumatory;
    }

    public float sigmoidalActivation(float x) {
        float gValue = (float) (1/(1 + Math.pow(Math.E, -x)));
        return  gValue;
    }

    public float feedForward(float[] inputs) {
        _exit = sigmoidalActivation(sumatory(inputs));
        return _exit;
    }

    public void setWeight(int pos, float value) {
        if(pos >= 0 && pos < _weights.length) {
            _weights[pos] = value;
        }
    }

    public float[] getWeights() { return  _weights; }

    public float getExit() {
        return _exit;
    }

    public float getError() { return _error; }

    public void calculateError(float x) {
        if(_currentLayer == LayerType.OUTPUT) {
            //_error = derivedSigmoidalActivation(_exit) * (1 - _exit) * (x - _exit);
            _error = _exit * (1 - _exit) * (x - _exit);
        } else {
            //_error = derivedSigmoidalActivation(_exit) * x;
            _error = _exit * (1 - _exit) * x;
        }
    }

    public static float randomFloat(float min, float max) {
        Random random = new Random();
        float randomFl = (random.nextFloat() * (min - max) - min);
        return randomFl;
    }
}
